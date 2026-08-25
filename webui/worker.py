"""
The single serial job worker for the WHYcast web UI (ADR-008, TASK-003 phase 2).

``python -m webui.worker`` is one long-lived process that claims queued jobs and
runs them **one at a time, in a child process**:

    claim_next -> spawn `python -m webui.runner <id>` -> mark_running(pid)
               -> wait, polling for cancel -> finish(status)

Three invariants, and why each one lives here:

* **Strictly serial.** ADR-008: one GPU, so at most one job at a time - GPU or
  not (Robert, 2026-08-24). The queue deliberately does not enforce this; a
  single-threaded worker holding a lock file does. No thread pool, no parallel
  lanes, no "cheap jobs may overtake".
* **Crash isolation.** The job runs in a child process, never inside this loop
  and never inside the web server. A CUDA out-of-memory kill takes down that
  child; the worker sees a non-zero exit code and records a failed job.
* **A job always ends.** Every path through :func:`_run_one` - normal exit,
  cancel, Ctrl+C, a database error mid-wait - lands in a ``finally`` that kills
  the process tree and writes a terminal status. A torch process that outlives
  its job row holds the GPU forever, which is the one failure this module exists
  to prevent.

Killing means killing the *tree*. The runner spawns ffmpeg, and torch spawns
data-loader workers; killing only the runner would leave them holding the GPU.
Windows: ``taskkill /F /T /PID`` after ``CREATE_NEW_PROCESS_GROUP``. POSIX:
``killpg`` after ``start_new_session``.

Pid liveness is checked with ``OpenProcess``/``GetExitCodeProcess`` on Windows,
never ``os.kill(pid, 0)``: on Windows :func:`os.kill` calls ``TerminateProcess``
for any signal other than the two console events, so the usual POSIX liveness
idiom would *kill* the process it was asking about.

No ``print()`` (ADR-008): the worker's own progress goes through
:mod:`whycast.events` into :class:`LoggingSink`, and only ``__main__`` exits.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import platform
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from whycast.errors import WhycastError
from whycast.events import ProgressEvent, emit, use_sink
from whycast.io_utils import sweep_stale_temps

from webui import db as webui_db
from webui import jobs
from webui.runner import (
    EXIT_BAD_JOB,
    EXIT_CANCELLED,
    db_path_from_env,
    output_dir_from_env,
)

__all__ = [
    "EXIT_OK",
    "EXIT_REFUSED",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_CANCEL_GRACE",
    "LOCK_SUFFIX",
    "WorkerAlreadyRunning",
    "LoggingSink",
    "WorkerLock",
    "lock_path",
    "read_lock_pid",
    "process_alive",
    "process_command_line",
    "is_runner_for_job",
    "kill_process_tree",
    "reap_stale_jobs",
    "run_worker",
    "main",
]

logger = logging.getLogger(__name__)

EXIT_OK = 0
#: The worker refused to start (another worker holds the lock, or the database
#: could not be opened). Nothing was claimed.
EXIT_REFUSED = 2

#: How often the queue is polled when it is empty, and how often a running
#: child is checked for a cancel request.
DEFAULT_POLL_INTERVAL = 1.0

#: How long a cancelled child is given to stop by itself before its process tree
#: is killed. The runner polls for cancel between steps, so a job that is
#: between steps stops cleanly inside this window; a job that is deep inside
#: torch never will, and gets the axe.
DEFAULT_CANCEL_GRACE = 5.0

#: Appended to the queue database path to name the worker's lock file. The lock
#: must be keyed on the queue, not on the log directory: see :func:`lock_path`.
LOCK_SUFFIX = ".worker.lock"

#: How long after a claim a ``running`` row with no pid is left alone. The
#: worker claims, spawns, then calls ``mark_running``; a reaper that judges the
#: row inside that window destroys a job that was about to start. The claim now
#: stamps the worker's own pid, so this only ever covers rows written by an
#: older build - but it is three lines and the failure it prevents is silent
#: job loss.
_SPAWN_GRACE_SECONDS = 30.0

#: How much of the child's console log is quoted into the job's error column
#: when the child died without recording a reason of its own.
_LOG_TAIL_CHARS = 600

#: A live orphan blocks the queue; say so at most this often instead of once a
#: second forever.
_ORPHAN_WARNING_INTERVAL = 60.0

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

_IS_WINDOWS = os.name == "nt"

# OpenProcess / GetExitCodeProcess constants (Windows).
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _kernel32():
    """The Windows kernel32 binding used for pid liveness, set up once.

    Two details that are bugs when left to ctypes' defaults:

    * ``use_last_error=True``. Without it, ``ctypes.get_last_error()`` reads a
      thread-local that ctypes never fills in, so the "access denied means it
      exists" branch below would never fire.
    * ``OpenProcess.restype = c_void_p``. The default return type is a 32-bit
      ``c_int``, which truncates a 64-bit HANDLE - the truncated value then
      fails ``CloseHandle`` and leaks a handle per call.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32


#: Bound once at import on Windows; ``None`` everywhere else.
_KERNEL32 = _kernel32() if _IS_WINDOWS else None

#: taskkill's "the process is not running" exit code. Not an error: the child
#: died between our check and our kill, which is the outcome we wanted anyway.
_TASKKILL_NOT_FOUND = 128


class WorkerAlreadyRunning(WhycastError):
    """Another worker process holds the lock file.

    Raised by :func:`run_worker`, not printed and not exited on: only
    ``__main__`` turns it into a non-zero exit code. Two workers would mean two
    jobs on one GPU, which ADR-008 exists to prevent.
    """


class LoggingSink:
    """An :class:`whycast.events.EventSink` that forwards to :mod:`logging`.

    The worker's own console output. Deliberately not
    :class:`whycast.events.ConsoleSink`, which uses ``print()`` - banned in the
    web layer by the ADR-008 Decision Contract. Job progress does not come
    through here at all: that happens in the child, into the job's
    ``events.jsonl``.
    """

    _LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    def __init__(self, target: Optional[logging.Logger] = None) -> None:
        self.logger = target or logger

    def emit(self, event: ProgressEvent) -> None:
        self.logger.log(
            self._LEVELS.get(event.level, logging.INFO), "[%s] %s", event.step, event.message
        )


# ---------------------------------------------------------------------------
# Process primitives
# ---------------------------------------------------------------------------


def process_alive(pid: Optional[int]) -> bool:
    """Whether a process with this pid exists right now.

    Windows uses ``OpenProcess`` + ``GetExitCodeProcess`` rather than
    ``os.kill(pid, 0)``: Python's :func:`os.kill` on Windows calls
    ``TerminateProcess`` for every signal except ``CTRL_C_EVENT`` and
    ``CTRL_BREAK_EVENT``, so the POSIX liveness idiom would kill the process it
    is asking about. Both steps are needed - a handle can still be opened for a
    process that has exited but not been reaped, and ``GetExitCodeProcess`` is
    what distinguishes it.

    Two honest caveats. A process that exited with code 259 reads as alive
    (``STILL_ACTIVE`` is 259; the API cannot tell them apart). And a pid may
    have been reused by an unrelated process - which is exactly why nothing in
    this module kills a pid it did not spawn.
    """
    if pid is None or not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False

    if not _IS_WINDOWS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # It exists; we are simply not allowed to signal it.
            return True
        except OSError:  # pragma: no cover - defensive
            return False
        return True

    kernel32 = _KERNEL32
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not handle:
        # Access denied means the process exists and belongs to someone else;
        # every other error (invalid parameter, in practice) means it does not.
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False  # pragma: no cover - defensive
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def process_command_line(pid: int, timeout: float = 10.0) -> Optional[str]:
    """The command line of a running process, or ``None`` if it cannot be read.

    ``None`` is not "no command line" - it is "this machine would not tell me",
    and every caller here must treat it as a refusal, never as a mismatch or a
    match.

    Windows has no cheap syscall for this without a dependency (the real answer
    lives in the target process's PEB), so it asks WMI: ``wmic`` first because
    it is roughly twice as fast, then PowerShell's ``Get-CimInstance``, since
    ``wmic`` is deprecated and already absent from some Windows 11 installs.
    POSIX reads ``/proc/<pid>/cmdline`` and falls back to ``ps``.

    Only ever called on the crash-recovery path (an orphaned runner someone
    asked to cancel), so a second or so of subprocess is not on any hot path.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None

    if not _IS_WINDOWS:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as stream:
                raw = stream.read()
            if raw:
                return " ".join(raw.decode("utf-8", "replace").split("\0")).strip()
        except OSError:
            pass
        return _run_text(["ps", "-p", str(pid), "-o", "args="], timeout)

    text = _run_text(
        ["wmic", "process", "where", f"processid={pid}", "get", "commandline", "/value"],
        timeout,
    )
    if text:
        for line in text.splitlines():
            key, _, value = line.partition("=")
            if key.strip().lower() == "commandline" and value.strip():
                return value.strip()

    text = _run_text(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
        ],
        timeout,
    )
    return text or None


def _run_text(command: List[str], timeout: float) -> Optional[str]:
    """Run a read-only query command and return its stdout, or ``None``."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,  # no print() reaches the console from here
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or "").strip()
    return output or None


def is_runner_for_job(pid: int, job_id: str) -> bool:
    """Whether ``pid`` really is ``python -m webui.runner <job_id>``.

    The guard on killing a process this worker did not spawn. A pid alone
    proves nothing - Windows reuses pids, and this machine handed the same pid
    to two unrelated processes minutes apart during testing - so a job row's
    ``pid`` column is a hint, not an identity. The command line is the identity:
    the runner is spawned as ``python -m webui.runner <job_id>``, and that job
    id is a ``uuid4().hex``, so a process whose command line carries both the
    module and the id is that job's runner beyond reasonable doubt.

    Returns False when the command line cannot be read at all. Refusing to kill
    is always the recoverable answer; killing the wrong process is not.
    """
    cmdline = process_command_line(pid)
    if not cmdline:
        return False
    lowered = cmdline.lower()
    return "webui.runner" in lowered and job_id.lower() in lowered


def kill_process_tree(pid: int, timeout: float = 10.0) -> bool:
    """Kill a process **and everything it spawned**. Returns True when it is gone.

    The runner's children are the reason this is a tree kill and not a
    :meth:`subprocess.Popen.terminate`: transcription runs ffmpeg, and torch
    starts worker processes of its own. Killing only the parent leaves them
    holding VRAM, and the next job then fails with an out-of-memory error that
    has nothing to do with the next job.

    Windows: ``taskkill /F /T /PID``, which walks the child tree the kernel
    records. Its exit code 128 means "no such process", which is success here.
    POSIX: ``SIGTERM`` to the process group (the runner is spawned with
    ``start_new_session``, so it *is* a group leader), then ``SIGKILL`` to
    whatever is left.

    Called for a process this worker spawned, whose pid it still holds open -
    so there is no window in which the pid could have been reused - or, on the
    crash-recovery path, for an orphaned runner whose identity
    :func:`is_runner_for_job` has confirmed from its command line. Never for a
    bare pid out of a database row.
    """
    if not process_alive(pid):
        return True

    if _IS_WINDOWS:
        try:
            completed = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,  # taskkill prints "SUCCESS:"; no print() here
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            emit("worker", f"taskkill for pid {pid} failed: {exc}", level="error", pid=pid)
            return not process_alive(pid)
        if completed.returncode not in (0, _TASKKILL_NOT_FOUND):
            emit(
                "worker",
                f"taskkill /F /T /PID {pid} exited {completed.returncode}: "
                f"{(completed.stderr or completed.stdout or '').strip()}",
                level="warning",
                pid=pid,
                returncode=completed.returncode,
            )
    else:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except (ProcessLookupError, PermissionError, OSError):
                break
            if _wait_gone(pid, min(timeout / 2, 5.0)):
                return True

    return _wait_gone(pid, timeout)


def _wait_gone(pid: int, timeout: float) -> bool:
    """Wait up to ``timeout`` seconds for a pid to disappear."""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        if not process_alive(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# The single-instance lock
# ---------------------------------------------------------------------------


def lock_path() -> str:
    """``<queue database>.worker.lock`` - next to the queue it protects.

    The lock has to name the *resource*, and the resource is the queue, not the
    log directory. It used to be ``<jobs_dir>/worker.lock``, built from
    ``WHYCAST_JOBS_DIR``, while the queue comes from ``WHYCAST_WEBUI_DB``. Those
    are independent variables, so two workers pointed at one database with
    different job directories each took their own lock and both claimed from
    the same ``jobs`` table - measured: 20 jobs drained by two workers, 11 and
    9. That is exactly the state ADR-008's "at most one job at a time" exists to
    prevent, and the enforcement mechanism was simply pointed at the wrong file.

    Deriving it from the database path also means the two travel together by
    construction: point ``WHYCAST_WEBUI_DB`` at a scratch queue and you get a
    scratch lock with it, with no second flag to remember.
    """
    return db_path_from_env() + LOCK_SUFFIX


class WorkerLock:
    """Exclusive lock file holding the worker's pid.

    Acquired with ``O_CREAT | O_EXCL``, which is atomic on Windows and POSIX
    alike: exactly one process can create the file.

    A **stale** lock - one whose pid is no longer alive, or whose contents are
    unreadable - may be taken over, because a worker killed with ``taskkill``
    never gets to clean up and the operator should not have to delete a file to
    restart it. Takeover is deliberately "remove, then create exclusively
    again": if two workers both decide the lock is stale, both remove it and
    exactly one wins the re-creation. The loser refuses to start rather than
    assuming it won.

    This is a pid file, not a kernel lock, so one narrow race survives: two
    workers can pass the liveness check in the same instant that a third writes
    a fresh lock. The write-then-verify step below closes the common case; on a
    single-user localhost machine the remainder is theoretical. A kernel-held
    lock (``msvcrt.locking``) would close it completely at the cost of a
    platform-specific handle that has to stay open for the worker's lifetime -
    not worth it here.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = os.path.abspath(path or lock_path())
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> str:
        """Take the lock. Raises :class:`WorkerAlreadyRunning` if someone has it."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        for attempt in (1, 2):
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                holder = read_lock_pid(self.path)
                if holder is not None and holder != os.getpid() and process_alive(holder):
                    raise WorkerAlreadyRunning(
                        f"Another WHYcast worker is already running (pid {holder}, "
                        f"lock {self.path}). Only one worker may run at a time: two "
                        f"would put two jobs on one GPU. Stop that one first, or "
                        f"delete the lock file if you are sure it is dead."
                    )
                if attempt == 2:
                    raise WorkerAlreadyRunning(
                        f"Another worker took the lock {self.path} while this one "
                        f"was reclaiming it. Refusing to start."
                    )
                emit(
                    "worker",
                    f"Lock {self.path} is stale ("
                    + (f"pid {holder} is not running" if holder else "no readable pid")
                    + "); taking it over.",
                    level="warning",
                    lock=self.path,
                    stale_pid=holder,
                )
                try:
                    os.remove(self.path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise WorkerAlreadyRunning(
                        f"Could not reclaim the stale lock {self.path}: {exc}"
                    ) from exc
                continue

            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(f"{os.getpid()}\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Verify we still own what we just wrote: another worker reclaiming
            # the same stale lock would have overwritten it.
            if read_lock_pid(self.path) != os.getpid():
                raise WorkerAlreadyRunning(
                    f"Another worker overwrote the lock {self.path} immediately "
                    f"after this one took it. Refusing to start."
                )
            self._held = True
            return self.path
        raise WorkerAlreadyRunning(  # pragma: no cover - the loop always returns
            f"Could not take the worker lock {self.path}."
        )

    def release(self) -> None:
        """Drop the lock, if we still hold it. Never raises."""
        if not self._held:
            return
        self._held = False
        try:
            # Only remove a lock that is still ours: if a takeover happened we
            # would otherwise delete the new worker's lock on the way out.
            if read_lock_pid(self.path) == os.getpid():
                os.remove(self.path)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not remove the worker lock %s: %s", self.path, exc)

    def __enter__(self) -> "WorkerLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def read_lock_pid(path: str) -> Optional[int]:
    """The pid in a lock file, or ``None`` when it is missing or unreadable.

    Unreadable counts as stale on purpose: an empty file is what a worker leaves
    behind when it is killed between creating the lock and writing to it.
    """
    try:
        with open(path, "r", encoding="utf-8") as stream:
            raw = stream.read(64).strip()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        pid = int(raw)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def reap_stale_jobs(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Settle jobs left ``running`` by a worker that died. Returns live orphans.

    Four cases, in the order they are checked:

    * **A parked verdict.** The job finished, but the database refused the
      write and :func:`webui.jobs.finish_or_record` left the verdict in the
      job's directory. That verdict is the truth and is applied here. Checked
      first, because the alternative is writing the opposite of what happened.
    * **A live pid that was asked to stop.** An orphaned runner someone
      cancelled from the dashboard. Its identity is confirmed from its command
      line (:func:`is_runner_for_job`) and then its process tree is killed and
      the row settled as ``cancelled``. This is what makes the dashboard's
      Cancel button a real remedy for a job that is too deep inside torch to
      poll for it - before, that button set a flag nothing would ever act on
      and the queue stayed frozen forever.
    * **A live pid nobody asked to stop.** A runner from a previous worker,
      possibly mid-transcription. Left strictly alone and returned to the
      caller: claiming a second job would put two on the GPU. The worker
      stalls, loudly, and says which process to stop.
    * **A dead pid.** Nothing is going to finish it. Settled ``cancelled`` if a
      cancel had been requested (the user pressed Cancel and the process is
      gone - that is a cancel, not a crash), otherwise ``failed``.
    """
    live: List[Dict[str, Any]] = []
    for job in jobs.list_jobs(conn, status="running", limit=None):
        pid = job["pid"]
        settled = _apply_pending_verdict(conn, job)

        if process_alive(pid):
            # A live pid stops this worker claiming, full stop - even when the
            # verdict above just settled the row. "At most one job at a time"
            # (ADR-008) is about processes, not about rows, and a runner that
            # has recorded its result may still be tearing down its CUDA
            # context. This costs at most one poll: the row is terminal now, so
            # the next pass does not see it at all.
            if not settled and job["cancel_requested"] and _may_retry_stop(job["id"]):
                if _stop_orphan(conn, job):
                    _ORPHAN_REFUSALS.pop(job["id"], None)
                    continue
                _ORPHAN_REFUSALS[job["id"]] = time.monotonic()
            live.append(job)
            continue

        if settled:
            continue
        if pid is None and _within_spawn_grace(job):
            # Claimed moments ago by a worker that has not called mark_running
            # yet. Judging it now is how a job gets destroyed before it runs.
            continue
        _reap_dead(conn, job)
    return live


#: Job ids whose orphan could not be identified, and when that was last tried.
#: Reading a process's command line costs a subprocess, and the reaper runs on
#: every poll: without this, an orphan whose pid cannot be confirmed would
#: spawn a WMI query and log the same refusal once a second for as long as it
#: lives. Process-local and deliberately not persisted - a fresh worker should
#: try again.
_ORPHAN_REFUSALS: Dict[str, float] = {}


def _may_retry_stop(job_id: str) -> bool:
    """Whether to try identifying this orphan again. The first try is always on."""
    last = _ORPHAN_REFUSALS.get(job_id)
    return last is None or (time.monotonic() - last) >= _ORPHAN_WARNING_INTERVAL


def _kill_hint(pid: int) -> str:
    """The exact command an operator should paste to stop a process tree."""
    return f"taskkill /F /T /PID {pid}" if _IS_WINDOWS else f"kill -9 {pid}"


def _within_spawn_grace(job: Dict[str, Any]) -> bool:
    """True while a pid-less ``running`` row is still young enough to be starting."""
    started = job.get("started_at")
    if not isinstance(started, (int, float)):
        return False
    return (time.time() - started) < _SPAWN_GRACE_SECONDS


def _apply_pending_verdict(conn: sqlite3.Connection, job: Dict[str, Any]) -> bool:
    """Apply a verdict parked on disk. True when this row is accounted for.

    True even when the write fails again, which is the point: a row that has a
    parked verdict must never fall through to "the worker running this job
    died". The verdict on disk is what happened; the reaper's guess is not, and
    a wrong verdict in the one record that cannot be rebuilt from disk is worse
    than a row that stays ``running`` until the database recovers.
    """
    verdict = jobs.read_pending_verdict(job["id"])
    if verdict is None:
        return False
    if jobs.finish_or_record(
        conn,
        job["id"],
        verdict["status"],
        exit_code=verdict["exit_code"],
        error=verdict["error"],
    ):
        emit(
            "worker",
            f"Job {job['id']} had recorded a {verdict['status']} verdict that "
            f"the database would not take at the time; applied it now.",
            level="warning",
            job_id=job["id"],
            status=verdict["status"],
            exit_code=verdict["exit_code"],
        )
    return True


def _stop_orphan(conn: sqlite3.Connection, job: Dict[str, Any]) -> bool:
    """Kill a cancelled orphan's process tree and settle it. True when settled.

    The identity check is the whole point. A job row's ``pid`` is a hint: pids
    are reused, and this machine reused one within minutes during testing. So
    the pid is only killed once its command line proves it is
    ``python -m webui.runner <this job id>``. If that cannot be read, nothing
    is killed and the operator is told exactly what to do by hand - refusing is
    recoverable, killing the wrong process is not.
    """
    job_id = job["id"]
    pid = job["pid"]
    if not is_runner_for_job(pid, job_id):
        emit(
            "worker",
            f"Job {job_id} was cancelled, but pid {pid} could not be confirmed "
            f"as its runner (its command line does not name webui.runner "
            f"{job_id}, or could not be read). Not killing it: a pid can be "
            f"reused by an unrelated process. Check pid {pid} yourself and stop "
            f"it if it is this job's runner.",
            level="warning",
            job_id=job_id,
            pid=pid,
        )
        return False

    emit(
        "worker",
        f"Job {job_id} was cancelled and its runner (pid {pid}) is still "
        f"running from an earlier worker. Confirmed it from its command line; "
        f"stopping its process tree.",
        level="warning",
        job_id=job_id,
        pid=pid,
    )
    if not kill_process_tree(pid):
        emit(
            "worker",
            f"Could not stop pid {pid} for cancelled job {job_id}; it is still "
            f"running. Stop it by hand: {_kill_hint(pid)}",
            level="error",
            job_id=job_id,
            pid=pid,
        )
        return False

    return jobs.finish_or_record(
        conn,
        job_id,
        "cancelled",
        exit_code=job["exit_code"],
        error=(
            f"Cancelled. Its runner (pid {pid}) had been orphaned by a worker "
            f"that died; the next worker confirmed the process and stopped it."
        ),
    )


def _reap_dead(conn: sqlite3.Connection, job: Dict[str, Any]) -> None:
    """Settle a ``running`` row whose process is gone."""
    job_id = job["id"]
    pid = job["pid"]
    cancelled = bool(job["cancel_requested"])
    where = f"(pid {pid}) is gone" if pid else "was never recorded"
    _ORPHAN_REFUSALS.pop(job_id, None)

    if cancelled:
        # Not a crash: someone pressed Cancel and the process is gone. Calling
        # that "the worker died" put a wrong status and a false cause into the
        # one record that cannot be rebuilt from disk.
        status = "cancelled"
        error = (
            f"Cancelled. Its process "
            + (f"(pid {pid}) " if pid else "")
            + "stopped after the cancel was requested, without recording a "
            "verdict of its own. See the job's events.jsonl for how far it got."
        )
    else:
        status = "failed"
        error = (
            f"The worker running this job died. Its process "
            + (f"(pid {pid}) " if pid else "")
            + "is no longer running, so no result was ever recorded. "
            "Check the job's events.jsonl for how far it got"
            + (
                " before re-running it - this job type spends money and, for "
                "force_episode, deletes the episode's artifacts first."
                if job["cost"]
                else ", then run it again."
            )
        )

    emit(
        "worker",
        f"Job {job_id} is marked running but its process {where}"
        + (
            "; it had been cancelled. Marking it cancelled."
            if cancelled
            else "; the worker that started it must have died. Marking it failed."
        ),
        level="warning",
        job_id=job_id,
        pid=pid,
    )
    try:
        jobs.finish_or_record(conn, job_id, status, exit_code=job["exit_code"], error=error)
    except ValueError as exc:  # pragma: no cover - defensive
        logger.error("Could not reap stale job %s: %s", job_id, exc)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_worker(
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    once: bool = False,
    cancel_grace: float = DEFAULT_CANCEL_GRACE,
    lock: Optional[WorkerLock] = None,
) -> None:
    """Claim and run jobs, one at a time, until interrupted.

    Args:
        poll_interval: seconds between queue polls when idle, and between cancel
            checks while a job runs.
        once: claim and run a single job, then return. Returns immediately when
            the queue is empty. This is what the tests drive.
        cancel_grace: seconds a cancelled child is given to stop by itself
            before its process tree is killed.
        lock: an alternative :class:`WorkerLock` (tests point it at a temporary
            directory). Defaults to ``<jobs_dir>/worker.lock``.

    Raises:
        WorkerAlreadyRunning: another worker holds the lock. Nothing was
            claimed; the caller decides what to do about it (``__main__`` exits
            with :data:`EXIT_REFUSED`).
        sqlite3.Error: the queue database could not be opened.
    """
    if poll_interval <= 0:
        raise ValueError(f"poll_interval must be positive, got {poll_interval!r}.")

    worker_id = f"{platform.node() or 'worker'}-{os.getpid()}"
    guard = lock or WorkerLock()

    with use_sink(LoggingSink()):
        with guard:
            conn = webui_db.init_db(db_path_from_env())
            try:
                jobs.ensure_job_schema(conn)
                emit(
                    "worker",
                    f"Worker {worker_id} ready; queue in {db_path_from_env()}, "
                    f"lock {guard.path}, artifacts in {output_dir_from_env()}, "
                    f"job logs in {jobs.jobs_dir()}"
                    + (" (single job, then exit)" if once else ""),
                    worker_id=worker_id,
                    pid=os.getpid(),
                    once=once,
                )
                _loop(conn, worker_id, poll_interval, once, cancel_grace)
            finally:
                emit("worker", f"Worker {worker_id} stopped.", worker_id=worker_id)
                try:
                    conn.close()
                except sqlite3.Error:  # pragma: no cover - defensive
                    pass


def _loop(
    conn: sqlite3.Connection,
    worker_id: str,
    poll_interval: float,
    once: bool,
    cancel_grace: float,
) -> None:
    """Claim-run-repeat. Returns after one job when ``once``."""
    last_orphan_warning = 0.0
    while True:
        orphans = reap_stale_jobs(conn)
        if orphans:
            now = time.monotonic()
            if now - last_orphan_warning >= _ORPHAN_WARNING_INTERVAL:
                last_orphan_warning = now
                emit(
                    "worker",
                    "Not claiming any job: "
                    + ", ".join(
                        f"{job['id']} (pid {job['pid']})" for job in orphans
                    )
                    + " is still running from an earlier worker. At most one job "
                    "may run at a time (ADR-008), so the queue waits. Two ways "
                    "out: cancel the job from the dashboard - this worker then "
                    "confirms that pid really is its runner and stops the whole "
                    "process tree - or stop it yourself with "
                    + ", ".join(_kill_hint(job["pid"]) for job in orphans)
                    + ".",
                    level="warning",
                    orphan_job_ids=[job["id"] for job in orphans],
                )
            if once:
                return
            time.sleep(poll_interval)
            continue

        try:
            job = jobs.claim_next(conn, worker_id)
        except (WhycastError, sqlite3.Error) as exc:
            emit("worker", f"Could not claim a job: {exc}", level="error")
            if once:
                raise
            time.sleep(poll_interval)
            continue

        if job is None:
            if once:
                return
            time.sleep(poll_interval)
            continue

        try:
            _run_one(conn, job, poll_interval, cancel_grace)
        except Exception as exc:
            # _run_one has already killed the child and settled the job row in
            # its finally block; this is about whether the *worker* survives.
            # It does: one broken job must not end the queue.
            emit(
                "worker",
                f"Worker error while running job {job['id']}: {exc}",
                level="error",
                job_id=job["id"],
            )
            logger.exception("Worker error while running job %s", job["id"])
            if once:
                raise
        if once:
            return


def _run_one(
    conn: sqlite3.Connection,
    job: Dict[str, Any],
    poll_interval: float,
    cancel_grace: float,
) -> None:
    """Spawn the runner for one claimed job, wait for it, record the outcome."""
    job_id = job["id"]

    # Cancel can arrive between enqueue and claim. Cheaper to notice here than
    # to start a process only to kill it.
    if _cancel_wanted(conn, job_id):
        emit(
            "worker",
            f"Job {job_id} was cancelled before it started; not spawning a runner.",
            level="warning",
            job_id=job_id,
        )
        _settle(conn, job_id, None, killed=True, failure=None, child_log=None)
        return

    command = [sys.executable, "-m", "webui.runner", job_id]
    child_log: Optional[str] = None
    proc: Optional[subprocess.Popen] = None
    killed = False
    failure: Optional[str] = None

    try:
        # Inside the try, so that a directory this worker cannot create ends as
        # a failed job with the reason on it, rather than a row left running
        # until the next startup reap notices it.
        child_log = os.path.join(jobs.ensure_job_dir(job_id), "runner.log")
        # The child's stdout and stderr are the last resort: if it dies before
        # it can write a single event (an import error, a CUDA driver crash),
        # this file is the only explanation, and _settle quotes its tail into
        # the job's error column.
        with open(child_log, "a", encoding="utf-8", errors="replace") as stream:
            stream.write(
                f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)} ===\n"
            )
            stream.flush()
            proc = subprocess.Popen(
                command,
                cwd=_REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
                **_spawn_kwargs(),
            )
        jobs.mark_running(conn, job_id, proc.pid)
        emit(
            "worker",
            f"Running {job['label']} job {job_id} as pid {proc.pid}",
            job_id=job_id,
            pid=proc.pid,
            job_type=job["type"],
            base_name=job["base_name"],
        )
        killed = _wait_for_child(conn, job_id, proc, poll_interval, cancel_grace)
    except BaseException as exc:
        # Includes KeyboardInterrupt: Ctrl+C on the worker must not leave a
        # torch process on the GPU. The finally below kills it and settles the
        # row; then the exception continues on its way.
        failure = f"{type(exc).__name__}: {exc}".strip()
        raise
    finally:
        if proc is not None and proc.poll() is None:
            emit(
                "worker",
                f"Stopping the process tree of job {job_id} (pid {proc.pid}).",
                level="warning",
                job_id=job_id,
                pid=proc.pid,
            )
            killed = True
            kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                logger.error(
                    "Process %s for job %s did not exit after a tree kill.",
                    proc.pid,
                    job_id,
                )
        _sweep_temps(job_id, after_kill=killed)
        _settle(conn, job_id, proc, killed=killed, failure=failure, child_log=child_log)


def _wait_for_child(
    conn: sqlite3.Connection,
    job_id: str,
    proc: subprocess.Popen,
    poll_interval: float,
    cancel_grace: float,
) -> bool:
    """Wait for the runner to exit, watching for a cancel. True if we killed it.

    ``Popen.poll`` is the authority on our own child - it reads the exit status
    the kernel kept for us, so it cannot be confused by a reused pid the way a
    liveness probe could.
    """
    while proc.poll() is None:
        if _cancel_wanted(conn, job_id):
            return _stop_child(job_id, proc, cancel_grace)
        time.sleep(poll_interval)
    return False


def _stop_child(job_id: str, proc: subprocess.Popen, cancel_grace: float) -> bool:
    """Ask a cancelled child to stop, then make it. Returns True (we cancelled).

    The grace period is what makes the cooperative path worth having: the runner
    polls for cancel between steps and exits 130 by itself, which stops the job
    at a boundary where no artifact is half-written. A job that is deep inside a
    transcription cannot answer, and gets its tree killed - safe because every
    artifact write is atomic (:mod:`whycast.io_utils`).
    """
    emit(
        "worker",
        f"Cancel requested for job {job_id}; giving pid {proc.pid} "
        f"{cancel_grace:g}s to stop on its own.",
        level="warning",
        job_id=job_id,
        pid=proc.pid,
    )
    deadline = time.monotonic() + max(0.0, cancel_grace)
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if proc.poll() is None:
        emit(
            "worker",
            f"Job {job_id} did not stop within the grace period; killing the "
            f"process tree of pid {proc.pid}.",
            level="warning",
            job_id=job_id,
            pid=proc.pid,
        )
        kill_process_tree(proc.pid)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            logger.error("Process %s for job %s survived a tree kill.", proc.pid, job_id)
    else:
        emit(
            "worker",
            f"Job {job_id} stopped on its own (exit code {proc.returncode}).",
            job_id=job_id,
            exit_code=proc.returncode,
        )
    return True


#: Age a temp file must reach before a routine (non-kill) sweep removes it.
#: Jobs are strictly serial, so nothing this worker runs can be writing - but a
#: CLI run in another terminal is outside that guarantee, and five minutes is
#: far longer than any single artifact write.
ROUTINE_TEMP_AGE_SECONDS = 300.0


def _sweep_temps(job_id: str, after_kill: bool) -> None:
    """Remove temp files that a write which never finished left behind.

    ``atomic_write_*`` discards its temp file in an ``except BaseException``
    handler, so a job that stops cooperatively cleans up after itself. A process
    killed with ``taskkill /F`` never runs that handler. Measured on Windows:
    killing a write mid-flight leaves ``.<artifact>.<random>.tmp`` next to the
    artifacts, which the scanner then reports as unmatched - one per killed
    write, accumulating forever.

    This runs when a job settles, whichever way it ended, so debris from an
    earlier crash is cleared too rather than lingering until someone notices it
    on the unmatched page. Safe because jobs are strictly serial (ADR-008): no
    other job can be writing. After a tree kill the children are provably gone,
    so every temp file is fair game; otherwise the age guard keeps a write that
    some other process may still hold.

    A failure to sweep is logged and ignored: tidying up must never turn a
    settled job into a worker crash.
    """
    directory = os.environ.get("WHYCAST_PODCAST_DIR") or webui_db.DEFAULT_PODCAST_DIR
    min_age = 0.0 if after_kill else ROUTINE_TEMP_AGE_SECONDS
    try:
        removed = sweep_stale_temps(directory, min_age_seconds=min_age)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Sweeping temp files after job %s failed: %s", job_id, exc)
        return
    if removed:
        emit(
            "worker",
            f"Removed {len(removed)} unfinished temp file(s) after job {job_id}.",
            job_id=job_id,
            removed=len(removed),
        )


def _settle(
    conn: sqlite3.Connection,
    job_id: str,
    proc: Optional[subprocess.Popen],
    killed: bool,
    failure: Optional[str],
    child_log: Optional[str],
) -> None:
    """Guarantee the job row has a terminal status, and that it is the true one.

    The runner records its own verdict, because it knows *why* it failed. This
    is the safety net for the cases where it could not: killed mid-run, an OOM
    kill, an import error before the first line of the event log.
    :func:`webui.jobs.finish` keeps the first verdict, so the worst case here is
    a redundant call, never a rewritten history.
    """
    exit_code = proc.returncode if proc is not None else None
    try:
        row = jobs.get_job(conn, job_id)
    except sqlite3.Error as exc:  # pragma: no cover - defensive
        logger.error("Could not read job %s back: %s", job_id, exc)
        return
    if row is None:  # pragma: no cover - defensive
        logger.warning("Job %s disappeared before it could be settled.", job_id)
        return

    if row["is_terminal"]:
        if (
            exit_code is not None
            and row["exit_code"] is not None
            and row["exit_code"] != exit_code
        ):
            # The runner said one thing, the process did another: it crashed
            # after recording its verdict (a torch teardown crash on exit is the
            # usual suspect). The recorded verdict stands - the work was done -
            # but this is worth knowing about.
            emit(
                "worker",
                f"Job {job_id} recorded exit code {row['exit_code']} but its "
                f"process exited {exit_code}. The recorded verdict "
                f"({row['status']}) stands.",
                level="warning",
                job_id=job_id,
                recorded_exit_code=row["exit_code"],
                actual_exit_code=exit_code,
            )
        else:
            logger.info(
                "Job %s already recorded itself as %s (exit code %s).",
                job_id,
                row["status"],
                row["exit_code"],
            )
        return

    parked = jobs.read_pending_verdict(job_id)
    if parked is not None:
        # The runner reached a verdict and the database would not take it, so
        # it left it in the job's directory. That is a first-hand account of
        # why this run ended; everything below is this worker guessing from an
        # exit code. Prefer the account. Without this the fallback text won the
        # race - the database is usually free again by the time we get here -
        # and then cleared the sidecar, so the reason was lost for good.
        emit(
            "worker",
            f"Job {job_id} left a {parked['status']} verdict on disk that the "
            f"database refused at the time; recording that rather than a guess "
            f"from exit code {exit_code}.",
            level="warning",
            job_id=job_id,
            status=parked["status"],
        )
        jobs.finish_or_record(
            conn,
            job_id,
            parked["status"],
            exit_code=parked["exit_code"] if parked["exit_code"] is not None else exit_code,
            error=parked["error"],
        )
        return

    if failure is not None:
        # Checked before `killed`, because the finally block sets `killed` for
        # any child still alive when something went wrong here. A worker that
        # broke mid-job did not cancel it, and saying "cancelled" would hide
        # the actual reason.
        status = "failed"
        error = f"The worker could not run this job: {failure}"
    elif killed:
        status = "cancelled"
        error = (
            "Cancelled: the worker stopped the job"
            + (f" (pid {proc.pid})" if proc is not None else " before it started")
            + "."
        )
    elif exit_code == 0:
        status = "succeeded"
        error = None
    elif exit_code == EXIT_CANCELLED:
        status = "cancelled"
        error = "Cancelled by the runner."
    else:
        status = "failed"
        tail = _log_tail(child_log)
        if exit_code == EXIT_BAD_JOB:
            # The one exit code the runner defines precisely: it refused to
            # start. Saying "probably out of memory" here sent people looking
            # at CUDA when the real cause was a missing input or an import
            # error (webui.runner.EXIT_BAD_JOB).
            reason = (
                f"The runner refused to start this job (exit code {exit_code}): "
                f"a bad job id, bad parameters, a missing input file or a "
                f"missing API key. Nothing was run and nothing was spent."
            )
        else:
            reason = (
                f"The runner exited with code {exit_code} without recording a "
                f"reason - it was probably killed (out of memory) or crashed "
                f"before it could."
            )
        error = reason + " See the job's events.jsonl for how far it got." + (
            f" Last output: {tail}" if tail else ""
        )

    try:
        jobs.finish_or_record(conn, job_id, status, exit_code=exit_code, error=error)
    except (ValueError, WhycastError) as exc:  # pragma: no cover - defensive
        logger.error("Could not record job %s as %s: %s", job_id, status, exc)


def _cancel_wanted(conn: sqlite3.Connection, job_id: str) -> bool:
    """Has a cancel been requested for this job?

    A row that has vanished counts as "stop": nobody is waiting for a result
    that can no longer be recorded anywhere.
    """
    try:
        return jobs.cancel_requested(conn, job_id)
    except ValueError:
        logger.warning("Job %s is gone from the queue; stopping it.", job_id)
        return True
    except sqlite3.Error as exc:
        logger.error("Could not check the cancel flag for job %s: %s", job_id, exc)
        return False


def _log_tail(path: Optional[str], limit: int = _LOG_TAIL_CHARS) -> str:
    """The last few hundred characters the *child* wrote, collapsed to one line.

    The ``=== timestamp command ===`` banners are this worker's own, written to
    separate one attempt from the next when the log is read by hand. They are
    dropped here: quoting our own header back into a job's error column says
    nothing about why the job died, and an empty tail is more honest than a
    line that looks like output and is not.
    """
    if not path or not os.path.isfile(path):
        return ""
    try:
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="replace") as stream:
            if size > limit * 4:
                stream.seek(size - limit * 4)
                stream.readline()  # drop the partial line the seek landed in
            text = stream.read()
    except OSError:  # pragma: no cover - defensive
        return ""
    lines = [
        line
        for line in text.splitlines()
        if not (line.startswith("=== ") and line.endswith(" ==="))
    ]
    collapsed = " ".join(" ".join(lines).split())
    return collapsed[-limit:] if len(collapsed) > limit else collapsed


def _spawn_kwargs() -> Dict[str, Any]:
    """Platform flags that make the child killable as a tree.

    Windows: ``CREATE_NEW_PROCESS_GROUP`` gives the child its own group, so a
    Ctrl+C in the worker's console does not reach it (the worker decides when a
    job stops) and ``taskkill /T`` has a group to walk.
    POSIX: ``start_new_session`` makes the child a process-group leader, which
    is what lets :func:`os.killpg` take down its whole tree.
    """
    if _IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the worker process."""
    parser = argparse.ArgumentParser(
        prog="python -m webui.worker",
        description=(
            "Run queued WHYcast jobs, one at a time, each in its own child "
            "process (ADR-008). One worker per machine."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single job and exit (returns immediately if the queue is empty).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        metavar="SECONDS",
        help=f"Queue poll interval (default: {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--cancel-grace",
        type=float,
        default=DEFAULT_CANCEL_GRACE,
        metavar="SECONDS",
        help=(
            "How long a cancelled job may take to stop by itself before its "
            f"process tree is killed (default: {DEFAULT_CANCEL_GRACE})."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="Console log level (default: info).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m webui.worker``. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")

    try:
        run_worker(
            poll_interval=args.poll_interval,
            once=args.once,
            cancel_grace=args.cancel_grace,
        )
    except WorkerAlreadyRunning as exc:
        logger.error("%s", exc)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        logger.info("Worker stopped by Ctrl+C.")
        return EXIT_OK
    except (sqlite3.Error, WhycastError, OSError) as exc:
        logger.error("Worker could not start: %s", exc)
        return EXIT_REFUSED
    return EXIT_OK


if __name__ == "__main__":
    # The one place this module may exit: it is a process entry point, not
    # library code (ADR-008 forbids exit() in the library, not in __main__).
    sys.exit(main())
