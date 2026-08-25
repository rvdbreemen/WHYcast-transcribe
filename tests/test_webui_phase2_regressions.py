"""
Regression tests for the phase-2 review findings (ADR-008, TASK-003).

One class per finding, named after what actually went wrong, because the value
of these tests is that the next person can see *which* failure each one pins.
Every one of them fails against the code as it was reviewed.

**No paid call is ever made here and no real job is ever run.** The only child
processes started are ``python -c "time.sleep(...)"`` stand-ins that carry a
runner-shaped command line - enough to exercise the identity check and the tree
kill without importing torch, and enough that a test which killed the wrong
process would take down the test session rather than pass quietly.

Everything runs against a temporary database, a temporary job-log directory and
a temporary podcast directory: the operator's real ``podcasts/``, real
``webui/whycast_webui.db`` and real ``logs/jobs`` are never touched.
"""

import ast
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webui import db as webui_db  # noqa: E402
from webui import jobs as jobs_module  # noqa: E402
from webui import worker as worker_module  # noqa: E402
from whycast.io_utils import BACKUP_SUFFIX, atomic_write_text  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def queue(tmp_path, monkeypatch):
    """A throwaway queue: its own database and its own job-log directory."""
    jobs_home = tmp_path / "jobs"
    jobs_home.mkdir()
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(jobs_home))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "queue.db"))
    # No backoff: these tests exercise the failure path, not the waiting.
    monkeypatch.setattr(jobs_module, "_FINISH_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    conn = webui_db.init_db(str(tmp_path / "queue.db"))
    jobs_module.ensure_job_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def sleeper(tmp_path):
    """Start stand-in 'runner' processes and guarantee they are cleaned up.

    The command line is what the identity check reads, so it has to look like
    the real thing: ``python ... -m webui.runner <job_id>``. The process itself
    only sleeps - importing the real runner would open the database and try to
    run the job.
    """
    started = []

    def start(job_id, seconds=30):
        proc = subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({seconds})",
             "-m", "webui.runner", job_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started.append(proc)
        # Wait until the OS can report a command line for it, or the identity
        # check would race process creation rather than test anything.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if worker_module.process_command_line(proc.pid):
                break
            time.sleep(0.2)
        return proc

    yield start

    for proc in started:
        if proc.poll() is None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True
            ) if os.name == "nt" else proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass


def _running_row(conn, job_id, pid, cancel_requested=False, started_at=None):
    """Force a job row into ``running`` with a given pid, bypassing the worker."""
    conn.execute(
        "UPDATE jobs SET status = 'running', pid = ?, cancel_requested = ?, "
        "started_at = ? WHERE id = ?",
        (pid, 1 if cancel_requested else 0, started_at or time.time(), job_id),
    )


# ---------------------------------------------------------------------------
# Blocker: a killed backup left a full-length file of NUL bytes
# ---------------------------------------------------------------------------


class TestBackupsArePublishedAtomically:
    """``_make_backup`` used ``shutil.copy2`` straight onto ``<path>.bak``.

    On Windows that pre-sizes the destination and then fills it, so a process
    killed mid-copy left a ``.bak`` of exactly the right size, with the right
    mtime, and mostly zeros - and raised no OSError, because the process was
    gone. A recovery that restored from it restored zeros.
    """

    def test_a_backup_interrupted_midway_leaves_the_previous_backup_intact(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "episode_42_transcript.txt"
        atomic_write_text(target, "version one\n")
        atomic_write_text(target, "version two\n")  # .bak is now "version one"
        backup = tmp_path / ("episode_42_transcript.txt" + BACKUP_SUFFIX)
        assert backup.read_text(encoding="utf-8") == "version one\n"

        def die_halfway(source, destination, length=0):
            destination.write(b"\0" * 4096)
            raise OSError("the copy was interrupted")

        monkeypatch.setattr(shutil, "copyfileobj", die_halfway)
        atomic_write_text(target, "version three\n")

        assert target.read_text(encoding="utf-8") == "version three\n"
        assert backup.read_text(encoding="utf-8") == "version one\n", (
            "a failed backup must leave the previous one whole, never a "
            "partially written file that still looks like a restore point"
        )
        assert [p.name for p in tmp_path.iterdir()] == sorted(
            [target.name, backup.name]
        ), "the failed backup's temp file must be cleaned up"

    def test_the_backup_reaches_its_final_name_by_rename(self, tmp_path, monkeypatch):
        """The mechanism, not just the outcome: ``.bak`` is only ever renamed into.

        Pinned deliberately. The outcome test above can be satisfied by a lucky
        copy; only a rename makes "the ``.bak`` is complete or it is the
        previous one" true for a kill at an arbitrary instant.
        """
        target = tmp_path / "episode_42_summary.txt"
        atomic_write_text(target, "one\n")

        destinations = []
        real_replace = os.replace

        def record(src, dst, *args, **kwargs):
            destinations.append(str(dst))
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", record)
        atomic_write_text(target, "two\n")

        backup = str(target) + BACKUP_SUFFIX
        assert backup in destinations, "the backup must be published by os.replace"
        assert str(target) in destinations
        assert destinations.index(backup) < destinations.index(str(target))

    def test_the_backup_keeps_the_previous_versions_mtime(self, tmp_path):
        """``copy2`` used to carry the metadata; ``copystat`` has to keep doing it."""
        target = tmp_path / "episode_42.txt"
        atomic_write_text(target, "one\n")
        old = (time.time() - 50_000, time.time() - 50_000)
        os.utime(target, old)

        atomic_write_text(target, "two\n")

        backup = tmp_path / ("episode_42.txt" + BACKUP_SUFFIX)
        assert backup.stat().st_mtime == pytest.approx(old[1], abs=2)


# ---------------------------------------------------------------------------
# Blocker: the single-worker lock did not name the queue it protects
# ---------------------------------------------------------------------------


class TestTheLockIsKeyedOnTheQueue:
    """It was ``<WHYCAST_JOBS_DIR>/worker.lock`` while the queue is
    ``WHYCAST_WEBUI_DB``. Two independent variables, so two workers on one
    database each took their own lock and both drained the same table.
    """

    def test_two_job_directories_over_one_database_share_one_lock(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "queue.db"))
        monkeypatch.setenv("WHYCAST_JOBS_DIR", str(tmp_path / "logsA"))
        first = worker_module.lock_path()
        monkeypatch.setenv("WHYCAST_JOBS_DIR", str(tmp_path / "logsB"))
        assert worker_module.lock_path() == first, (
            "the job-log directory must not be able to hand out a second lock "
            "on the same queue"
        )

    def test_two_databases_get_two_locks(self, tmp_path, monkeypatch):
        monkeypatch.setenv("WHYCAST_JOBS_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "a.db"))
        first = worker_module.lock_path()
        monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "b.db"))
        assert worker_module.lock_path() != first
        assert first.startswith(str(tmp_path / "a.db"))


# ---------------------------------------------------------------------------
# Blocker: a job claimed milliseconds ago was reaped as "never started"
# ---------------------------------------------------------------------------


class TestAFreshlyClaimedJobSurvives:
    """``claim_next`` published ``running`` with ``pid`` still NULL, and the
    reaper read that as "a job whose pid was never recorded" and failed it. The
    job never ran and its row said the worker had died.
    """

    def test_the_claim_publishes_the_claimers_pid(self, queue):
        jobs_module.enqueue(queue, "selftest")
        claimed = jobs_module.claim_next(queue, "worker-under-test")
        assert claimed["pid"] == os.getpid()

    def test_the_reaper_leaves_a_job_claimed_a_moment_ago_alone(self, queue):
        jobs_module.enqueue(queue, "selftest")
        claimed = jobs_module.claim_next(queue, "worker-under-test")

        worker_module.reap_stale_jobs(queue)

        after = jobs_module.get_job(queue, claimed["id"])
        assert after["status"] == "running"
        assert after["error"] is None

    def test_a_pidless_row_is_still_spared_inside_the_spawn_window(self, queue):
        """Belt and braces for rows written by an older build."""
        job = jobs_module.enqueue(queue, "selftest")
        _running_row(queue, job["id"], pid=None, started_at=time.time())

        worker_module.reap_stale_jobs(queue)

        assert jobs_module.get_job(queue, job["id"])["status"] == "running"

    def test_a_pidless_row_older_than_the_window_is_settled(self, queue):
        job = jobs_module.enqueue(queue, "selftest")
        _running_row(
            queue,
            job["id"],
            pid=None,
            started_at=time.time() - worker_module._SPAWN_GRACE_SECONDS - 5,
        )

        worker_module.reap_stale_jobs(queue)

        assert jobs_module.get_job(queue, job["id"])["status"] == "failed"


# ---------------------------------------------------------------------------
# Blocker: a completed job was recorded as "the worker running this job died"
# ---------------------------------------------------------------------------


class TestAVerdictSurvivesADatabaseThatRefusesIt:
    """``_settle`` called ``finish`` once and swallowed ``sqlite3.Error``. One
    transient write failure and the true verdict died with the stack frame,
    after which the next reap wrote the opposite one - "run it again" on a job
    that had already succeeded and already been paid for.
    """

    def test_a_refused_verdict_is_parked_on_disk(self, queue, tmp_path):
        job = jobs_module.enqueue(queue, "selftest")
        _running_row(queue, job["id"], pid=os.getpid())

        broken = sqlite3.connect(str(tmp_path / "queue.db"))
        broken.close()  # every statement now raises sqlite3.ProgrammingError

        assert jobs_module.finish_or_record(broken, job["id"], "succeeded", 0) is False

        parked = jobs_module.read_pending_verdict(job["id"])
        assert parked["status"] == "succeeded"
        assert parked["exit_code"] == 0

    def test_the_next_reap_applies_the_parked_verdict(self, queue):
        job = jobs_module.enqueue(queue, "selftest")
        # A dead pid: without the sidecar this is the "the worker died" case.
        _running_row(queue, job["id"], pid=999_999_999)
        jobs_module._write_pending_verdict(
            job["id"], "succeeded", 0, None, RuntimeError("database is locked")
        )

        worker_module.reap_stale_jobs(queue)

        settled = jobs_module.get_job(queue, job["id"])
        assert settled["status"] == "succeeded"
        assert settled["exit_code"] == 0
        assert "died" not in (settled["error"] or "")
        assert jobs_module.read_pending_verdict(job["id"]) is None

    def test_the_worker_prefers_the_runners_parked_reason_over_its_own_guess(
        self, queue
    ):
        """Otherwise the rescue channel rescues the status and loses the reason.

        The database is usually free again by the time ``_settle`` runs, so the
        worker's generic "probably killed (out of memory)" would win the race
        and then clear the sidecar - deleting the runner's first-hand account
        of why the job died.
        """
        job = jobs_module.enqueue(queue, "selftest")
        _running_row(queue, job["id"], pid=os.getpid())
        jobs_module._write_pending_verdict(
            job["id"],
            "failed",
            1,
            "PipelineError: the transcript was empty",
            RuntimeError("database is locked"),
        )

        class StubProc:
            returncode = 1
            pid = os.getpid()

        worker_module._settle(
            queue, job["id"], StubProc(), killed=False, failure=None, child_log=None
        )

        settled = jobs_module.get_job(queue, job["id"])
        assert settled["status"] == "failed"
        assert settled["error"] == "PipelineError: the transcript was empty"
        assert "out of memory" not in settled["error"]
        assert jobs_module.read_pending_verdict(job["id"]) is None

    def test_a_successful_write_removes_the_sidecar(self, queue):
        job = jobs_module.enqueue(queue, "selftest")
        _running_row(queue, job["id"], pid=os.getpid())
        jobs_module._write_pending_verdict(job["id"], "failed", 1, "old", None)

        assert jobs_module.finish_or_record(queue, job["id"], "succeeded", 0) is True
        assert jobs_module.read_pending_verdict(job["id"]) is None


# ---------------------------------------------------------------------------
# Blocker: "cancel the job from the dashboard" did nothing for an orphan
# ---------------------------------------------------------------------------


class TestCancellingAnOrphanActuallyStopsIt:
    """The worker told the operator to cancel from the dashboard, and for a
    runner that cannot poll for cancel that did nothing at all: the row stayed
    ``running`` with ``cancel_requested = 1`` forever and no queued job ever
    ran again. Now the worker confirms the process from its command line and
    kills its tree.
    """

    def test_a_confirmed_orphan_is_stopped_and_recorded_as_cancelled(
        self, queue, sleeper
    ):
        job = jobs_module.enqueue(queue, "selftest")
        proc = sleeper(job["id"])
        _running_row(queue, job["id"], pid=proc.pid, cancel_requested=True)

        orphans = worker_module.reap_stale_jobs(queue)

        assert orphans == [], "a stopped orphan must not keep blocking the queue"
        settled = jobs_module.get_job(queue, job["id"])
        assert settled["status"] == "cancelled"
        assert worker_module._wait_gone(proc.pid, 10), "the process tree must be gone"

    def test_an_unconfirmed_pid_is_never_killed(self, queue):
        """The pytest process is alive and is not a runner. It must survive.

        This is the pid-reuse guard, and it is written so that a regression
        cannot pass quietly: if the worker killed a pid it could not identify,
        this test session would be the process it killed.
        """
        job = jobs_module.enqueue(queue, "selftest")
        _running_row(queue, job["id"], pid=os.getpid(), cancel_requested=True)

        orphans = worker_module.reap_stale_jobs(queue)

        assert [o["id"] for o in orphans] == [job["id"]]
        assert jobs_module.get_job(queue, job["id"])["status"] == "running"
        assert worker_module.process_alive(os.getpid())

    def test_a_live_orphan_nobody_cancelled_is_left_running(self, queue, sleeper):
        job = jobs_module.enqueue(queue, "selftest")
        proc = sleeper(job["id"])
        _running_row(queue, job["id"], pid=proc.pid, cancel_requested=False)

        orphans = worker_module.reap_stale_jobs(queue)

        assert [o["id"] for o in orphans] == [job["id"]]
        assert proc.poll() is None, "a job nobody cancelled must not be killed"

    def test_the_identity_check_refuses_a_pid_it_cannot_match(self, sleeper):
        proc = sleeper("aaaaaaaabbbbbbbbccccccccdddddddd")
        assert worker_module.is_runner_for_job(
            proc.pid, "aaaaaaaabbbbbbbbccccccccdddddddd"
        )
        assert not worker_module.is_runner_for_job(proc.pid, "0" * 32)
        assert not worker_module.is_runner_for_job(os.getpid(), "0" * 32)


# ---------------------------------------------------------------------------
# Minor findings
# ---------------------------------------------------------------------------


class TestSmallerCorrections:
    def test_a_cancelled_job_whose_process_died_is_cancelled_not_failed(self, queue):
        """The user pressed Cancel; reporting "the worker died" was simply wrong."""
        job = jobs_module.enqueue(queue, "selftest")
        _running_row(queue, job["id"], pid=999_999_999, cancel_requested=True)

        worker_module.reap_stale_jobs(queue)

        settled = jobs_module.get_job(queue, job["id"])
        assert settled["status"] == "cancelled"
        assert "died" not in (settled["error"] or "")

    def test_exit_code_two_is_reported_as_a_refusal_to_start(self, queue):
        """``EXIT_BAD_JOB`` is the runner's "bad job, bad params, missing input".

        Calling that "probably killed (out of memory)" pointed people at CUDA
        for a missing file.
        """
        job = jobs_module.enqueue(queue, "selftest")
        _running_row(queue, job["id"], pid=os.getpid())

        class StubProc:
            returncode = 2
            pid = os.getpid()

        worker_module._settle(
            queue, job["id"], StubProc(), killed=False, failure=None, child_log=None
        )

        error = jobs_module.get_job(queue, job["id"])["error"]
        assert "refused to start" in error
        assert "out of memory" not in error

    def test_mark_running_does_not_stamp_a_pid_on_a_finished_job(self, queue):
        job = jobs_module.enqueue(queue, "selftest")
        jobs_module.claim_next(queue, "worker-under-test")
        jobs_module.finish(queue, job["id"], "succeeded", exit_code=0)
        before = jobs_module.get_job(queue, job["id"])["pid"]

        jobs_module.mark_running(queue, job["id"], 4242)

        after = jobs_module.get_job(queue, job["id"])
        assert after["status"] == "succeeded"
        assert after["pid"] == before, "pid must mean 'the process that ran this job'"


# ---------------------------------------------------------------------------
# Blocker: python -m webui ignored --db / --podcast-dir / --host
# ---------------------------------------------------------------------------


class TestTheEntryPointFlagsReachTheApplication:
    """``webui/__main__.py`` imported ``webui.app`` at module scope, which ran
    ``app = create_app()`` and read the environment *before* ``main()`` wrote
    the flags into it. The flags were silently ignored and the banner reported
    them as applied, so a "scratch" instance served the real index and enqueued
    into the real queue.
    """

    def test_nothing_from_webui_app_is_imported_at_module_scope(self):
        """The rule, pinned where it can be checked without starting a server."""
        source = (REPO_ROOT / "webui" / "__main__.py").read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                assert not any(name.startswith("webui.app") for name in names), (
                    "importing webui.app at module scope builds the application "
                    "before main() can set the environment"
                )
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "") != "webui.app", (
                    "importing webui.app at module scope builds the application "
                    "before main() can set the environment"
                )

    def test_the_flags_win_over_a_hostile_environment(self, tmp_path):
        """A fresh process, because import order is the thing under test."""
        podcasts = tmp_path / "podcasts"
        podcasts.mkdir()
        (podcasts / "episode_9.mp3").write_bytes(b"fake")
        wanted_db = tmp_path / "wanted.db"

        program = textwrap.dedent(
            """
            import json, sys, uvicorn
            uvicorn.run = lambda *a, **kw: None
            from webui.__main__ import main
            main(sys.argv[1:])
            import webui.app as app_module
            print(json.dumps({
                "podcast_dir": app_module.app.state.podcast_dir,
                "db_path": app_module.app.state.db_path,
                "allowed_hosts": list(app_module.app.state.allowed_hosts),
            }))
            """
        )
        # Control the whole WHYCAST_ environment rather than inheriting it.
        # This test is precisely about flags versus environment, so a stray
        # WHYCAST_WEBUI_PORT left behind by another test file would decide the
        # outcome. Everything else (PATH, TEMP) is inherited as usual.
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("WHYCAST_")
        }
        env["WHYCAST_PODCAST_DIR"] = str(tmp_path / "WRONG_PODCASTS")
        env["WHYCAST_WEBUI_DB"] = str(tmp_path / "WRONG.db")
        env["WHYCAST_WEBUI_HOST"] = "0.0.0.0"

        completed = subprocess.run(
            [sys.executable, "-c", program,
             "--podcast-dir", str(podcasts),
             "--db", str(wanted_db),
             "--host", "127.0.0.1"],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=180,
        )
        assert completed.returncode == 0, completed.stderr
        state = json.loads(completed.stdout.strip().splitlines()[-1])

        assert state["podcast_dir"] == str(podcasts)
        assert state["db_path"] == str(wanted_db)
        assert "*" not in state["allowed_hosts"], (
            "the Host allowlist must follow the address actually bound, not a "
            "leftover WHYCAST_WEBUI_HOST - accepting every Host on a loopback "
            "bind reopens DNS rebinding and collapses the origin check with it"
        )
        assert "127.0.0.1" in state["allowed_hosts"]
