"""
The job queue for the WHYcast web UI (ADR-008, TASK-003 phase 2).

This module is the *queue*, and nothing else. It never spawns a process, never
touches ``podcasts/``, never imports the pipeline. Three parties share it:

* :mod:`webui.app`   - enqueues jobs and reads them back for the dashboard.
* :mod:`webui.worker` - the single serial worker process: claims, spawns the
  runner as a child process, records the exit.
* :mod:`webui.runner` - the child process itself: polls :func:`cancel_requested`
  and writes its event log next to :func:`events_path`.

Design notes worth knowing before changing this module:

* **Same database as the episode index, different lifetime.** ``webui.db``
  caches ``podcasts/``: delete it and a rescan rebuilds every row. Job history
  is *not* rebuildable from disk - it is the only record that a run happened
  and why it failed. So the queue carries its **own** version key,
  ``jobs_schema_version`` in the shared ``meta`` table, and never rides on
  :data:`webui.db.SCHEMA_VERSION`. Bumping the index version drops
  ``episodes``/``artifacts``/``unmatched`` and deliberately leaves this table
  alone (``webui.db._MANAGED_META_KEYS`` lists only the keys that module owns),
  and a jobs migration must likewise never touch the index tables.

* **One connection, one lock.** ``webui.db.init_db`` hands out a single
  ``sqlite3.Connection`` shared across FastAPI's threadpool, and guards it with
  a module-level ``RLock``. This module imports *that* lock rather than making
  its own: two different locks would let a ``BEGIN IMMEDIATE`` here land in the
  middle of ``rescan``'s open transaction on the same connection, which SQLite
  reports as "cannot start a transaction within a transaction" - or worse,
  silently reads half a rescan.

* **The lock is not what makes claiming safe.** In production the web server
  and the worker are separate *processes* with separate connections, so an
  in-process lock cannot serialise them. Safety comes from SQL:
  :func:`claim_next` runs one ``UPDATE ... WHERE status='queued' AND id =
  (SELECT ... LIMIT 1) RETURNING id`` inside ``BEGIN IMMEDIATE``. The lock only
  keeps *threads sharing one connection* from opening two transactions on it.
  :func:`_claim_next_sql` is the lock-free core, so a test can drive it from
  many connections at once and actually exercise the SQL.

* **Serial execution is the worker's invariant, not this module's.** ADR-008:
  "At most one GPU job may run at a time, enforced by the single worker
  process." :func:`claim_next` deliberately does *not* refuse to hand out a
  second job while one runs - a queue primitive that could only ever be called
  once is untestable, and the worker (one process, one lockfile, one job at a
  time) is where the invariant is actually enforced. :func:`active_job` exists
  so the worker and the UI can see the running job.

* **``params`` is text in SQLite, a dict in the API.** Unknown job types are a
  :class:`ValueError`, never a silent no-op: a typo'd type must fail at the
  enqueue button, not silently sit in the queue forever.

* **No path ever comes out of a job.** ``base_name`` is an opaque index key,
  not a filename fragment. The runner resolves it through
  :func:`webui.db.get_episode` and uses the ``audio_path`` the *index* returns.
  ``params`` must never be joined onto a directory (ADR-008 security contract).

No ``print()``, no ``exit()``: progress goes through :mod:`whycast.events`,
failures are raised (ADR-008 Decision Contract).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from whycast.errors import WhycastError
from whycast.events import emit

# The shared connection's lock, on purpose - see the module docstring. It is an
# RLock, so a jobs call nested inside another jobs call on the same thread is
# fine; a jobs call nested inside an open db.py transaction is not, and _begin()
# turns that into a clear error rather than a confusing SQLite one.
from webui.db import _LOCK as _CONN_LOCK

__all__ = [
    "JOBS_SCHEMA_VERSION",
    "JOB_TYPES",
    "STATUSES",
    "TERMINAL_STATUSES",
    "DEFAULT_JOBS_DIR",
    "JobQueueError",
    "ensure_job_schema",
    "enqueue",
    "claim_next",
    "mark_running",
    "finish",
    "request_cancel",
    "cancel_requested",
    "get_job",
    "list_jobs",
    "active_job",
    "jobs_dir",
    "events_path",
    "ensure_job_dir",
    "PENDING_VERDICT_FILENAME",
    "finish_or_record",
    "read_pending_verdict",
    "clear_pending_verdict",
]

#: Version of the ``jobs`` table layout. Independent of
#: :data:`webui.db.SCHEMA_VERSION`: job history cannot be rebuilt from disk, so
#: a mismatch here migrates, it never drops.
JOBS_SCHEMA_VERSION = 1

#: The meta key holding :data:`JOBS_SCHEMA_VERSION`. Deliberately *not* in
#: ``webui.db._MANAGED_META_KEYS``, so an index schema rebuild leaves it alone.
_VERSION_KEY = "jobs_schema_version"

#: The only statuses that may ever appear in the ``status`` column. Enforced
#: twice: a CHECK constraint below, and validation in :func:`finish`.
STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")

#: Statuses a job can never leave.
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")

#: Statuses :func:`finish` accepts as a destination.
_FINISH_STATUSES = TERMINAL_STATUSES

#: ``UPDATE ... RETURNING`` (the race-free claim) needs SQLite 3.35.
_MIN_SQLITE_VERSION = (3, 35, 0)

#: Job error text is a UI blurb, not a log: the full traceback belongs in the
#: job's ``events.jsonl``. Longer messages are truncated with a marker so a
#: reader knows to go look there.
_MAX_ERROR_CHARS = 2000

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

#: Default root for per-job event logs: ``<repo>/logs/jobs``. Overridden by
#: ``WHYCAST_JOBS_DIR`` (ADR-007: environment variables, no parallel config
#: store). Nothing is created at import; see :func:`ensure_job_dir`.
DEFAULT_JOBS_DIR = os.path.join(_REPO_ROOT, "logs", "jobs")

#: Job ids are ``uuid4().hex``. The pattern is wider than that on purpose (a
#: test may use a readable id) but narrow enough that a job id can never be
#: anything but one path segment: no separators, no dots, no ``..``.
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Where a verdict goes when the database refused to take it. Sits in the job's
#: own log directory, next to ``events.jsonl`` and ``runner.log``. See
#: :func:`finish_or_record`.
PENDING_VERDICT_FILENAME = "verdict.json"

#: How many times :func:`finish_or_record` retries a rejected write, and how
#: long it waits between attempts.
#:
#: Be clear about the real budget: the backoff below is the *smaller* half. Each
#: attempt already blocks inside SQLite for up to its ``busy_timeout`` (5 s from
#: ``sqlite3.connect``) before it raises, so three attempts is up to ~15 s of
#: SQLite plus 0.7 s of waiting here - call it 15 s stalled in a worker that has
#: a queue to drain. That is the ceiling this is tuned against, not the 0.7.
#:
#: The retry is only here for contention that clears in milliseconds, which is
#: the common case. It is explicitly *not* the fix for a long hold: the failure
#: this was written against held the write lock for 35-40 s, which no bounded
#: retry can outlast. The sidecar is what makes that case survivable.
_FINISH_ATTEMPTS = 3
_FINISH_BACKOFF_SECONDS = (0.2, 0.5)

#: The dispatch table. :mod:`webui.runner` switches on these exact keys, and
#: :mod:`webui.app` reads ``cost`` to decide whether to show a "this spends
#: money" confirmation before enqueueing.
#:
#: ``gpu``  - claims the single CUDA device (ADR-001 transcription, ADR-002
#:            diarization). Informational here: the worker is serial for every
#:            job type, so this drives UI labelling, not scheduling.
#: ``cost`` - makes paid OpenAI calls (ADR-003).
#: ``requires_base_name`` - the job is meaningless without an episode key.
JOB_TYPES: Dict[str, Dict[str, Any]] = {
    "selftest": {
        "gpu": False,
        "cost": False,
        "requires_base_name": False,
        "label": "Self-test",
        "description": (
            "Emits progress events for a few seconds and succeeds. Spends no "
            "money and touches no GPU; exists so the queue, worker, event "
            "stream and cancel path can be verified end to end."
        ),
    },
    "fetch_all": {
        "gpu": False,
        "cost": False,
        "requires_base_name": False,
        "label": "Download all episodes",
        "description": (
            "Downloads every episode audio file from the RSS feed that is not "
            "on disk yet. No transcription, no OpenAI calls."
        ),
    },
    "fetch_latest": {
        "gpu": True,
        "cost": True,
        "requires_base_name": False,
        "label": "Fetch latest and process",
        "description": (
            "Downloads the newest episode from the RSS feed and runs the full "
            "pipeline on it: transcription, diarization, and the OpenAI "
            "post-processing steps."
        ),
    },
    "full_episode": {
        "gpu": True,
        "cost": True,
        "requires_base_name": True,
        "label": "Process episode",
        "description": (
            "Runs the full pipeline on one episode's audio: transcription, "
            "diarization, and the OpenAI post-processing steps. Existing "
            "artifacts are kept unless the pipeline overwrites them."
        ),
    },
    "force_episode": {
        "gpu": True,
        "cost": True,
        "requires_base_name": True,
        "label": "Reprocess episode (force)",
        "description": (
            "Deletes this episode's existing artifacts and runs the full "
            "pipeline again from the audio. The audio and the saved speaker "
            "mapping are kept - the mapping is human input, not an artifact. "
            "The most expensive job type: it pays for both GPU time and every "
            "OpenAI step."
        ),
    },
    "postprocess": {
        "gpu": False,
        "cost": True,
        "requires_base_name": True,
        "label": "Re-run post-processing",
        "description": (
            "Re-runs cleanup, summary, blog and history from the transcript "
            "already on disk. The cheap iteration path: OpenAI calls only, no "
            "transcription and no GPU."
        ),
    },
    "retranscribe": {
        "gpu": True,
        "cost": False,
        "requires_base_name": True,
        "label": "Re-transcribe (no AI steps)",
        "description": (
            "Diarizes and transcribes the audio again, and stops there. The one "
            "GPU job that costs nothing: no OpenAI calls at all. Use it after "
            "changing the Whisper model, the vocabulary or the diarization "
            "settings. The summary, blog and history already on disk are left "
            "alone and will describe the previous transcript until you re-run "
            "post-processing."
        ),
    },
    "speakers": {
        "gpu": False,
        "cost": True,
        "requires_base_name": True,
        "label": "Re-run speaker assignment",
        "description": (
            "Re-runs speaker identification and assignment over the existing "
            "transcript. OpenAI calls only, no transcription and no GPU."
        ),
    },
}

#: One post-processing step, re-runnable on its own (TASK-004 follow-up).
#:
#: ``needs`` names the artifact kinds the step reads. They have to be on disk,
#: which is the whole reason ``<base>_cleaned.txt`` is written again: without
#: it, nothing downstream of cleanup could be repeated without redoing cleanup
#: too, and paying for it.
#:
#: ``writes`` is what the step replaces, and therefore what is moved aside
#: before it runs (ADR-011).
STEP_JOBS = {
    "cleanup": {
        "label": "Cleanup only",
        "needs": (),          # starts from the transcript, like a full run
        "writes": ("cleaned",),
        "description": (
            "Cleans the transcript again and writes <base>_cleaned.txt. Every "
            "step below reads that file, so re-run this one after editing the "
            "cleanup prompt."
        ),
    },
    "summary": {
        "label": "Summary only",
        "needs": ("cleaned",),
        "writes": ("summary",),
        "description": "Rewrites the summary from the cleaned transcript.",
    },
    "blog": {
        "label": "Blog only",
        "needs": ("cleaned", "summary"),
        "writes": ("blog",),
        "description": "Rewrites the blog post from the cleaned transcript and the summary.",
    },
    "blog_alt1": {
        "label": "Alternative blog only",
        "needs": ("cleaned", "summary"),
        "writes": ("blog_alt1",),
        "description": (
            "Rewrites the alternative blog post. Needs prompts/blog_alt1_prompt.txt, "
            "which is currently renamed to .bk, so this step does nothing until that "
            "is put back."
        ),
    },
    "history": {
        "label": "History extraction only",
        "needs": ("cleaned",),
        "writes": ("history",),
        "description": "Rewrites the history extraction from the cleaned transcript.",
    },
}



#: One job per post-processing step, so a single step can be repeated on its own
#: instead of paying for the four that were already right.
#:
#: Registered from the table above, which :mod:`webui.runner` also reads for its
#: handlers and backup rules. The table lives here because this module imports
#: nothing from the pipeline: putting it in the runner and importing it back
#: made webui.jobs depend on webui.runner and webui.runner depend on webui.jobs,
#: so `import webui.runner` failed outright. `python -m webui.runner` survived it
#: by accident - `-m` loads the file as __main__ and imports the module a second
#: time - which is exactly the kind of luck that hides a broken import.
for _name, _spec in STEP_JOBS.items():
    JOB_TYPES[f"step_{_name}"] = {
        "gpu": False,
        "cost": True,
        "requires_base_name": True,
        "label": _spec["label"],
        "description": _spec["description"],
    }
del _name, _spec


class JobQueueError(WhycastError):
    """Raised when the queue is asked to do something it cannot do safely.

    Distinct from :class:`ValueError`, which this module raises for bad
    *arguments* (unknown job type, unknown status, unknown job id). This one
    means the queue itself is in a state the caller must not paper over: a
    SQLite too old for the race-free claim, or a transaction already open on
    the shared connection.
    """


# The statements are executed one by one, not through executescript(): that
# helper implicitly COMMITs any pending transaction before it runs, which would
# silently break a caller's open transaction. Same reasoning as webui.db.
_META_STATEMENT = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)"""

_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS jobs (
    id               TEXT    PRIMARY KEY,      -- uuid4().hex
    type             TEXT    NOT NULL,         -- a key of JOB_TYPES
    params           TEXT    NOT NULL,         -- JSON object, "{}" when empty
    base_name        TEXT,                     -- episode key, NULL when not episode-scoped
    status           TEXT    NOT NULL
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    cancel_requested INTEGER NOT NULL DEFAULT 0
        CHECK (cancel_requested IN (0, 1)),
    worker_id        TEXT,                     -- which worker claimed it
    pid              INTEGER,                  -- the runner child, for taskkill /T
    exit_code        INTEGER,                  -- the child's exit code
    error            TEXT,                     -- short blurb; detail is in events.jsonl
    created_at       REAL    NOT NULL,         -- epoch seconds, like webui.db mtimes
    started_at       REAL,
    finished_at      REAL
)""",
    # The claim's ORDER BY. rowid breaks ties inside one clock tick and cannot
    # be part of an index, so it is left to the query.
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)",
    # The dashboard: newest first, optionally filtered by status.
    "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)",
    # "what has run for this episode": the episode detail page.
    "CREATE INDEX IF NOT EXISTS idx_jobs_base_name ON jobs(base_name)",
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def ensure_job_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate the queue tables. Idempotent; safe on every startup.

    Callable on a bare connection: it creates ``meta`` itself rather than
    assuming :func:`webui.db.init_db` ran first, so the worker process can open
    the database and get straight to work.

    Migration policy, and why it differs from :mod:`webui.db`:

    * unknown version (fresh database) - create the tables, record the version;
    * same version - ``CREATE TABLE IF NOT EXISTS`` and stop;
    * older version - apply the migration ladder in order, then record the new
      version. Job rows are never dropped: they are not rebuildable from disk;
    * newer version - warn and leave everything untouched. That means an older
      build was pointed at a database a newer build wrote. Destroying job
      history to "fix" it would be the worse outcome; a query that then fails
      on a missing column fails loudly, which is what we want.

    Args:
        conn: connection from :func:`webui.db.init_db`, or any connection with
            ``row_factory = sqlite3.Row``.

    Raises:
        JobQueueError: if SQLite is older than 3.35 (no ``UPDATE ...
            RETURNING``, so :func:`claim_next` could not be race-free), or if a
            transaction is already open on this connection.
        sqlite3.Error: if the schema cannot be created.
    """
    if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
        raise JobQueueError(
            "The job queue needs SQLite "
            f"{'.'.join(str(p) for p in _MIN_SQLITE_VERSION)} or newer for "
            f"UPDATE ... RETURNING (the race-free claim); this Python is "
            f"linked against {sqlite3.sqlite_version}."
        )

    warning: Optional[str] = None
    with _CONN_LOCK:
        # Before the meta DDL, not just before the BEGIN: a CREATE TABLE inside
        # someone else's open transaction fails with a far less helpful message
        # than this one.
        _require_no_transaction(conn)
        conn.execute(_META_STATEMENT)
        stored = _read_meta_value(conn, _VERSION_KEY)
        stored_version = _as_int(stored)

        if stored is not None and stored_version is None:
            warning = (
                f"meta['{_VERSION_KEY}'] is {stored!r}, which is not a version "
                f"number. Treating the queue as un-versioned and creating any "
                f"missing tables; no job row is touched."
            )

        if stored_version is not None and stored_version > JOBS_SCHEMA_VERSION:
            emit(
                "jobs",
                f"Job queue schema in the database is version {stored_version}, "
                f"newer than this build's {JOBS_SCHEMA_VERSION}. Leaving it "
                f"untouched: job history is not rebuildable from disk.",
                level="warning",
                stored_schema_version=stored_version,
                code_schema_version=JOBS_SCHEMA_VERSION,
            )
            return

        _begin(conn)
        try:
            for statement in _SCHEMA_STATEMENTS:
                conn.execute(statement)
            _migrate(conn, stored_version)
            _write_meta_value(conn, _VERSION_KEY, JOBS_SCHEMA_VERSION)
            conn.execute("COMMIT")
        except BaseException:
            # Guarded exactly as webui.db does it: if the COMMIT itself failed,
            # SQLite may already have ended the transaction and an unguarded
            # ROLLBACK would raise "no transaction is active" *over* the real
            # error.
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    if warning is not None:
        emit("jobs", warning, level="warning", stored_schema_version=stored)


def _migrate(conn: sqlite3.Connection, stored_version: Optional[int]) -> None:
    """Apply schema steps from ``stored_version`` up to the current one.

    Empty at version 1: the ``CREATE TABLE IF NOT EXISTS`` above *is* the
    migration from "nothing". Every future step goes here as an
    ``if stored_version < N:`` block of ``ALTER TABLE`` statements, and must
    keep job rows intact - the whole point of a separate version key.
    """
    return None


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def enqueue(
    conn: sqlite3.Connection,
    job_type: str,
    params: Optional[Dict[str, Any]] = None,
    base_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a queued job and return it.

    Args:
        conn: connection from :func:`webui.db.init_db`.
        job_type: a key of :data:`JOB_TYPES`.
        params: JSON-serialisable options for the runner. ``None`` means ``{}``.
            These are *arguments*, never a path: the runner must resolve
            ``base_name`` through the index and must not join anything from
            here onto a directory (ADR-008).
        base_name: the episode this job targets, as the index keys it. Required
            for the episode-scoped types (``full_episode``, ``force_episode``,
            ``postprocess``, ``speakers``); optional otherwise.

    Returns:
        The created job, as :func:`get_job` would return it.

    Raises:
        ValueError: unknown ``job_type``; ``params`` not a JSON-serialisable
            dict; a required ``base_name`` missing, empty, or containing a path
            separator.
    """
    spec = JOB_TYPES.get(job_type)
    if spec is None:
        raise ValueError(
            f"Unknown job type {job_type!r}. Known types: "
            f"{', '.join(sorted(JOB_TYPES))}."
        )

    payload = _params_json(params)
    base = _clean_base_name(base_name, job_type, bool(spec["requires_base_name"]))
    job_id = uuid.uuid4().hex
    created = time.time()

    with _CONN_LOCK:
        # One statement in autocommit mode: atomic on its own, no BEGIN needed.
        conn.execute(
            "INSERT INTO jobs "
            "(id, type, params, base_name, status, cancel_requested, created_at) "
            "VALUES (?, ?, ?, ?, 'queued', 0, ?)",
            (job_id, job_type, payload, base, created),
        )
        row = _select_job(conn, job_id)

    job = _job_dict(row)
    emit(
        "jobs",
        f"Queued {spec['label']}"
        + (f" for {base}" if base else "")
        + f" (job {job_id})",
        job_id=job_id,
        job_type=job_type,
        base_name=base,
        gpu=bool(spec["gpu"]),
        cost=bool(spec["cost"]),
    )
    return job


def claim_next(
    conn: sqlite3.Connection, worker_id: str, pid: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Atomically take the oldest queued job and mark it running.

    Race-free across processes, which is what actually matters: the web server
    and the worker hold separate connections. One ``UPDATE ... WHERE
    status='queued' AND id = (SELECT ... ORDER BY created_at, rowid LIMIT 1)
    RETURNING id`` inside ``BEGIN IMMEDIATE``. ``BEGIN IMMEDIATE`` takes the
    write lock up front, so a second caller either waits (SQLite's busy
    timeout, 5s by default from ``sqlite3.connect``) and then finds the row no
    longer ``queued``, or gets nothing back. Two callers can never win the same
    job; the loser gets ``None``.

    ``pid`` comes back holding the **claiming process's own** pid, not the
    runner's - the child does not exist yet. :func:`mark_running` overwrites it
    with the child's pid a few milliseconds later.

    That placeholder is not cosmetic. A ``running`` row with ``pid IS NULL``
    reads to :func:`webui.worker.reap_stale_jobs` as "a job whose pid was never
    recorded", which it settles as failed - and a second worker could hit that
    window between the claim and ``mark_running`` and destroy a job that had
    just been claimed and never ran (it happened: 4 of 14 jobs, measured).
    Publishing the claimer's pid closes the window with the truth: for those
    milliseconds the process responsible for this job really is the worker.

    Args:
        conn: connection from :func:`webui.db.init_db`.
        worker_id: identifies the claiming worker in the job row. Non-empty.
        pid: the claiming process's pid; defaults to this process's. Only a
            test that is not the claimer passes something else.

    Returns:
        The claimed job with ``status == "running"``, or ``None`` when the
        queue holds no queued job.

    Raises:
        ValueError: if ``worker_id`` is empty or ``pid`` is not positive.
        JobQueueError: if a transaction is already open on this connection.

    This does **not** refuse to hand out a job while another one runs. Serial
    execution is the single worker process's invariant (ADR-008), not a
    property of the queue; see the module docstring.
    """
    if not isinstance(worker_id, str) or not worker_id.strip():
        raise ValueError("worker_id must be a non-empty string.")
    if pid is None:
        pid = os.getpid()
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"pid must be a positive integer, got {pid!r}.")

    with _CONN_LOCK:
        job = _claim_next_sql(conn, worker_id.strip(), pid)

    if job is not None:
        emit(
            "jobs",
            f"Worker {job['worker_id']} claimed {job['label']} job {job['id']}",
            job_id=job["id"],
            job_type=job["type"],
            base_name=job["base_name"],
            worker_id=job["worker_id"],
        )
    return job


def _claim_next_sql(
    conn: sqlite3.Connection, worker_id: str, pid: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """The lock-free core of :func:`claim_next`. See it for the contract.

    Split out for one reason: correctness here comes from SQLite, not from the
    in-process lock, and a proof that runs everything through one lock proves
    nothing. Tests drive *this* from several connections at once. Production
    callers use :func:`claim_next`, because threads sharing the one connection
    from :func:`webui.db.init_db` must not open two transactions on it.
    """
    if pid is None:
        pid = os.getpid()
    started = time.time()
    _begin(conn)
    try:
        # RETURNING id rather than RETURNING *: the id is all the claim needs to
        # prove, and re-selecting inside the same transaction keeps this from
        # depending on the column order of the table. fetchall(), not
        # fetchone(): a RETURNING statement is only stepped to completion once
        # its rows are consumed, and it must complete before the COMMIT.
        claimed = conn.execute(
            "UPDATE jobs "
            "   SET status = 'running', worker_id = ?, started_at = ?, pid = ? "
            " WHERE status = 'queued' "
            "   AND id = (SELECT id FROM jobs WHERE status = 'queued' "
            "             ORDER BY created_at, rowid LIMIT 1) "
            "RETURNING id",
            (worker_id, started, pid),
        ).fetchall()
        row = _select_job(conn, claimed[0]["id"]) if claimed else None
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise

    return None if row is None else _job_dict(row)


def mark_running(conn: sqlite3.Connection, job_id: str, pid: int) -> None:
    """Record the pid of the runner child the worker just spawned.

    Called right after ``subprocess.Popen``. The pid is what cancel needs:
    ``taskkill /F /T /PID`` takes down the whole tree, so torch and ffmpeg
    grandchildren die with the job instead of holding the GPU.

    Idempotent, and inert on a job that has already reached a terminal status:
    neither the status nor the pid is touched there. Nothing kills from a
    terminal row - the worker holds its own ``Popen`` and kills ``proc.pid`` -
    so stamping one would only make ``pid`` stop meaning "the process that ran
    this job" and mislead a post-mortem. A ``queued`` job is moved to
    ``running`` so a caller that spawns without claiming - a test - still ends
    up with a consistent row.

    Args:
        conn: connection from :func:`webui.db.init_db`.
        job_id: the job's id.
        pid: the child process id. Must be a positive integer.

    Raises:
        ValueError: unknown ``job_id``, or a ``pid`` that is not a positive int.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError(f"pid must be a positive integer, got {pid!r}.")

    now = time.time()
    with _CONN_LOCK:
        cursor = conn.execute(
            "UPDATE jobs "
            "   SET pid = CASE WHEN status IN "
            "                  ('succeeded', 'failed', 'cancelled') THEN pid "
            "             ELSE ? END, "
            "       status = CASE WHEN status = 'queued' THEN 'running' ELSE status END, "
            "       started_at = COALESCE(started_at, ?) "
            " WHERE id = ?",
            (pid, now, job_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"No such job: {job_id!r}.")


def finish(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    exit_code: Optional[int] = None,
    error: Optional[str] = None,
) -> None:
    """Move a job to a terminal status and stamp its outcome.

    Only ``queued`` and ``running`` jobs move. A job that is already terminal
    is left exactly as it was and a warning event is emitted: the worker and a
    late cancel can both reach for the same job, and the *first* verdict is the
    true one - silently overwriting "failed, CUDA out of memory" with
    "cancelled" would destroy the only record of why the run died.

    Args:
        conn: connection from :func:`webui.db.init_db`.
        job_id: the job's id.
        status: one of ``succeeded``, ``failed``, ``cancelled``.
        exit_code: the child's exit code, when there was a child.
        error: short human-readable reason, truncated to
            :data:`_MAX_ERROR_CHARS`. The full detail belongs in the job's
            ``events.jsonl``.

    Raises:
        ValueError: unknown ``job_id`` or a ``status`` outside
            :data:`TERMINAL_STATUSES`.
    """
    if status not in _FINISH_STATUSES:
        raise ValueError(
            f"finish() status must be one of {', '.join(_FINISH_STATUSES)}; "
            f"got {status!r}."
        )
    if exit_code is not None:
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError(f"exit_code must be an int or None, got {exit_code!r}.")

    finished = time.time()
    text = _short_error(error)
    stale: Optional[Dict[str, Any]] = None

    with _CONN_LOCK:
        _begin(conn)
        try:
            row = _select_job(conn, job_id)
            if row is None:
                raise ValueError(f"No such job: {job_id!r}.")
            if row["status"] in TERMINAL_STATUSES:
                stale = _job_dict(row)
            else:
                conn.execute(
                    "UPDATE jobs "
                    "   SET status = ?, exit_code = ?, error = ?, finished_at = ?, "
                    "       started_at = COALESCE(started_at, ?) "
                    " WHERE id = ?",
                    (status, exit_code, text, finished, finished, job_id),
                )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    if stale is not None:
        emit(
            "jobs",
            f"Job {job_id} is already {stale['status']}; refusing to overwrite "
            f"it with {status}. The first verdict stands.",
            level="warning",
            job_id=job_id,
            existing_status=stale["status"],
            rejected_status=status,
        )
        return

    emit(
        "jobs",
        f"Job {job_id} {status}"
        + (f" (exit code {exit_code})" if exit_code is not None else "")
        + (f": {text}" if text else ""),
        level="error" if status == "failed" else "info",
        job_id=job_id,
        status=status,
        exit_code=exit_code,
    )


def finish_or_record(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    exit_code: Optional[int] = None,
    error: Optional[str] = None,
) -> bool:
    """:func:`finish`, but a database that refuses the write does not lose it.

    Returns True when the verdict reached the database (including the "a
    verdict was already there" case, which is also a success: the history is
    correct either way). False means the verdict went to the sidecar file
    instead and the next worker startup will apply it.

    Why this exists. ``finish`` used to be called exactly once from
    :func:`webui.worker._settle`, inside an ``except sqlite3.Error:
    logger.error(...)``. One transient write failure - ``SQLITE_BUSY`` past the
    5 s busy timeout, a disk I/O error, a full disk - and the true verdict died
    with that stack frame: the row stayed ``running`` with a now-dead pid, and
    the *next* poll's ``reap_stale_jobs`` wrote the opposite verdict, telling
    the user a job that had succeeded "died, then run it again". For
    ``force_episode`` that advice deletes the artifacts and re-pays for the
    whole pipeline. Measured, 2 out of 2 attempts.

    Two layers, because neither alone is enough:

    1. **Retry.** Most contention clears in milliseconds; a few short waits
       cost nothing and fix the common case.
    2. **A sidecar file.** The reproduction held the write lock for 35-40 s, so
       no bounded retry could have won. ``<jobs_dir>/<job_id>/verdict.json``
       holds the verdict until a process can write it; ``reap_stale_jobs``
       reads it before it concludes anything about a dead pid. Writing a small
       file in a directory this job already owns is the one thing still
       available when SQLite is not.

    ``finish`` keeps the first verdict, so replaying a stale sidecar is a
    no-op, never a rewrite of history.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(_FINISH_ATTEMPTS):
        try:
            finish(conn, job_id, status, exit_code=exit_code, error=error)
        except sqlite3.Error as exc:
            # The retryable one: contention, a locked file, a transient I/O
            # error. Everything else is either the caller's mistake (ValueError,
            # which propagates) or structural (JobQueueError: a transaction is
            # already open on this connection), and waiting cannot fix those.
            last_error = exc
            if attempt < len(_FINISH_BACKOFF_SECONDS):
                time.sleep(_FINISH_BACKOFF_SECONDS[attempt])
            continue
        except JobQueueError as exc:
            last_error = exc
            break
        clear_pending_verdict(job_id)
        return True

    _write_pending_verdict(job_id, status, exit_code, error, last_error)
    return False


def _write_pending_verdict(
    job_id: str,
    status: str,
    exit_code: Optional[int],
    error: Optional[str],
    cause: Optional[BaseException],
) -> None:
    """Park a verdict on disk for whoever can next reach the database."""
    payload = {
        "job_id": job_id,
        "status": status,
        "exit_code": exit_code,
        "error": error,
        "written_at": time.time(),
        "written_by_pid": os.getpid(),
        "database_error": None if cause is None else " ".join(str(cause).split()),
    }
    try:
        directory = ensure_job_dir(job_id)
        path = os.path.join(directory, PENDING_VERDICT_FILENAME)
        # Same write-temp-then-replace shape as whycast.io_utils, for the same
        # reason: a half-written verdict file read by the next worker would be
        # worse than no verdict file at all. Not imported from there - that
        # module is stdlib-only pipeline code and this is four lines.
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    except (OSError, ValueError, TypeError) as exc:
        emit(
            "jobs",
            f"Could not record the {status} verdict for job {job_id} anywhere: "
            f"the database refused it ({cause}) and so did the disk ({exc}).",
            level="error",
            job_id=job_id,
        )
        return
    emit(
        "jobs",
        f"The database would not take the {status} verdict for job {job_id} "
        f"({cause}); it is parked in {PENDING_VERDICT_FILENAME} and the next "
        f"worker startup will apply it.",
        level="warning",
        job_id=job_id,
        status=status,
        exit_code=exit_code,
    )


def read_pending_verdict(job_id: str) -> Optional[Dict[str, Any]]:
    """A verdict parked by :func:`finish_or_record`, or ``None``.

    Unreadable or malformed counts as ``None``: this is a best-effort rescue
    path, and a corrupt file must not stop a worker from starting.
    """
    try:
        path = os.path.join(jobs_dir(), _safe_job_id(job_id), PENDING_VERDICT_FILENAME)
    except ValueError:
        return None
    try:
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") not in _FINISH_STATUSES:
        return None
    exit_code = payload.get("exit_code")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        exit_code = None
    error = payload.get("error")
    return {
        "status": payload["status"],
        "exit_code": exit_code,
        "error": error if isinstance(error, str) else None,
        "written_at": payload.get("written_at"),
    }


def clear_pending_verdict(job_id: str) -> None:
    """Remove a parked verdict. Never raises; a leftover file is harmless."""
    try:
        path = os.path.join(jobs_dir(), _safe_job_id(job_id), PENDING_VERDICT_FILENAME)
        os.unlink(path)
    except (OSError, ValueError):
        return


def request_cancel(conn: sqlite3.Connection, job_id: str) -> Dict[str, Any]:
    """Ask for a job to stop, and return the job as it now stands.

    Two different things, depending on where the job is:

    * **queued** - it goes straight to ``cancelled``. Nothing has started, so
      there is nothing to kill and no artifact to half-write.
    * **running** - ``cancel_requested`` is flagged and the status stays
      ``running``. The flag is the signal the runner polls
      (:func:`cancel_requested`) and the worker acts on by killing the process
      tree; whoever gets there first calls :func:`finish` with ``cancelled``.
      The job is only cancelled once it has actually stopped.
    * **terminal** - nothing happens. The row is returned unchanged.

    Args:
        conn: connection from :func:`webui.db.init_db`.
        job_id: the job's id.

    Returns:
        The job row after the request.

    Raises:
        ValueError: unknown ``job_id``.
    """
    now = time.time()
    with _CONN_LOCK:
        _begin(conn)
        try:
            row = _select_job(conn, job_id)
            if row is None:
                raise ValueError(f"No such job: {job_id!r}.")
            previous = row["status"]
            if previous == "queued":
                conn.execute(
                    "UPDATE jobs "
                    "   SET status = 'cancelled', cancel_requested = 1, "
                    "       finished_at = ?, error = COALESCE(error, ?) "
                    " WHERE id = ? AND status = 'queued'",
                    (now, "Cancelled before it started.", job_id),
                )
            elif previous == "running":
                conn.execute(
                    "UPDATE jobs SET cancel_requested = 1 "
                    " WHERE id = ? AND status = 'running'",
                    (job_id,),
                )
            row = _select_job(conn, job_id)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    job = _job_dict(row)
    if previous == "queued":
        message = f"Cancelled queued job {job_id} before it started"
    elif previous == "running":
        message = (
            f"Cancel requested for running job {job_id}; "
            f"waiting for the runner to stop"
        )
    else:
        message = f"Job {job_id} is already {previous}; cancel does nothing"
    emit(
        "jobs",
        message,
        level="info" if previous in ("queued", "running") else "warning",
        job_id=job_id,
        previous_status=previous,
        status=job["status"],
    )
    return job


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def cancel_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    """Whether someone has asked for this job to stop. The runner polls this.

    Literally the ``cancel_requested`` flag, and nothing else - the same value
    :func:`get_job` returns under that key. It answers "did a human press
    Cancel", not "should I stop", so a job that succeeded normally reads
    ``False`` and the UI can render a "cancelling..." badge straight off it.

    The runner wants the slightly wider question, and should ask it in full::

        if jobs.cancel_requested(conn, job_id) or jobs.get_job(conn, job_id)["is_terminal"]:
            stop()

    The second half matters after a worker restart: the new worker reaps the
    stale ``running`` row as failed, and an orphaned runner still chewing on the
    GPU has to notice. Spelling it out at that one call site is clearer than
    folding it into a function whose name then means two things.

    Cheap on purpose - one primary-key lookup - so a runner can call it between
    pipeline steps without thinking about cost.

    Raises:
        ValueError: unknown ``job_id``. A job row never disappears on its own,
            so a missing one means the caller has the wrong id; returning
            ``False`` would let a cancelled run continue to completion.
    """
    with _CONN_LOCK:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"No such job: {job_id!r}.")
    return bool(row["cancel_requested"])


def get_job(conn: sqlite3.Connection, job_id: str) -> Optional[Dict[str, Any]]:
    """Return one job, or ``None`` if there is no such id."""
    with _CONN_LOCK:
        row = _select_job(conn, job_id)
    return None if row is None else _job_dict(row)


def list_jobs(
    conn: sqlite3.Connection,
    status: Union[str, Iterable[str], None] = None,
    limit: Optional[int] = 50,
) -> List[Dict[str, Any]]:
    """Return jobs newest first, optionally filtered by status.

    Args:
        conn: connection from :func:`webui.db.init_db`.
        status: one status, an iterable of statuses, or ``None`` for all.
        limit: maximum number of rows; ``None`` for no limit.

    Returns:
        Job dicts, ordered ``created_at`` descending (ties broken by insertion
        order, so a burst of enqueues in one clock tick still reads newest
        first).

    Raises:
        ValueError: a status outside :data:`STATUSES`, or a non-positive limit.
    """
    wanted = _clean_statuses(status)
    sql = "SELECT * FROM jobs"
    params: List[Any] = []
    if wanted:
        sql += f" WHERE status IN ({', '.join('?' for _ in wanted)})"
        params.extend(wanted)
    sql += " ORDER BY created_at DESC, rowid DESC"
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive integer or None, got {limit!r}.")
        sql += " LIMIT ?"
        params.append(limit)

    with _CONN_LOCK:
        rows = conn.execute(sql, params).fetchall()
    return [_job_dict(row) for row in rows]


def active_job(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """Return the running job, or ``None``.

    There is at most one by construction (one worker, one job at a time). If
    the table ever holds more, the oldest is returned and a warning is emitted
    rather than the extras being hidden: two running jobs means two processes
    are on the single GPU, which is exactly the failure ADR-008 exists to
    prevent and must be visible.
    """
    with _CONN_LOCK:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = 'running' "
            "ORDER BY started_at, rowid"
        ).fetchall()
    if not rows:
        return None

    jobs = [_job_dict(row) for row in rows]
    if len(jobs) > 1:
        emit(
            "jobs",
            f"{len(jobs)} jobs are marked running at once "
            f"({', '.join(job['id'] for job in jobs)}). At most one worker may "
            f"run at a time (ADR-008); the queue is reporting the oldest.",
            level="warning",
            running_job_ids=[job["id"] for job in jobs],
        )
    return jobs[0]


# ---------------------------------------------------------------------------
# Event log locations
# ---------------------------------------------------------------------------


def jobs_dir() -> str:
    """Root directory for per-job event logs.

    ``WHYCAST_JOBS_DIR`` if set, else ``<repo>/logs/jobs``. Pure: it computes a
    path and creates nothing, so importing this module or rendering a page
    never makes a directory. Use :func:`ensure_job_dir` before writing.
    """
    return os.path.abspath(os.environ.get("WHYCAST_JOBS_DIR") or DEFAULT_JOBS_DIR)


def events_path(job_id: str) -> str:
    """Path of a job's JSONL event log: ``<jobs_dir>/<job_id>/events.jsonl``.

    Creates nothing - a reader (the SSE endpoint) must be able to ask where a
    log *would* be without making a directory as a side effect. The writer
    calls :func:`ensure_job_dir` first.

    Raises:
        ValueError: if ``job_id`` is not a single safe path segment. Job ids
            come from ``uuid4().hex``, but this is the one place a job
            identifier becomes a filesystem path, so it is checked here rather
            than trusted.
    """
    return os.path.join(jobs_dir(), _safe_job_id(job_id), "events.jsonl")


def ensure_job_dir(job_id: str) -> str:
    """Create (if needed) and return a job's log directory.

    The lazy half of :func:`events_path`: the runner calls this once at start
    up, then opens ``events.jsonl`` inside it.

    Raises:
        ValueError: if ``job_id`` is not a single safe path segment.
        OSError: if the directory cannot be created.
    """
    directory = os.path.join(jobs_dir(), _safe_job_id(job_id))
    os.makedirs(directory, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _require_no_transaction(conn: sqlite3.Connection) -> None:
    """Refuse to work inside someone else's open transaction.

    Not paranoia: the lock this module shares with :mod:`webui.db` is an
    ``RLock``, so a jobs call reached from inside an open ``rescan``
    transaction *on the same thread* sails straight past the lock and then
    fails somewhere deep in SQLite. This turns that into a sentence that says
    what is actually wrong.
    """
    if conn.in_transaction:
        raise JobQueueError(
            "A transaction is already open on this connection. The job queue "
            "shares one connection (and one lock) with the episode index; it "
            "cannot open a nested transaction. Finish the outer transaction "
            "first."
        )


def _begin(conn: sqlite3.Connection) -> None:
    """Open a write transaction, refusing to nest.

    ``BEGIN IMMEDIATE`` takes the write lock immediately instead of upgrading
    later, which is what keeps a claim from racing another process.
    """
    _require_no_transaction(conn)
    conn.execute("BEGIN IMMEDIATE")


def _select_job(conn: sqlite3.Connection, job_id: str) -> Optional[sqlite3.Row]:
    """One job row by id, or None. Caller holds the lock."""
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def _job_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Shape one job row into the dict the API and templates consume.

    A superset of the columns: ``params`` comes back as a dict, timestamps gain
    ISO strings (as :mod:`webui.db` does for mtimes), and ``gpu``/``cost``/
    ``label`` are lifted out of :data:`JOB_TYPES` so a template can show the
    "this spends money" warning without importing anything.
    """
    spec = JOB_TYPES.get(row["type"])
    started = row["started_at"]
    finished = row["finished_at"]
    status = row["status"]

    if started is None:
        duration: Optional[float] = None
    elif finished is not None:
        duration = finished - started
    elif status == "running":
        # A running job's duration is "so far". Terminal-without-finished_at
        # cannot happen through this module, and reporting None there is more
        # honest than a number that keeps growing for a job that stopped.
        duration = time.time() - started
    else:
        duration = None

    return {
        "id": row["id"],
        "type": row["type"],
        "params": _params_dict(row["params"], row["id"]),
        "base_name": row["base_name"],
        "status": status,
        "is_terminal": status in TERMINAL_STATUSES,
        "cancel_requested": bool(row["cancel_requested"]),
        "worker_id": row["worker_id"],
        "pid": row["pid"],
        "exit_code": row["exit_code"],
        "error": row["error"],
        "created_at": row["created_at"],
        "created_at_iso": _iso(row["created_at"]),
        "started_at": started,
        "started_at_iso": _iso(started),
        "finished_at": finished,
        "finished_at_iso": _iso(finished),
        "duration_seconds": duration,
        # Unknown type: a row written by a newer build. Report it rather than
        # crashing the dashboard, and assume the expensive answer for both
        # flags so the UI never under-warns about a job it does not know.
        "label": spec["label"] if spec else row["type"],
        "gpu": bool(spec["gpu"]) if spec else True,
        "cost": bool(spec["cost"]) if spec else True,
        "known_type": spec is not None,
    }


def _params_json(params: Optional[Dict[str, Any]]) -> str:
    """Validate and serialise ``params``. Always a JSON object, never null."""
    if params is None:
        return "{}"
    if not isinstance(params, dict):
        raise ValueError(
            f"params must be a dict or None, got {type(params).__name__}."
        )
    if any(not isinstance(key, str) for key in params):
        raise ValueError("params keys must all be strings (they become JSON).")
    try:
        return json.dumps(params, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"params must be JSON-serialisable: {exc}") from exc


def _params_dict(payload: Optional[str], job_id: str) -> Dict[str, Any]:
    """Parse stored params back to a dict, degrading to ``{}`` on damage.

    Everything this module writes is valid JSON, so a failure here means the
    row was edited by hand or the file is corrupt. The dashboard listing every
    other job matters more than this one row, so the parse failure becomes a
    warning event and an empty dict instead of a 500.
    """
    if not payload:
        return {}
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        emit(
            "jobs",
            f"Job {job_id} has unreadable params ({exc}); treating them as empty.",
            level="warning",
            job_id=job_id,
        )
        return {}
    if not isinstance(value, dict):
        emit(
            "jobs",
            f"Job {job_id} has params of type {type(value).__name__}, expected "
            f"an object; treating them as empty.",
            level="warning",
            job_id=job_id,
        )
        return {}
    return value


def _clean_base_name(
    base_name: Optional[str], job_type: str, required: bool
) -> Optional[str]:
    """Validate the episode key a job targets.

    ``base_name`` is an index key, and the runner looks it up with
    :func:`webui.db.get_episode`. It is rejected here if it looks like a path
    anyway: defence in depth, so a future caller that forgets the rule and
    joins it onto a directory cannot be handed ``..\\..\\.env``.
    """
    if base_name is None or (isinstance(base_name, str) and not base_name.strip()):
        if required:
            raise ValueError(
                f"Job type {job_type!r} targets one episode, so base_name is "
                f"required."
            )
        return None
    if not isinstance(base_name, str):
        raise ValueError(
            f"base_name must be a string, got {type(base_name).__name__}."
        )

    cleaned = base_name.strip()
    if any(char in cleaned for char in ("/", "\\", "\0")) or cleaned in (".", ".."):
        raise ValueError(
            f"base_name must be an episode key, not a path: {base_name!r}."
        )
    return cleaned


def _clean_statuses(status: Union[str, Iterable[str], None]) -> List[str]:
    """Normalise the ``list_jobs`` filter to a validated list of statuses."""
    if status is None:
        return []
    if isinstance(status, str):
        candidates: Sequence[str] = [status]
    else:
        candidates = list(status)
    cleaned: List[str] = []
    for value in candidates:
        if value not in STATUSES:
            raise ValueError(
                f"Unknown status {value!r}. Known statuses: {', '.join(STATUSES)}."
            )
        if value not in cleaned:
            cleaned.append(value)
    if not cleaned:
        raise ValueError(
            "An empty status filter would match nothing; pass None for all jobs."
        )
    return cleaned


def _short_error(error: Optional[str]) -> Optional[str]:
    """Trim an error to a UI-sized blurb, pointing at the event log for more."""
    if error is None:
        return None
    text = str(error).strip()
    if not text:
        return None
    if len(text) <= _MAX_ERROR_CHARS:
        return text
    return text[:_MAX_ERROR_CHARS] + "... (truncated; see the job's events.jsonl)"


def _safe_job_id(job_id: str) -> str:
    """Return ``job_id`` if it is a single safe path segment, else raise."""
    if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
        raise ValueError(
            f"Job id {job_id!r} is not a safe path segment. Ids are "
            f"uuid4().hex; only letters, digits, '-' and '_' are accepted, "
            f"at most 64 characters."
        )
    return job_id


def _read_meta_value(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def _write_meta_value(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iso(epoch: Optional[float]) -> Optional[str]:
    """Local-time ISO 8601 string for an epoch timestamp, or None.

    Mirrors :func:`webui.db._iso` so both halves of the database render times
    the same way in the UI.
    """
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat()
    except (OSError, OverflowError, ValueError):
        return None
