"""
The job runner: one job, one child process (ADR-008, TASK-003 phase 2).

``python -m webui.runner <job_id>`` is spawned by :mod:`webui.worker` and does
exactly one thing: run that job's pipeline call to completion, writing every
progress event to ``<jobs_dir>/<job_id>/events.jsonl`` on the way.

It runs in a **child process** on purpose (ADR-008 Decision Contract). The
pipeline loads torch, CUDA and whisper; a CUDA out-of-memory kill takes the
whole interpreter with it. Isolating that in a child means the web server and
the worker survive it, and the worker still records a verdict for the job.

Who writes what
---------------
The runner owns the *event log* and the *reason*. It knows why a job failed, so
it records its own verdict with :func:`webui.jobs.finish` before it exits. The
worker is the safety net: it guarantees a terminal status exists even when the
child never got the chance to write one (import error, OOM kill, taskkill).
:func:`webui.jobs.finish` keeps the first verdict, so the two can never fight.

Exit codes - the discriminator is "did the pipeline run":

* ``0``   - the job's pipeline call returned without raising.
* ``1``   - the pipeline ran and failed (:class:`whycast.errors.WhycastError`
  or any other exception out of the library).
* ``2``   - we refused to start: unknown job id or type, an episode that is not
  in the index, no transcript on disk, no OpenAI API key configured. Nothing
  was spent, nothing was written.
* ``130`` - cancelled. Either the runner saw ``cancel_requested`` between steps
  and stopped cooperatively (the clean path), or it was interrupted.

Honesty about progress
----------------------
:class:`JsonlEventSink` records the ``progress`` the pipeline itself reports and
derives nothing else, with one exception: :func:`step_marker_progress` reads the
workflow's own ``[2/4]`` markers. That is the pipeline's declared step index,
not an estimate of how far into the step we are - a 40-minute transcription sits
at 0.25 for its entire duration. No percentage here is invented.

Cancellation
------------
The runner polls :func:`webui.jobs.cancel_requested` between steps, where
stopping is free and no artifact is half-written, and exits 130. It does *not*
poll inside a pipeline call: raising out of an event sink would land in the
middle of ``full_workflow``'s ``except Exception`` blocks and be swallowed. A
cancel that arrives mid-transcription is handled by the worker killing the
process tree, which is what the atomic writes in :mod:`whycast.io_utils` exist
to make safe.

No ``print()``, and no ``sys.exit()`` outside ``__main__`` (ADR-008): this
module is importable library code; only the entry point at the bottom exits.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from whycast.errors import PipelineError, SecurityError, WhycastError
from whycast.events import ProgressEvent, emit, use_sink

from webui import db as webui_db
from webui import jobs

__all__ = [
    "EXIT_OK",
    "EXIT_PIPELINE_ERROR",
    "EXIT_BAD_JOB",
    "EXIT_CANCELLED",
    "DEFAULT_RSSFEED",
    "JobCancelled",
    "JobInputError",
    "JsonlEventSink",
    "step_marker_progress",
    "db_path_from_env",
    "output_dir_from_env",
    "rssfeed_from_env",
    "run_job",
    "main",
]

logger = logging.getLogger(__name__)

#: The pipeline call returned without raising.
EXIT_OK = 0
#: The pipeline ran and failed.
EXIT_PIPELINE_ERROR = 1
#: We refused to start: bad job, bad params, missing input, missing API key.
EXIT_BAD_JOB = 2
#: Cancelled - 130 is the shell's convention for "terminated by SIGINT".
EXIT_CANCELLED = 130

#: Same default feed as ``transcribe.py`` and
#: :func:`whycast.pipeline.workflow.full_workflow`, so the CLI and the web UI
#: fetch from the same place when ``WHYCAST_RSSFEED`` is unset (ADR-007).
DEFAULT_RSSFEED = "https://whycast.podcast.audio/@whycast/feed.xml"

#: Longest error text handed to :func:`webui.jobs.finish`. The stack trace goes
#: to the event log; the database gets a sentence.
_MAX_DB_ERROR_CHARS = 500

#: ``[2/4]`` in a workflow message. See :func:`step_marker_progress`.
_STEP_MARKER_RE = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")

#: Cancel is polled this often while the self-test sleeps. Short enough that a
#: cancel feels immediate, long enough that the poll costs nothing.
_CANCEL_POLL_SECONDS = 0.25

#: Self-test guard rails. A job that asks for a week of sleep is a typo, not a
#: request.
_SELFTEST_MAX_SECONDS = 3600.0
_SELFTEST_MAX_STEPS = 200

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)


class JobCancelled(WhycastError):
    """Raised inside the runner when a cancel request is seen between steps."""


class JobInputError(WhycastError):
    """The job cannot be started as specified - exit code 2.

    Distinct from :class:`whycast.errors.PipelineError`, which means the
    pipeline ran and failed. This one means nothing ran: an episode that is not
    in the index, an episode with no audio, a post-processing job with no
    transcript on disk, a missing API key. Nothing was spent and nothing was
    written, and the fix is a different job rather than a retry.
    """


# ---------------------------------------------------------------------------
# The event log
# ---------------------------------------------------------------------------


def step_marker_progress(message: str) -> Optional[float]:
    """Progress implied by a ``[2/4]`` step marker in a pipeline message.

    :func:`whycast.pipeline.workflow.full_workflow` announces its phases as
    ``[1/4] Preparing audio ...``. That marker is the pipeline's own statement
    about where it is, so turning it into ``(step - 1) / total`` is reporting,
    not guessing: ``[2/4]`` means two of four phases are still ahead and one is
    done, hence ``0.25``. Progress *within* a phase is unknown and stays
    unknown - the number does not move again until the next marker.

    Returns ``None`` when the message carries no marker, which is the common
    case; the event then keeps whatever ``progress`` the pipeline set (usually
    ``None``), and nothing is fabricated.
    """
    match = _STEP_MARKER_RE.search(message or "")
    if match is None:
        return None
    try:
        step = int(match.group(1))
        total = int(match.group(2))
    except (TypeError, ValueError):  # pragma: no cover - the regex is digits
        return None
    if total <= 0 or step < 1 or step > total:
        return None
    return (step - 1) / total


class JsonlEventSink:
    """An :class:`whycast.events.EventSink` that appends JSON lines to a file.

    One JSON object per line, one line per event, with exactly these keys:

    ``seq``
        Monotonic integer from 1. The SSE endpoint uses it as the event id, so
        a browser that reconnects with ``Last-Event-ID`` can say "everything
        after 42" and get exactly that.
    ``ts``
        Epoch seconds, taken from the event (not from write time).
    ``step``, ``message``, ``level``, ``progress``, ``data``
        Straight from :class:`whycast.events.ProgressEvent`. ``progress`` is
        filled in from a ``[2/4]`` marker when the pipeline left it empty; see
        :func:`step_marker_progress`.

    Flushed after every line. A reader tailing the file is the whole point: a
    buffered sink would show a 40-minute transcription as total silence and
    then a burst at the end. ``fsync`` is deliberately *not* called - it would
    cost a disk round-trip per line to protect a log that is regenerated by
    re-running the job, and the reader sees the bytes either way.

    Never raises at the caller: a sink that throws would take down a pipeline
    step over a logging problem. An event whose ``data`` cannot be serialised is
    written with the data replaced by its ``repr``; a write that fails is
    reported through :mod:`logging` once and then the sink goes quiet.

    That last case is worth knowing about as a reader: if the log becomes
    unwritable mid-job (a full disk), the job keeps running to completion and
    the event log simply stops. The explanation is in the worker's
    ``runner.log`` for that job, not in ``events.jsonl`` - which by definition
    could not record it. A job whose events stop but whose status still settles
    is that, not a hung job.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.fspath(path))
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Append, never truncate: the log is the only record of what a job did,
        # and a re-run of the same id (which should not happen) must not erase
        # the previous attempt. newline="\n" keeps the file one JSON object per
        # LF-terminated line on Windows too, which is what a JSONL reader
        # expects.
        self._handle = open(self.path, "a", encoding="utf-8", newline="\n")
        self._seq = 0
        self._broken = False

    @property
    def seq(self) -> int:
        """Sequence number of the last event written."""
        return self._seq

    def emit(self, event: ProgressEvent) -> None:
        """Append one event. Never raises."""
        if self._broken:
            return
        self._seq += 1
        progress = event.progress
        if progress is None:
            progress = step_marker_progress(event.message)
        record: Dict[str, Any] = {
            "seq": self._seq,
            "ts": float(getattr(event, "timestamp", time.time())),
            "step": event.step,
            "message": event.message,
            "level": event.level,
            "progress": progress,
            "data": event.data or {},
        }
        try:
            line = json.dumps(record, ensure_ascii=False, default=repr)
        except (TypeError, ValueError):
            # default=repr covers unknown types; circular references still
            # raise. The message matters more than the extras, so keep the
            # event and drop the payload.
            record["data"] = {"unserialisable": repr(event.data)}
            line = json.dumps(record, ensure_ascii=False, default=repr)
        try:
            self._handle.write(line + "\n")
            self._handle.flush()
        except (OSError, ValueError) as exc:
            self._broken = True
            logger.error("Event log %s is no longer writable: %s", self.path, exc)

    def close(self) -> None:
        """Close the file. Safe to call twice."""
        try:
            self._handle.close()
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not close event log %s: %s", self.path, exc)

    def __enter__(self) -> "JsonlEventSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Configuration (ADR-007: environment variables, no parallel config store)
# ---------------------------------------------------------------------------


def db_path_from_env() -> str:
    """Index database path from ``WHYCAST_WEBUI_DB``, else the repo default.

    Mirrors :func:`webui.app.db_path_from_env` deliberately instead of
    importing it: importing :mod:`webui.app` builds the FastAPI application at
    module level, and a job child process has no business loading a web
    framework to find out where the database lives.
    """
    return os.path.abspath(os.environ.get("WHYCAST_WEBUI_DB") or webui_db.DEFAULT_DB_PATH)


def output_dir_from_env() -> str:
    """The directory jobs read from and write artifacts to.

    ``WHYCAST_PODCAST_DIR`` first, because that is the directory the web UI
    indexes - artifacts written anywhere else are invisible to it.
    ``WHYCAST_OUTPUT_DIR`` (what ``transcribe.py`` uses) is honoured as a
    fallback so a CLI-configured machine keeps working, and the repository's
    ``podcasts/`` is the last resort.

    Always absolute, and always passed explicitly to the pipeline: letting
    ``full_workflow`` fall back to its own ``./podcasts`` default would resolve
    against the child's working directory.
    """
    return os.path.abspath(
        os.environ.get("WHYCAST_PODCAST_DIR")
        or os.environ.get("WHYCAST_OUTPUT_DIR")
        or webui_db.DEFAULT_PODCAST_DIR
    )


def rssfeed_from_env() -> str:
    """Podcast feed from ``WHYCAST_RSSFEED``, else :data:`DEFAULT_RSSFEED`.

    Deliberately not taken from job params. ADR-007 puts configuration in
    environment variables; a URL in a job row would be a second configuration
    store, and one that a web request could set.
    """
    return os.environ.get("WHYCAST_RSSFEED") or DEFAULT_RSSFEED


# ---------------------------------------------------------------------------
# Job context
# ---------------------------------------------------------------------------


class _Context:
    """Everything a job handler needs, resolved once and validated once."""

    def __init__(self, conn: sqlite3.Connection, job: Dict[str, Any]) -> None:
        self.conn = conn
        self.job = job
        self.job_id = job["id"]
        self.job_type = job["type"]
        self.params: Dict[str, Any] = job["params"] or {}
        self.base_name: Optional[str] = job["base_name"]
        self.output_dir = output_dir_from_env()

    def check_cancel(self, where: str) -> None:
        """Stop the job if a cancel has been requested. Called between steps.

        Asks the wider question :mod:`webui.jobs` documents: an explicit cancel,
        *or* a job row that has already reached a terminal status. The second
        half matters after a worker restart - the new worker reaps the stale
        ``running`` row as failed, and an orphaned runner still holding the GPU
        has to notice that nobody is waiting for it any more.
        """
        try:
            cancelled = jobs.cancel_requested(self.conn, self.job_id)
            row = jobs.get_job(self.conn, self.job_id)
        except (ValueError, sqlite3.Error) as exc:
            # The row vanished or the database is unreadable. Either way nobody
            # is going to record what this process does, so stop.
            raise JobCancelled(
                f"Job {self.job_id} is no longer readable ({exc}); stopping at {where}."
            ) from exc
        if cancelled:
            raise JobCancelled(f"Cancel requested; stopping at {where}.")
        if row is None or row["is_terminal"]:
            status = "gone" if row is None else row["status"]
            raise JobCancelled(
                f"Job {self.job_id} is already {status}; stopping at {where}."
            )

    def sleep(self, seconds: float, where: str) -> None:
        """Sleep in slices, checking for a cancel between them."""
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.check_cancel(where)
            time.sleep(min(_CANCEL_POLL_SECONDS, remaining))


# ---------------------------------------------------------------------------
# Job handlers
# ---------------------------------------------------------------------------


def _job_selftest(ctx: _Context) -> None:
    """Emit progress events for a few seconds, then succeed (or fail on demand).

    Spends nothing and touches no GPU. It exists so the queue, the worker, the
    SSE stream and the cancel path can be proven end to end on a machine with no
    API key and no CUDA - the one job type a test may run.

    Params (all optional):
        ``seconds``: total runtime, default 5, capped at an hour.
        ``steps``: how many progress events, default 8, capped at 200.
        ``fail``: raise at the end, to exercise the failure path.
    """
    seconds = _float_param(ctx, "seconds", 5.0, 0.0, _SELFTEST_MAX_SECONDS)
    steps = int(_float_param(ctx, "steps", 8, 1, _SELFTEST_MAX_STEPS))
    fail = bool(ctx.params.get("fail", False))

    emit(
        "selftest",
        f"Self-test: {steps} steps over about {seconds:g} seconds"
        + (" (asked to fail at the end)" if fail else ""),
        progress=0.0,
        steps=steps,
        seconds=seconds,
        fail=fail,
    )
    per_step = seconds / steps if steps else 0.0
    for step in range(1, steps + 1):
        ctx.check_cancel(f"self-test step {step}/{steps}")
        emit(
            "selftest",
            f"[{step}/{steps}] Self-test step {step} of {steps}",
            progress=step / steps,
            step_index=step,
            step_count=steps,
        )
        ctx.sleep(per_step, f"self-test step {step}/{steps}")

    if fail:
        raise PipelineError(
            "Self-test failed on purpose (params fail=true). This is what a "
            "real pipeline failure looks like from the outside."
        )
    emit("selftest", "Self-test finished; nothing was spent and nothing was written.")


def _job_fetch_all(ctx: _Context) -> None:
    """Download every feed episode that is not on disk yet. No GPU, no cost."""
    from whycast.pipeline.feed import download_all_episodes_from_rssfeed

    feed = rssfeed_from_env()
    emit(
        "feed",
        f"Downloading missing episodes from {feed} into {ctx.output_dir}",
        rssfeed=feed,
        output_dir=ctx.output_dir,
    )
    download_all_episodes_from_rssfeed(feed, ctx.output_dir)


def _job_fetch_latest(ctx: _Context) -> None:
    """Fetch the newest feed episode and run the full pipeline on it."""
    from whycast.pipeline.workflow import full_workflow

    feed = rssfeed_from_env()
    emit(
        "workflow",
        f"Fetching the latest episode from {feed} and processing it",
        rssfeed=feed,
        output_dir=ctx.output_dir,
    )
    full_workflow(audio_file=None, rssfeed=feed, output_dir=ctx.output_dir)


def _job_full_episode(ctx: _Context) -> None:
    """Run the full pipeline on one episode's existing audio."""
    from whycast.pipeline.workflow import full_workflow

    episode, audio = _resolve_audio(ctx)
    emit(
        "workflow",
        f"Processing {episode['base_name']} from {os.path.basename(audio)}",
        base_name=episode["base_name"],
        audio_path=audio,
        output_dir=ctx.output_dir,
    )
    full_workflow(audio_file=audio, output_dir=ctx.output_dir)


def _job_force_episode(ctx: _Context) -> None:
    """Delete this episode's artifacts, then run the full pipeline again.

    The audio is passed as ``exclude_files`` on top of
    :func:`whycast.pipeline.feed.delete_episode_files`'s own ``.mp3`` guard: the
    library only protects ``.mp3``, and this repository also holds ``.m4a`` and
    ``.wav`` episodes that must survive a reprocess.
    """
    from whycast.pipeline.feed import delete_episode_files
    from whycast.pipeline.workflow import full_workflow

    episode, audio = _resolve_audio(ctx)
    base = episode["base_name"]
    emit(
        "workflow",
        f"Force reprocess: deleting existing artifacts for {base} in "
        f"{ctx.output_dir} (the audio file is kept)",
        level="warning",
        base_name=base,
        audio_path=audio,
        output_dir=ctx.output_dir,
    )
    delete_episode_files(base, ctx.output_dir, exclude_files=[audio])
    ctx.check_cancel("after deleting the old artifacts")
    emit("workflow", f"Processing {base} from {os.path.basename(audio)}")
    full_workflow(audio_file=audio, output_dir=ctx.output_dir)


def _job_postprocess(ctx: _Context) -> None:
    """Re-run cleanup, summary, blog and history from the transcript on disk.

    The cheap iteration path: OpenAI calls only, no transcription and no GPU.
    """
    from whycast.pipeline.postprocess import process_transcript_workflow

    episode, text, source = _read_transcript(ctx)
    base = episode["base_name"]
    emit(
        "workflow",
        f"Re-running post-processing for {base} from {os.path.basename(source)} "
        f"({len(text)} characters)",
        base_name=base,
        source=source,
        characters=len(text),
    )
    results = process_transcript_workflow(text, base, ctx.output_dir)
    produced = sorted(key for key, value in (results or {}).items() if value)
    skipped = sorted(key for key, value in (results or {}).items() if not value)
    emit(
        "workflow",
        "Post-processing produced: " + (", ".join(produced) or "nothing"),
        level="info" if produced else "warning",
        produced=produced,
        skipped=skipped,
    )


def _job_speakers(ctx: _Context) -> None:
    """Re-run speaker identification and assignment over the existing transcript."""
    from whycast.pipeline.speakers import speaker_assignment_step

    episode, text, source = _read_transcript(ctx)
    base = episode["base_name"]
    emit(
        "speakers",
        f"Re-running speaker assignment for {base} from "
        f"{os.path.basename(source)} ({len(text)} characters)",
        base_name=base,
        source=source,
        characters=len(text),
    )
    assigned = speaker_assignment_step(text, base, ctx.output_dir)
    if not assigned:
        # Not an error: the step returns None when the transcript carries no
        # SPEAKER_ tags at all. Saying so is the difference between "nothing to
        # do" and a silent no-op that looks like success.
        emit(
            "speakers",
            f"No speaker assignment was written for {base}. The source "
            f"transcript carries no SPEAKER_ labels, or every attempt to map "
            f"them failed - see the events above.",
            level="warning",
            base_name=base,
            source=source,
        )
    else:
        emit(
            "speakers",
            f"Speaker assignment complete for {base} ({len(assigned)} characters).",
            base_name=base,
            characters=len(assigned),
        )


#: type -> handler. Keys are exactly :data:`webui.jobs.JOB_TYPES`; the check at
#: import time below turns "someone added a job type and forgot the runner"
#: into a failure here rather than a job that queues and never runs.
_HANDLERS: Dict[str, Callable[[_Context], None]] = {
    "selftest": _job_selftest,
    "fetch_all": _job_fetch_all,
    "fetch_latest": _job_fetch_latest,
    "full_episode": _job_full_episode,
    "force_episode": _job_force_episode,
    "postprocess": _job_postprocess,
    "speakers": _job_speakers,
}


# ---------------------------------------------------------------------------
# Input resolution - never from params, always through the index
# ---------------------------------------------------------------------------


def _resolve_episode(ctx: _Context) -> Dict[str, Any]:
    """Look the job's episode up in the index.

    ``base_name`` is an opaque key, never a path fragment (ADR-008 security
    contract). Every filesystem path a job touches comes back from the index,
    which built it by scanning the podcast directory.
    """
    if not ctx.base_name:
        raise JobInputError(
            f"Job type {ctx.job_type!r} targets one episode, but this job has "
            f"no base_name."
        )
    episode = webui_db.get_episode(ctx.conn, ctx.base_name)
    if episode is None:
        raise JobInputError(
            f"Episode {ctx.base_name!r} is not in the index. Rescan the "
            f"podcast directory ({ctx.output_dir}) and try again."
        )
    return episode


def _resolve_audio(ctx: _Context) -> Tuple[Dict[str, Any], str]:
    """The episode and its audio file, both taken from the index."""
    episode = _resolve_episode(ctx)
    audio = episode.get("audio_path")
    if not audio:
        raise JobInputError(
            f"Episode {episode['base_name']} has no audio file in the index, "
            f"so there is nothing to transcribe."
        )
    if not os.path.isfile(audio):
        raise JobInputError(
            f"The audio file the index lists for {episode['base_name']} is "
            f"gone: {audio}. Rescan and try again."
        )
    _require_inside(audio, ctx.output_dir, f"audio for {episode['base_name']}")
    return episode, audio


def _read_transcript(ctx: _Context) -> Tuple[Dict[str, Any], str, str]:
    """Read the transcript a re-run should start from.

    Preference order is ``merged`` then ``transcript``, matching what
    :func:`whycast.pipeline.workflow.full_workflow` feeds to the post-processing
    steps: the merged transcript is the same content with consecutive lines from
    one speaker grouped, which is what the prompts were tuned on. Both keep the
    ``SPEAKER_*`` labels that :func:`speaker_assignment_step` needs; the
    already-assigned artifact is deliberately *not* a candidate, because
    re-assigning names that are already names is a no-op that looks like
    success.

    Returns ``(episode, text, path)``.
    """
    episode = _resolve_episode(ctx)
    artifacts: List[Dict[str, Any]] = episode.get("artifacts") or []
    candidates = [
        artifact
        for kind in ("merged", "transcript")
        for artifact in _sorted_by_format(a for a in artifacts if a["kind"] == kind)
    ]
    if not candidates:
        raise JobInputError(
            f"Episode {episode['base_name']} has no transcript on disk, so "
            f"there is nothing to re-process. Run the full pipeline first."
        )

    path = candidates[0]["path"]
    _require_inside(path, ctx.output_dir, f"transcript for {episode['base_name']}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise JobInputError(f"Could not read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise JobInputError(
            f"{path} is not valid UTF-8 ({exc}); the pipeline writes UTF-8, so "
            f"this file was written by something else."
        ) from exc
    if not text.strip():
        raise JobInputError(f"{path} is empty, so there is nothing to process.")
    return episode, text, path


def _sorted_by_format(artifacts) -> List[Dict[str, Any]]:
    """Order artifacts by format preference: txt, html, wiki, md."""
    from whycast.episodes import ARTIFACT_FORMATS

    order = {fmt: index for index, fmt in enumerate(ARTIFACT_FORMATS)}
    return sorted(artifacts, key=lambda a: order.get(a["fmt"], len(order)))


def _require_inside(path: str, root: str, what: str) -> None:
    """Refuse to touch a file outside the configured podcast directory.

    Belt and braces. These paths come from the index, which only ever scanned
    the podcast directory, so this can only fire when the index is stale
    relative to ``WHYCAST_PODCAST_DIR`` - which is worth a clear error rather
    than a job that silently reads from wherever the last scan pointed.
    """
    try:
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(root)
        inside = os.path.commonpath([real_path, real_root]) == real_root
    except (OSError, ValueError):
        inside = False
    if not inside:
        raise JobInputError(
            f"The {what} sits outside the configured podcast directory "
            f"({root}): {path}. The index is stale; rescan and try again."
        )


def _float_param(
    ctx: _Context, name: str, default: float, low: float, high: float
) -> float:
    """Read one numeric param, clamped to a sane range."""
    raw = ctx.params.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise JobInputError(
            f"Parameter {name!r} must be a number, got {raw!r}."
        ) from exc
    if value != value:  # NaN
        raise JobInputError(f"Parameter {name!r} must be a number, got {raw!r}.")
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_job(job_id: str) -> int:
    """Run one job to completion and return the process exit code.

    Everything - including failures to open the database - is reported through
    the job's event log, because that log is what the UI shows. The only case
    that cannot be logged is a job id that is not a safe path segment, which is
    reported through :mod:`logging` and exits 2.
    """
    try:
        jobs.ensure_job_dir(job_id)
        path = jobs.events_path(job_id)
    except (ValueError, OSError) as exc:
        logger.error("Cannot open an event log for job %r: %s", job_id, exc)
        return EXIT_BAD_JOB

    sink = JsonlEventSink(path)
    try:
        with use_sink(sink):
            return _run_with_sink(job_id)
    finally:
        sink.close()


def _run_with_sink(job_id: str) -> int:
    """The body of :func:`run_job`, with the event sink already installed."""
    started = time.time()
    try:
        conn = webui_db.init_db(db_path_from_env())
        jobs.ensure_job_schema(conn)
    except (sqlite3.Error, OSError, WhycastError) as exc:
        emit(
            "job",
            f"Cannot open the job database ({db_path_from_env()}): {exc}",
            level="error",
            job_id=job_id,
        )
        return EXIT_BAD_JOB

    try:
        return _run_job_row(conn, job_id, started)
    finally:
        try:
            conn.close()
        except sqlite3.Error:  # pragma: no cover - defensive
            pass


def _run_job_row(conn: sqlite3.Connection, job_id: str, started: float) -> int:
    """Load the row, dispatch, and record the verdict."""
    job = jobs.get_job(conn, job_id)
    if job is None:
        emit(
            "job",
            f"No such job: {job_id}. Nothing to run.",
            level="error",
            job_id=job_id,
        )
        return EXIT_BAD_JOB
    if job["is_terminal"]:
        emit(
            "job",
            f"Job {job_id} is already {job['status']}; refusing to run it again.",
            level="error",
            job_id=job_id,
            status=job["status"],
        )
        return EXIT_BAD_JOB

    handler = _HANDLERS.get(job["type"])
    spec = jobs.JOB_TYPES.get(job["type"])
    if handler is None or spec is None:
        message = (
            f"Job {job_id} has type {job['type']!r}, which this runner does not "
            f"know how to execute. Known types: {', '.join(sorted(_HANDLERS))}."
        )
        emit("job", message, level="error", job_id=job_id, job_type=job["type"])
        _record(conn, job_id, "failed", EXIT_BAD_JOB, message)
        return EXIT_BAD_JOB

    # Normally dead code, and deliberately kept. The claim now stamps the
    # claiming worker's pid and ``mark_running`` replaces it with ours within
    # milliseconds, so a row reaching us with no pid means nobody claimed it
    # through the worker at all - this runner was started by hand. Recording
    # our own pid then is what keeps the next worker's reaper from calling this
    # job dead while it is holding the GPU. Only ever *fills*, never
    # overwrites: the worker's value is the process it can kill.
    if job["pid"] is None:
        try:
            jobs.mark_running(conn, job_id, os.getpid())
        except (ValueError, sqlite3.Error) as exc:  # pragma: no cover - defensive
            logger.warning("Could not record the runner pid for job %s: %s", job_id, exc)

    ctx = _Context(conn, job)
    emit(
        "job",
        f"Starting {spec['label']}"
        + (f" for {ctx.base_name}" if ctx.base_name else "")
        + f" (job {job_id}, pid {os.getpid()})",
        progress=0.0,
        job_id=job_id,
        job_type=job["type"],
        params=ctx.params,
        base_name=ctx.base_name,
        gpu=bool(spec["gpu"]),
        cost=bool(spec["cost"]),
        output_dir=ctx.output_dir,
        pid=os.getpid(),
        python=sys.executable,
    )

    try:
        if spec["cost"]:
            # Before any GPU work: a job that will spend twenty minutes on the
            # GPU and then discover it cannot pay for the OpenAI half has wasted
            # twenty minutes.
            _require_api_key(ctx)
        ctx.check_cancel("before starting")
        handler(ctx)
    except JobCancelled as exc:
        return _finish_cancelled(conn, job_id, str(exc), started)
    except KeyboardInterrupt:
        return _finish_cancelled(conn, job_id, "Interrupted.", started)
    except JobInputError as exc:
        return _finish_failed(conn, job_id, exc, EXIT_BAD_JOB, started)
    except Exception as exc:
        # Everything else - whycast.errors.* and anything the library did not
        # anticipate - is "the pipeline ran and failed": exit code 1.
        return _finish_failed(conn, job_id, exc, EXIT_PIPELINE_ERROR, started)

    elapsed = time.time() - started
    emit(
        "job",
        f"Job {job_id} finished successfully in {_duration(elapsed)}.",
        progress=1.0,
        job_id=job_id,
        status="succeeded",
        exit_code=EXIT_OK,
        duration_seconds=elapsed,
    )
    _record(conn, job_id, "succeeded", EXIT_OK, None)
    return EXIT_OK


def _require_api_key(ctx: _Context) -> None:
    """Fail fast when no usable OpenAI key is configured.

    Imported here rather than at module level for two reasons: it keeps the
    self-test job free of every pipeline import, and importing
    :mod:`whycast.pipeline.llm` pulls in :mod:`whycast.config`, which is what
    loads ``.env`` - checking ``os.environ`` before that import would report a
    missing key on a machine that is configured perfectly well.
    """
    from whycast.pipeline.llm import ensure_api_key

    try:
        ensure_api_key()
    except (ValueError, SecurityError) as exc:
        raise JobInputError(
            f"This job makes paid OpenAI calls, but the API key is not usable: "
            f"{str(exc).rstrip('.')}. Set OPENAI_API_KEY in .env or the "
            f"environment and try again. Nothing was started and nothing was "
            f"spent."
        ) from exc
    emit(
        "job",
        "OpenAI API key found; this job will make paid API calls.",
        cost=True,
    )


def _finish_cancelled(
    conn: sqlite3.Connection, job_id: str, reason: str, started: float
) -> int:
    """Record a cooperative stop. Never 'failed' - a cancel is not a failure."""
    elapsed = time.time() - started
    emit(
        "job",
        f"Job {job_id} cancelled after {_duration(elapsed)}: {reason}",
        level="warning",
        job_id=job_id,
        status="cancelled",
        exit_code=EXIT_CANCELLED,
        duration_seconds=elapsed,
    )
    _record(conn, job_id, "cancelled", EXIT_CANCELLED, reason)
    return EXIT_CANCELLED


def _finish_failed(
    conn: sqlite3.Connection,
    job_id: str,
    exc: BaseException,
    exit_code: int,
    started: float,
) -> int:
    """Record a failure: one sentence in the database, the stack in the log."""
    elapsed = time.time() - started
    summary = _short_error(exc)
    emit(
        "job",
        f"Job {job_id} failed after {_duration(elapsed)}: {summary}",
        level="error",
        job_id=job_id,
        status="failed",
        exit_code=exit_code,
        duration_seconds=elapsed,
        error_type=type(exc).__name__,
        # The stack belongs here, in the file nobody has to scroll past in the
        # dashboard, and never in the database's error column.
        traceback=traceback.format_exc(),
    )
    logger.error("Job %s failed: %s", job_id, summary, exc_info=exc)
    _record(conn, job_id, "failed", exit_code, summary)
    return exit_code


def _record(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    exit_code: int,
    error: Optional[str],
) -> None:
    """Write the runner's own verdict, tolerating a database that is gone.

    :func:`webui.jobs.finish_or_record` retries a rejected write and, if the
    database still will not take it, parks the verdict in this job's own
    directory so the next worker startup applies it. That matters most here:
    this process knows *why* the job ended, and the worker's fallback - the
    exit code - is a weaker answer that used to be overwritten by "the worker
    running this job died" a second later. ``finish_or_record`` emits its own
    event when it parks a verdict, so nothing is logged twice here.
    """
    try:
        jobs.finish_or_record(conn, job_id, status, exit_code=exit_code, error=error)
    except (ValueError, WhycastError) as exc:
        emit(
            "job",
            f"Could not record the {status} verdict for job {job_id}: {exc}. "
            f"The worker will fall back to the exit code.",
            level="error",
            job_id=job_id,
        )
        logger.error("Could not record job %s as %s: %s", job_id, status, exc)


def _short_error(exc: BaseException) -> str:
    """One sentence for the database: type and message, no stack, no newlines."""
    text = " ".join(str(exc).split()) or exc.__class__.__doc__ or ""
    summary = f"{type(exc).__name__}: {text}".strip().rstrip(":")
    if len(summary) > _MAX_DB_ERROR_CHARS:
        summary = summary[:_MAX_DB_ERROR_CHARS] + "... (see the job's events.jsonl)"
    return summary


def _duration(seconds: float) -> str:
    """Human-sized duration: '4.2s', '3m 07s', '1h 12m'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m webui.runner <job_id>``. Returns the process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] in ("-h", "--help"):
        logger.error(
            "Usage: python -m webui.runner <job_id>. This is spawned by "
            "python -m webui.worker; it is not normally run by hand."
        )
        return EXIT_BAD_JOB

    # Same logging as the CLI shim: pipeline modules log to transcribe.log, and
    # anything that reaches the console lands in the job's runner.log because
    # the worker redirects our stdout and stderr into it.
    try:
        from whycast.logging_setup import setup_logging

        setup_logging()
    except Exception:  # pragma: no cover - logging must never block a job
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    return run_job(args[0])


if __name__ == "__main__":
    # The one place this module may exit: it is a process entry point, not
    # library code (ADR-008 forbids exit() in the library, not in __main__).
    sys.exit(main())
