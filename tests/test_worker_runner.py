"""
End-to-end tests for :mod:`webui.worker` and :mod:`webui.runner`
(ADR-008, TASK-003 phase 2).

Unlike the API tests, these really do start processes: ``run_worker(once=True)``
claims a job and spawns ``python -m webui.runner <job_id>`` exactly as it does
in production, and the assertions are made about the row and the event log that
child left behind.

**The cost rule, and how it is enforced rather than remembered.** The only job
type any of this runs is ``selftest``: it sleeps, emits progress events and
returns, and it is the one type with ``cost=False`` and ``gpu=False``. Every
path here that can start a worker goes through :func:`run_worker_once` or
:func:`worker_in_thread`, and both call :func:`assert_queue_is_free` first,
which fails the test if the queue holds anything but a self-test. That is
deliberate belt and braces: ``WHYCAST_*`` environment variables cannot make
this safe, because the child loads ``.env`` itself through
:mod:`whycast.config`, so a real API key is always within the child's reach.
The queue assertion is what actually guarantees no paid call and no GPU job.

The two tests that need pipeline code patch it (see
:func:`test_a_patched_pipeline_call_reaches_the_handler`); nothing here ever
imports torch, faster-whisper or the OpenAI client.

Everything runs against a temporary database, a temporary job-log directory and
a temporary podcast directory. The operator's real ``podcasts/``, real
``webui/whycast_webui.db`` and real ``logs/jobs`` are never touched.
"""

import json
import os
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webui import db as webui_db  # noqa: E402
from webui import jobs as jobs_module  # noqa: E402
from webui import runner as runner_module  # noqa: E402
from webui import worker as worker_module  # noqa: E402
from whycast.events import ProgressEvent  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def podcast_dir(tmp_path):
    """A synthetic podcast directory: one episode with audio and a transcript."""
    root = tmp_path / "podcasts"
    root.mkdir()
    (root / "episode_1.mp3").write_bytes(b"fake-mp3-bytes")
    (root / "episode_1.txt").write_text(
        "SPEAKER_00: hello\nSPEAKER_01: world\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def queue_home(tmp_path, podcast_dir, monkeypatch):
    """A scratch queue, job-log directory and podcast directory.

    ``WHYCAST_WEBUI_DB`` carries the worker lock with it -
    :func:`webui.worker.lock_path` is the database path plus a suffix - so this
    one variable keeps both the queue and the lock out of the repository.
    """
    jobs_home = tmp_path / "jobs"
    jobs_home.mkdir()
    db_path = tmp_path / "queue.db"
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(jobs_home))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(db_path))
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcast_dir))
    monkeypatch.delenv("WHYCAST_OUTPUT_DIR", raising=False)
    # Module-level state in webui.worker: an orphan whose identity could not be
    # confirmed is remembered so the refusal is not logged once a second. It
    # would otherwise leak into the next test.
    worker_module._ORPHAN_REFUSALS.clear()
    yield db_path
    worker_module._ORPHAN_REFUSALS.clear()


@pytest.fixture
def conn(queue_home, podcast_dir):
    """A connection with both schemas and the podcast directory indexed."""
    connection = webui_db.init_db(str(queue_home))
    jobs_module.ensure_job_schema(connection)
    webui_db.rescan(connection, str(podcast_dir))
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def sleeper():
    """Start stand-in processes and guarantee they are stopped again.

    The command line is shaped like the real runner (``-m webui.runner
    <job_id>``) because :func:`webui.worker.is_runner_for_job` reads exactly
    that; the process itself only sleeps, so nothing is imported and nothing
    can run a job.
    """
    started = []

    def start(job_id="idle", seconds=120):
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                f"import time; time.sleep({seconds})",
                "-m",
                "webui.runner",
                job_id,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started.append(proc)
        return proc

    yield start

    for proc in started:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass


# ---------------------------------------------------------------------------
# The cost guard, and the helpers every worker test goes through
# ---------------------------------------------------------------------------


def assert_queue_is_free(conn):
    """Refuse to start a worker while the queue holds a job that costs money.

    The single most important line in this file. ``selftest`` is the only type
    that spends nothing and claims no GPU; a worker started over any other
    queued job would run the real pipeline against the operator's OpenAI key.
    """
    expensive = [
        job
        for job in jobs_module.list_jobs(conn, status="queued", limit=None)
        if job["type"] != "selftest"
    ]
    assert not expensive, (
        "COST GUARD: refusing to start a worker while the queue holds "
        + ", ".join(f"{job['type']} ({job['id']})" for job in expensive)
        + ". Only selftest jobs may ever be run by the test suite."
    )


def run_worker_once(conn, poll_interval=0.05, cancel_grace=5.0):
    """One claim-run-exit cycle, after the cost guard has had its say."""
    assert_queue_is_free(conn)
    worker_module.run_worker(
        poll_interval=poll_interval, once=True, cancel_grace=cancel_grace
    )


def worker_in_thread(conn, poll_interval=0.05, cancel_grace=5.0):
    """Start ``run_worker(once=True)`` in a thread; returns ``(thread, box)``.

    ``box`` collects whatever the worker raised, so a test can assert on it
    after joining instead of losing it in a dead thread.
    """
    assert_queue_is_free(conn)
    box = {}

    def run():
        try:
            worker_module.run_worker(
                poll_interval=poll_interval, once=True, cancel_grace=cancel_grace
            )
        except BaseException as exc:  # pragma: no cover - re-raised by the test
            box["error"] = exc

    thread = threading.Thread(target=run, name="worker-under-test")
    thread.start()
    return thread, box


def wait_until(predicate, timeout=90, message="condition"):
    """Poll ``predicate`` until it is true, or fail the test. No pytest-timeout here."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail(f"timed out after {timeout}s waiting for {message}")


def read_events(job_id):
    """Every event the runner wrote for a job, parsed, in file order.

    Strict: a line that is not valid JSON fails the test, because that is
    exactly what the SSE endpoint would have to drop.
    """
    path = jobs_module.events_path(job_id)
    assert os.path.isfile(path), f"the runner wrote no event log at {path}"
    with open(path, "r", encoding="utf-8") as stream:
        lines = [line for line in stream.read().splitlines() if line.strip()]
    records = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(json.loads(line))
        except ValueError as exc:  # pragma: no cover - the assertion is the point
            pytest.fail(f"line {number} of {path} is not valid JSON ({exc}): {line!r}")
    return records


def peek_events(job_id):
    """The events written *so far*, for polling while the job is still running.

    Tolerant where :func:`read_events` is strict: the writer flushes per line
    but a reader can still catch the last one mid-write, and here that is a
    "not yet", not a failure.
    """
    try:
        with open(jobs_module.events_path(job_id), "r", encoding="utf-8") as stream:
            lines = stream.read().splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def enqueue_selftest(conn, seconds=0.2, steps=3, **extra):
    """A self-test job small enough to run inside a test."""
    params = {"seconds": seconds, "steps": steps}
    params.update(extra)
    return jobs_module.enqueue(conn, "selftest", params=params)


# ---------------------------------------------------------------------------
# A real job, end to end
# ---------------------------------------------------------------------------


def test_a_real_selftest_job_runs_end_to_end(conn):
    """Claim, spawn, run, record - the whole path, with a real child process."""
    job = enqueue_selftest(conn, seconds=0.2, steps=4)

    run_worker_once(conn)

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "succeeded", row["error"]
    assert row["exit_code"] == runner_module.EXIT_OK
    assert row["error"] is None
    assert row["is_terminal"] is True
    assert row["pid"] and row["pid"] != os.getpid(), "the job ran in its own process"
    assert row["worker_id"], "the claiming worker is recorded"
    assert row["started_at"] and row["finished_at"]
    assert row["finished_at"] >= row["started_at"]
    assert not worker_module.process_alive(row["pid"]), "the child was not reaped"


def test_the_event_log_of_a_real_job_is_coherent(conn):
    """One JSON object per line, seq from 1, strictly increasing, no gaps.

    The SSE endpoint uses ``seq`` as the browser's ``Last-Event-ID``, so a
    repeated or skipped number is a message a reconnecting client silently
    loses. This is the contract that makes resume work.
    """
    started = time.time()
    job = enqueue_selftest(conn, seconds=0.2, steps=4)

    run_worker_once(conn)

    events = read_events(job["id"])
    assert len(events) >= 6, "a four-step self-test should say more than this"
    assert all(isinstance(event, dict) for event in events)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))

    for event in events:
        assert set(event) == {"seq", "ts", "step", "message", "level", "progress", "data"}
        assert isinstance(event["step"], str) and event["step"]
        assert isinstance(event["message"], str) and event["message"]
        assert event["level"] in ("info", "warning", "error")
        assert isinstance(event["data"], dict)
        assert started - 1 <= event["ts"] <= time.time() + 1
        assert event["progress"] is None or 0.0 <= event["progress"] <= 1.0

    assert events[0]["step"] == "job"
    assert "Starting" in events[0]["message"]
    assert events[0]["data"]["cost"] is False, "a self-test must never claim a cost"
    assert events[0]["data"]["gpu"] is False

    steps = [event for event in events if event["step"] == "selftest"]
    assert len(steps) >= 4
    numbered = [event["progress"] for event in steps if event["data"].get("step_index")]
    assert numbered == sorted(numbered), "progress must never go backwards"

    done = [event for event in events if "finished successfully" in event["message"]]
    assert len(done) == 1
    assert done[0]["progress"] == 1.0, "the last word from the job itself is 'complete'"
    # The queue's own verdict is emitted through the same sink, so it lands in
    # this file too and is the last line. That is what a reader tailing the log
    # sees as the end of the job.
    assert events[-1]["step"] == "jobs"
    assert events[-1]["data"]["status"] == "succeeded"
    assert not any(event["level"] == "error" for event in events)


def test_a_failing_selftest_lands_failed_with_a_visible_reason(conn):
    """The failure path: the row says what went wrong, the log says where."""
    job = enqueue_selftest(conn, seconds=0.05, steps=2, fail=True)

    run_worker_once(conn)

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "failed"
    assert row["exit_code"] == runner_module.EXIT_PIPELINE_ERROR
    assert row["error"], "a failed job with no reason is a bug report nobody can act on"
    assert "PipelineError" in row["error"]
    assert "on purpose" in row["error"]
    assert "\n" not in row["error"], "the database gets a sentence, not a stack"

    events = read_events(job["id"])
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    failures = [event for event in events if event["level"] == "error"]
    assert failures, "the event log must carry the failure too"
    with_stack = [event for event in failures if "traceback" in event["data"]]
    assert len(with_stack) == 1, (
        "the stack belongs in the log, which is the half nobody has to scroll past"
    )
    assert "Traceback" in with_stack[0]["data"]["traceback"]
    assert with_stack[0]["data"]["error_type"] == "PipelineError"
    assert events[-1]["data"]["status"] == "failed", "the queue's verdict closes the log"


def test_cancelling_a_running_job_stops_the_child(conn):
    """Cancel mid-run: the row ends cancelled and the process is really gone.

    A long self-test, cancelled while it is between steps. The runner polls for
    the flag every 0.25s, so it stops itself and exits 130 rather than being
    killed - and that cooperative exit is what makes cancel safe for a real
    pipeline, because it happens at a boundary where no artifact is half
    written.
    """
    job = enqueue_selftest(conn, seconds=60, steps=120)

    thread, box = worker_in_thread(conn)
    try:
        # The claim publishes the *claiming* process's pid for the few
        # milliseconds before mark_running replaces it with the child's, so
        # "not ours" is what identifies the runner.
        pid = wait_until(
            lambda: (jobs_module.get_job(conn, job["id"]) or {}).get("pid")
            not in (None, os.getpid())
            and jobs_module.get_job(conn, job["id"])["pid"],
            message="the worker to spawn the runner",
        )
        assert jobs_module.get_job(conn, job["id"])["status"] == "running"
        assert worker_module.process_alive(pid), "the runner should be running now"
        # Mid-run means mid-run: wait until the job has actually started
        # stepping, so this is not the "cancelled before it started" path
        # wearing a disguise.
        wait_until(
            lambda: any(
                event["data"].get("step_index") for event in peek_events(job["id"])
            ),
            message="the self-test to start stepping",
        )

        jobs_module.request_cancel(conn, job["id"])
    finally:
        thread.join(timeout=120)
    assert not thread.is_alive(), "the worker did not return after the cancel"
    assert "error" not in box, box.get("error")

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "cancelled", row["error"]
    assert row["cancel_requested"] is True
    assert row["exit_code"] == runner_module.EXIT_CANCELLED, (
        "the runner should have stopped itself between steps, not been killed"
    )
    assert not worker_module.process_alive(pid), f"pid {pid} survived the cancel"

    events = read_events(job["id"])
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert any("cancelled" in event["message"].lower() for event in events)
    assert events[-1]["data"].get("status") == "cancelled"


def test_a_job_cancelled_before_it_starts_never_spawns_anything(conn):
    """Cancel between enqueue and claim: cheaper to notice than to kill."""
    job = enqueue_selftest(conn, seconds=30, steps=60)
    jobs_module.request_cancel(conn, job["id"])

    run_worker_once(conn)

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "cancelled"
    assert row["pid"] is None, "nothing should have been spawned"
    assert not os.path.exists(
        os.path.join(jobs_module.jobs_dir(), job["id"], "runner.log")
    ), "a job that never started should not have a runner log"


# ---------------------------------------------------------------------------
# One worker at a time
# ---------------------------------------------------------------------------


def test_a_second_worker_refuses_to_start(conn, sleeper):
    """Two workers would mean two jobs on one GPU. The lock says no.

    The lock file is made to hold the pid of a process that is genuinely alive
    and is not this one - that is the state a running worker leaves - so the
    liveness check has something real to find.
    """
    holder = sleeper()
    lock_file = Path(worker_module.lock_path())
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text(f"{holder.pid}\n", encoding="utf-8")
    job = enqueue_selftest(conn)

    with pytest.raises(worker_module.WorkerAlreadyRunning) as excinfo:
        run_worker_once(conn)

    assert str(holder.pid) in str(excinfo.value)
    assert jobs_module.get_job(conn, job["id"])["status"] == "queued", (
        "a refused worker must not have claimed anything"
    )
    assert lock_file.read_text(encoding="utf-8").strip() == str(holder.pid), (
        "the refused worker must not have taken the other one's lock"
    )


def test_the_lock_travels_with_the_queue(queue_home):
    """The lock names the resource it protects, and the resource is the queue.

    It used to be derived from ``WHYCAST_JOBS_DIR`` while the queue comes from
    ``WHYCAST_WEBUI_DB``; two workers pointed at one database with different
    job directories then each took their own lock and both claimed from the
    same table.
    """
    assert worker_module.lock_path() == str(queue_home) + worker_module.LOCK_SUFFIX
    assert not worker_module.lock_path().startswith(jobs_module.jobs_dir())


def test_a_stale_lock_is_taken_over(conn, caplog):
    """A worker killed with taskkill never cleans up. That must not need a human.

    An unreadable lock file is the honest version of stale: it is what a worker
    leaves behind when it is killed between creating the file and writing its
    pid into it, and it needs no guess about whether some pid is still alive.
    """
    lock_file = Path(worker_module.lock_path())
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text("", encoding="utf-8")
    assert worker_module.read_lock_pid(str(lock_file)) is None
    job = enqueue_selftest(conn, seconds=0.05, steps=1)

    with caplog.at_level("WARNING", logger="webui.worker"):
        run_worker_once(conn)

    assert any("stale" in record.getMessage() for record in caplog.records), (
        "taking over a stale lock must be announced, not done quietly"
    )
    assert jobs_module.get_job(conn, job["id"])["status"] == "succeeded", (
        "after the takeover the worker should have run the job"
    )
    assert not lock_file.exists(), "a worker that stops releases its lock"


def test_a_live_orphan_blocks_the_queue_instead_of_doubling_up(conn, sleeper, caplog):
    """A runner from a dead worker is left alone and nothing new is claimed."""
    orphan = enqueue_selftest(conn, seconds=30, steps=60)
    jobs_module.claim_next(conn, "dead-worker")
    holder = sleeper(job_id=orphan["id"])
    jobs_module.mark_running(conn, orphan["id"], holder.pid)
    waiting = enqueue_selftest(conn, seconds=0.05, steps=1)

    with caplog.at_level("WARNING", logger="webui.worker"):
        run_worker_once(conn)

    assert jobs_module.get_job(conn, orphan["id"])["status"] == "running", (
        "a live orphan must be left strictly alone"
    )
    assert jobs_module.get_job(conn, waiting["id"])["status"] == "queued", (
        "claiming while an orphan runs would put two jobs on one GPU"
    )
    assert holder.poll() is None, "the orphan was killed without being asked to stop"
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Not claiming any job" in messages
    assert str(holder.pid) in messages, "the operator must be told what to stop"


# ---------------------------------------------------------------------------
# Crash recovery: what the next worker startup makes of a dead run
# ---------------------------------------------------------------------------


def test_a_job_left_running_by_a_dead_worker_is_failed_on_the_next_start(
    conn, monkeypatch
):
    job = enqueue_selftest(conn)
    jobs_module.claim_next(conn, "dead-worker")
    jobs_module.mark_running(conn, job["id"], 424242)
    monkeypatch.setattr(worker_module, "process_alive", lambda pid: False)

    run_worker_once(conn)

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "failed"
    assert "died" in row["error"]
    assert "no result was ever recorded" in row["error"]
    assert str(424242) in row["error"], "say which process, so it can be looked up"


def test_a_dead_run_that_had_been_cancelled_is_reconciled_as_cancelled(
    conn, monkeypatch
):
    """'The worker died' is the wrong story when a human pressed Cancel."""
    job = enqueue_selftest(conn)
    jobs_module.claim_next(conn, "dead-worker")
    jobs_module.mark_running(conn, job["id"], 424243)
    jobs_module.request_cancel(conn, job["id"])
    monkeypatch.setattr(worker_module, "process_alive", lambda pid: False)

    run_worker_once(conn)

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "cancelled"
    assert "died" not in row["error"]
    assert "Cancelled" in row["error"]


def test_a_verdict_parked_on_disk_wins_over_the_reapers_guess(conn, monkeypatch):
    """The runner's own account beats anything the next worker can infer.

    Without this the row said "the worker running this job died, run it again" -
    advice that, for ``force_episode``, deletes the artifacts and pays for the
    whole pipeline a second time.
    """
    job = enqueue_selftest(conn)
    jobs_module.claim_next(conn, "dead-worker")
    jobs_module.mark_running(conn, job["id"], 424244)
    directory = jobs_module.ensure_job_dir(job["id"])
    Path(directory, jobs_module.PENDING_VERDICT_FILENAME).write_text(
        json.dumps(
            {
                "job_id": job["id"],
                "status": "succeeded",
                "exit_code": 0,
                "error": None,
                "written_at": time.time(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(worker_module, "process_alive", lambda pid: False)

    run_worker_once(conn)

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "succeeded"
    assert row["exit_code"] == 0
    assert jobs_module.read_pending_verdict(job["id"]) is None, (
        "an applied verdict must be cleared, or it is replayed forever"
    )


def test_reaping_settles_only_dead_rows(conn, monkeypatch):
    """A queued job is not a crash; it must survive the reaper untouched."""
    queued = enqueue_selftest(conn)
    running = enqueue_selftest(conn)
    jobs_module.claim_next(conn, "dead-worker")  # claims the older one
    jobs_module.mark_running(conn, queued["id"], 424245)
    monkeypatch.setattr(worker_module, "process_alive", lambda pid: False)

    orphans = worker_module.reap_stale_jobs(conn)

    assert orphans == []
    assert jobs_module.get_job(conn, queued["id"])["status"] == "failed"
    assert jobs_module.get_job(conn, running["id"])["status"] == "queued"


# ---------------------------------------------------------------------------
# The runner, in this process
# ---------------------------------------------------------------------------


def test_run_job_refuses_an_unsafe_job_id(queue_home):
    """The one failure it cannot report through the event log - there is none."""
    assert runner_module.run_job("../../etc/passwd") == runner_module.EXIT_BAD_JOB
    assert not os.path.exists(os.path.join(jobs_module.jobs_dir(), "..", "events.jsonl"))


def test_run_job_reports_an_unknown_job_through_its_event_log(conn):
    assert runner_module.run_job("deadbeef") == runner_module.EXIT_BAD_JOB

    events = read_events("deadbeef")
    assert events[-1]["level"] == "error"
    assert "No such job" in events[-1]["message"]


def test_run_job_refuses_to_rerun_a_terminal_job(conn):
    job = enqueue_selftest(conn)
    jobs_module.finish(conn, job["id"], "succeeded", exit_code=0)

    assert runner_module.run_job(job["id"]) == runner_module.EXIT_BAD_JOB

    events = read_events(job["id"])
    assert "refusing to run it again" in events[-1]["message"]
    assert jobs_module.get_job(conn, job["id"])["status"] == "succeeded"


def test_run_job_fails_a_row_whose_type_it_cannot_execute(conn):
    job = enqueue_selftest(conn)
    conn.execute("UPDATE jobs SET type = 'from_the_future' WHERE id = ?", (job["id"],))

    assert runner_module.run_job(job["id"]) == runner_module.EXIT_BAD_JOB

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "failed"
    assert "does not know how to execute" in row["error"]


def test_a_selftest_run_in_process_records_itself(conn):
    """The same job the worker spawns, run here, so the sink can be inspected."""
    job = enqueue_selftest(conn, seconds=0.05, steps=2)
    jobs_module.claim_next(conn, "test-worker")

    assert runner_module.run_job(job["id"]) == runner_module.EXIT_OK

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "succeeded"
    assert row["exit_code"] == 0
    assert [event["seq"] for event in read_events(job["id"])] == list(
        range(1, len(read_events(job["id"])) + 1)
    )


def test_a_cancelled_row_stops_the_runner_between_steps(conn):
    """The runner asks the wider question: cancelled *or* already terminal."""
    job = enqueue_selftest(conn, seconds=5, steps=50)
    jobs_module.claim_next(conn, "test-worker")
    jobs_module.request_cancel(conn, job["id"])

    assert runner_module.run_job(job["id"]) == runner_module.EXIT_CANCELLED

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "cancelled"
    assert row["exit_code"] == runner_module.EXIT_CANCELLED


def test_a_patched_pipeline_call_reaches_the_handler(conn, monkeypatch, podcast_dir):
    """A job type that would spend money, with the pipeline replaced by a stub.

    This is how the suite tests anything past ``selftest``: the handler imports
    its pipeline function lazily, so putting a fake module in ``sys.modules``
    means the real one is never imported, never mind called. The assertion is
    that the runner hands the transcript *from the index* to the pipeline - the
    path is never built from job params (ADR-008).
    """
    calls = []

    fake = types.ModuleType("whycast.pipeline.postprocess")

    def process_transcript_workflow(text, base_name, output_dir):
        calls.append({"text": text, "base_name": base_name, "output_dir": output_dir})
        return {"summary": "written", "blog": None}

    fake.process_transcript_workflow = process_transcript_workflow
    monkeypatch.setitem(sys.modules, "whycast.pipeline.postprocess", fake)
    # The key check imports whycast.pipeline.llm, which is the module that would
    # reach for a real key. Stub it out: this test is about the handler.
    monkeypatch.setattr(runner_module, "_require_api_key", lambda ctx: None)

    job = jobs_module.enqueue(conn, "postprocess", base_name="episode_1")
    jobs_module.claim_next(conn, "test-worker")

    assert runner_module.run_job(job["id"]) == runner_module.EXIT_OK

    assert len(calls) == 1, "the handler must call the pipeline exactly once"
    assert calls[0]["base_name"] == "episode_1"
    assert calls[0]["text"] == "SPEAKER_00: hello\nSPEAKER_01: world\n"
    assert Path(calls[0]["output_dir"]) == podcast_dir
    assert jobs_module.get_job(conn, job["id"])["status"] == "succeeded"

    messages = " ".join(event["message"] for event in read_events(job["id"]))
    assert "Post-processing produced: summary" in messages


def test_a_cost_job_without_a_key_fails_before_anything_runs(conn, monkeypatch):
    """Fail fast: twenty minutes of GPU before discovering you cannot pay is worse."""
    reached = []
    monkeypatch.setitem(
        sys.modules,
        "whycast.pipeline.postprocess",
        types.SimpleNamespace(
            process_transcript_workflow=lambda *a, **k: reached.append(a)
        ),
    )

    def refuse(ctx):
        raise runner_module.JobInputError("This job makes paid OpenAI calls, but ...")

    monkeypatch.setattr(runner_module, "_require_api_key", refuse)

    job = jobs_module.enqueue(conn, "postprocess", base_name="episode_1")
    jobs_module.claim_next(conn, "test-worker")

    assert runner_module.run_job(job["id"]) == runner_module.EXIT_BAD_JOB

    assert reached == [], "nothing may run when the key check failed"
    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "failed"
    assert "paid OpenAI calls" in row["error"]


def test_an_episode_the_index_does_not_know_is_a_bad_job_not_a_crash(conn, monkeypatch):
    # Stubbed for the same reason as above, and for one more: the real check
    # imports whycast.pipeline.llm, so without this the outcome would depend on
    # whether the machine running the tests happens to have a key configured.
    monkeypatch.setattr(runner_module, "_require_api_key", lambda ctx: None)
    job = jobs_module.enqueue(conn, "postprocess", base_name="episode_1")
    conn.execute("UPDATE jobs SET base_name = 'ghost' WHERE id = ?", (job["id"],))
    jobs_module.claim_next(conn, "test-worker")

    assert runner_module.run_job(job["id"]) == runner_module.EXIT_BAD_JOB

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "failed"
    assert "not in the index" in row["error"]


# ---------------------------------------------------------------------------
# The event sink itself
# ---------------------------------------------------------------------------


def test_the_sink_numbers_events_from_one_and_appends(tmp_path):
    path = tmp_path / "events.jsonl"
    with runner_module.JsonlEventSink(str(path)) as sink:
        sink.emit(ProgressEvent(step="a", message="first"))
        sink.emit(ProgressEvent(step="a", message="second"))
        assert sink.seq == 2

    # A second sink on the same file continues the file, never truncates it:
    # the log is the only record of what a job did.
    with runner_module.JsonlEventSink(str(path)) as sink:
        sink.emit(ProgressEvent(step="b", message="third"))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["message"] for record in records] == ["first", "second", "third"]
    assert [record["seq"] for record in records] == [1, 2, 1]


def test_the_sink_writes_one_lf_terminated_line_per_event(tmp_path):
    """A JSONL reader splits on \\n; CRLF here would be a Windows-only bug."""
    path = tmp_path / "events.jsonl"
    with runner_module.JsonlEventSink(str(path)) as sink:
        sink.emit(ProgressEvent(step="a", message="een café — 42 🎙️"))

    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    assert json.loads(raw.decode("utf-8"))["message"] == "een café — 42 🎙️"


def test_the_sink_never_raises_at_the_caller(tmp_path):
    """A logging problem must not take down a pipeline step."""
    path = tmp_path / "events.jsonl"
    circular: dict = {}
    circular["self"] = circular

    with runner_module.JsonlEventSink(str(path)) as sink:
        sink.emit(ProgressEvent(step="a", message="round and round", data=circular))
        sink.emit(ProgressEvent(step="a", message="still going"))

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert "unserialisable" in records[0]["data"]
    assert records[1]["message"] == "still going"


@pytest.mark.parametrize(
    "message, expected",
    [
        ("[1/4] Preparing audio", 0.0),
        ("[2/4] Transcribing", 0.25),
        ("[4/4] Post-processing", 0.75),
        ("no marker here", None),
        ("[0/4] impossible", None),
        ("[5/4] impossible", None),
        ("[1/0] impossible", None),
        ("", None),
    ],
)
def test_step_markers_become_progress(message, expected):
    assert runner_module.step_marker_progress(message) == expected


def test_a_pipeline_supplied_progress_is_never_overwritten(tmp_path):
    path = tmp_path / "events.jsonl"
    with runner_module.JsonlEventSink(str(path)) as sink:
        sink.emit(ProgressEvent(step="a", message="[2/4] halfway", progress=0.9))

    assert json.loads(path.read_text(encoding="utf-8"))["progress"] == 0.9


# ---------------------------------------------------------------------------
# Configuration (ADR-007: environment variables, no parallel config store)
# ---------------------------------------------------------------------------


def test_the_runner_reads_its_paths_from_the_environment(queue_home, podcast_dir):
    assert runner_module.db_path_from_env() == str(queue_home)
    assert runner_module.output_dir_from_env() == str(podcast_dir)
    assert runner_module.rssfeed_from_env() == runner_module.DEFAULT_RSSFEED


def test_the_podcast_directory_wins_over_the_cli_output_directory(
    queue_home, podcast_dir, tmp_path, monkeypatch
):
    """Artifacts written anywhere else would be invisible to the web UI."""
    monkeypatch.setenv("WHYCAST_OUTPUT_DIR", str(tmp_path / "elsewhere"))
    assert runner_module.output_dir_from_env() == str(podcast_dir)

    monkeypatch.delenv("WHYCAST_PODCAST_DIR")
    assert runner_module.output_dir_from_env() == str(tmp_path / "elsewhere")


def test_the_feed_url_cannot_come_from_job_params(conn, monkeypatch):
    """A URL in a job row would be a second configuration store a request can set."""
    monkeypatch.setenv("WHYCAST_RSSFEED", "https://example.invalid/feed.xml")
    assert runner_module.rssfeed_from_env() == "https://example.invalid/feed.xml"
    assert "rssfeed" not in str(jobs_module.JOB_TYPES["fetch_all"])
