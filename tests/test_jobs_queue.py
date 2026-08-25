"""
Tests for :mod:`webui.jobs` - the job queue itself (ADR-008, TASK-003 phase 2).

This file tests the *queue* and nothing else: no process is spawned, no HTTP
request is made, no pipeline function is called and no OpenAI call is possible
from here - :mod:`webui.jobs` does not import the pipeline at all. Every test
runs against a throwaway SQLite database and a throwaway job-log directory, so
the operator's real ``webui/whycast_webui.db`` and real ``logs/jobs`` are never
touched.

The three properties worth stating up front, because they are what the rest of
phase 2 is built on:

* **A job is handed out exactly once.** Proven under concurrent threads *and*
  under two operating-system processes, because in production the web server
  and the worker are separate processes with separate connections. A proof that
  runs everything through the in-process lock would prove nothing about that.
* **Status only moves one way.** queued -> running -> terminal, and a terminal
  job is immovable: the *first* verdict is the true one.
* **Job history is not rebuildable from disk.** So the ``jobs`` table has to
  survive everything the episode index does to the database it shares, and vice
  versa.
"""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webui import db as webui_db  # noqa: E402
from webui import jobs as jobs_module  # noqa: E402
from whycast.events import CollectSink, use_sink  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def queue_home(tmp_path, monkeypatch):
    """A scratch database path and job-log directory, wired through the env.

    ``WHYCAST_JOBS_DIR`` is read on every :func:`webui.jobs.jobs_dir` call, so
    forgetting it would write verdict sidecars into the operator's real
    ``logs/jobs``.
    """
    jobs_home = tmp_path / "jobs"
    jobs_home.mkdir()
    db_path = tmp_path / "queue.db"
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(jobs_home))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(db_path))
    return db_path


@pytest.fixture
def conn(queue_home):
    """A connection with both schemas on it, closed afterwards."""
    connection = webui_db.init_db(str(queue_home))
    jobs_module.ensure_job_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture
def podcast_dir(tmp_path):
    """A synthetic podcast directory the index can be built from."""
    root = tmp_path / "podcasts"
    root.mkdir()
    (root / "episode_1.mp3").write_bytes(b"fake-mp3")
    (root / "episode_1.txt").write_text("transcript one\n", encoding="utf-8")
    (root / "episode_2.mp3").write_bytes(b"fake-mp3")
    return root


def open_second_connection(queue_home):
    """Another connection to the same file, as a second process would have."""
    connection = webui_db.init_db(str(queue_home))
    jobs_module.ensure_job_schema(connection)
    return connection


def statuses(connection):
    """``{job_id: status}`` straight out of the table, bypassing the helpers."""
    return {
        row["id"]: row["status"]
        for row in connection.execute("SELECT id, status FROM jobs").fetchall()
    }


# ---------------------------------------------------------------------------
# enqueue: what the queue will and will not accept
# ---------------------------------------------------------------------------


def test_enqueue_returns_a_queued_job_with_the_type_flags(conn):
    job = jobs_module.enqueue(conn, "selftest")

    assert job["status"] == "queued"
    assert job["type"] == "selftest"
    assert job["is_terminal"] is False
    assert job["cost"] is False and job["gpu"] is False
    assert job["known_type"] is True
    assert job["params"] == {}
    assert job["pid"] is None and job["worker_id"] is None
    assert job["created_at_iso"]


def test_enqueue_rejects_an_unknown_job_type(conn):
    with pytest.raises(ValueError) as excinfo:
        jobs_module.enqueue(conn, "mine_dogecoin")

    # The message has to name the known types: this is what a caller sees.
    assert "mine_dogecoin" in str(excinfo.value)
    assert "selftest" in str(excinfo.value)
    assert jobs_module.list_jobs(conn) == []


@pytest.mark.parametrize("job_type", sorted(jobs_module.JOB_TYPES))
def test_every_known_type_is_accepted_and_keeps_its_flags(conn, job_type):
    spec = jobs_module.JOB_TYPES[job_type]
    base = "episode_1" if spec["requires_base_name"] else None

    job = jobs_module.enqueue(conn, job_type, base_name=base)

    assert job["type"] == job_type
    assert job["cost"] == bool(spec["cost"])
    assert job["gpu"] == bool(spec["gpu"])


def test_episode_scoped_types_demand_a_base_name(conn):
    for job_type, spec in jobs_module.JOB_TYPES.items():
        if not spec["requires_base_name"]:
            continue
        with pytest.raises(ValueError, match="base_name"):
            jobs_module.enqueue(conn, job_type)
        with pytest.raises(ValueError, match="base_name"):
            jobs_module.enqueue(conn, job_type, base_name="   ")


@pytest.mark.parametrize(
    "base_name",
    [
        "../secrets",
        r"..\secrets",
        "sub/episode_1",
        r"sub\episode_1",
        "..",
        ".",
        "episode\0_1",
    ],
)
def test_a_base_name_that_looks_like_a_path_is_refused(conn, base_name):
    """Defence in depth: base_name is an index key, never a path fragment."""
    with pytest.raises(ValueError, match="not a path|path"):
        jobs_module.enqueue(conn, "postprocess", base_name=base_name)
    assert jobs_module.list_jobs(conn) == []


def test_base_name_is_stripped_but_otherwise_kept_verbatim(conn):
    job = jobs_module.enqueue(conn, "postprocess", base_name="  episode_1  ")
    assert job["base_name"] == "episode_1"


@pytest.mark.parametrize(
    "params",
    [
        ["not", "a", "dict"],
        "a string",
        42,
    ],
)
def test_params_must_be_a_dict(conn, params):
    with pytest.raises(ValueError, match="params must be a dict"):
        jobs_module.enqueue(conn, "selftest", params=params)


def test_params_keys_must_be_strings(conn):
    with pytest.raises(ValueError, match="keys"):
        jobs_module.enqueue(conn, "selftest", params={1: "one"})


def test_params_must_be_json_serialisable(conn):
    with pytest.raises(ValueError, match="JSON"):
        jobs_module.enqueue(conn, "selftest", params={"when": object()})


def test_params_round_trip_including_unicode(conn):
    payload = {"seconds": 0.5, "note": "één café — 42", "nested": {"a": [1, 2]}}
    job = jobs_module.enqueue(conn, "selftest", params=payload)

    assert jobs_module.get_job(conn, job["id"])["params"] == payload


def test_unreadable_params_degrade_to_empty_rather_than_crashing(conn):
    """A hand-edited row must not take the whole dashboard down with it."""
    job = jobs_module.enqueue(conn, "selftest")
    conn.execute("UPDATE jobs SET params = ? WHERE id = ?", ("{not json", job["id"]))

    with use_sink(CollectSink()) as sink:
        reloaded = jobs_module.get_job(conn, job["id"])

    assert reloaded["params"] == {}
    assert any(event.level == "warning" for event in sink.events)


# ---------------------------------------------------------------------------
# claim_next: exactly once, under threads and under processes
# ---------------------------------------------------------------------------


def test_claim_next_takes_the_oldest_job_first(conn):
    first = jobs_module.enqueue(conn, "selftest", params={"n": 1})
    second = jobs_module.enqueue(conn, "selftest", params={"n": 2})

    claimed = jobs_module.claim_next(conn, "worker-a")

    assert claimed["id"] == first["id"]
    assert claimed["status"] == "running"
    assert claimed["worker_id"] == "worker-a"
    assert claimed["pid"] == os.getpid(), "the claimer publishes its own pid"
    assert claimed["started_at"] is not None
    assert jobs_module.get_job(conn, second["id"])["status"] == "queued"


def test_claim_next_returns_none_on_an_empty_queue(conn):
    assert jobs_module.claim_next(conn, "worker-a") is None


def test_claim_next_never_hands_out_a_running_or_terminal_job(conn):
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.claim_next(conn, "worker-a")

    assert jobs_module.claim_next(conn, "worker-b") is None, "running is not claimable"

    jobs_module.finish(conn, job["id"], "succeeded", exit_code=0)
    assert jobs_module.claim_next(conn, "worker-c") is None, "terminal is not claimable"


@pytest.mark.parametrize("worker_id", ["", "   ", None, 7])
def test_claim_next_demands_a_worker_id(conn, worker_id):
    with pytest.raises(ValueError, match="worker_id"):
        jobs_module.claim_next(conn, worker_id)


@pytest.mark.parametrize("pid", [0, -1, True, "1234"])
def test_claim_next_demands_a_real_pid(conn, pid):
    with pytest.raises(ValueError, match="pid"):
        jobs_module.claim_next(conn, "worker-a", pid=pid)


def test_concurrent_threads_each_get_a_different_job(queue_home, conn):
    """Eight threads, eight connections, one job each - never the same one twice.

    Driven through ``_claim_next_sql`` on *separate* connections on purpose.
    ``claim_next`` serialises on the module lock, so racing it would prove only
    that the lock works; what has to hold is that SQLite's ``BEGIN IMMEDIATE``
    plus ``UPDATE ... RETURNING`` is safe between connections, because in
    production those live in different processes.
    """
    workers = 8
    jobs_per_worker = 5
    expected = {
        jobs_module.enqueue(conn, "selftest", params={"n": n})["id"]
        for n in range(workers * jobs_per_worker)
    }

    connections = [open_second_connection(queue_home) for _ in range(workers)]
    barrier = threading.Barrier(workers)
    claimed: list = []
    errors: list = []
    lock = threading.Lock()

    def drain(index):
        connection = connections[index]
        mine = []
        try:
            barrier.wait(timeout=30)
            while True:
                job = jobs_module._claim_next_sql(connection, f"worker-{index}")
                if job is None:
                    break
                mine.append(job["id"])
        except BaseException as exc:  # pragma: no cover - reported below
            with lock:
                errors.append(exc)
        with lock:
            claimed.extend(mine)

    threads = [threading.Thread(target=drain, args=(i,)) for i in range(workers)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), "a claiming thread hung"
    finally:
        for connection in connections:
            connection.close()

    assert errors == []
    assert len(claimed) == len(set(claimed)), "a job was handed out twice"
    assert set(claimed) == expected, "a job was never handed out"
    assert set(statuses(conn).values()) == {"running"}


#: The child of :func:`test_two_processes_each_get_a_different_job`. Written to
#: a file rather than passed to ``python -c`` so a failure has a real traceback
#: with real line numbers.
_CLAIMER_SOURCE = '''
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.environ["WHYCAST_WEBUI_DB"] = sys.argv[2]
os.environ["WHYCAST_JOBS_DIR"] = sys.argv[3]

from webui import db as webui_db
from webui import jobs

worker_id = sys.argv[4]
go_file = sys.argv[5]
out_file = sys.argv[6]
ready_file = sys.argv[7]

conn = webui_db.init_db(sys.argv[2])
jobs.ensure_job_schema(conn)

# Everything expensive - starting Python, importing webui - is behind us now.
# The parent waits for this file before it says go, so both children are
# actually inside claim_next at the same time and the race is real.
with open(ready_file, "w", encoding="utf-8") as stream:
    stream.write("ready")

deadline = time.monotonic() + 60
while not os.path.exists(go_file):
    if time.monotonic() > deadline:
        raise SystemExit("the parent never said go")
    time.sleep(0.005)

claimed = []
idle = 0
while idle < 5:
    job = jobs.claim_next(conn, worker_id)
    if job is None:
        idle += 1
        time.sleep(0.02)
        continue
    idle = 0
    claimed.append(job["id"])
    # Widen the window the two processes overlap in. Without it the first one
    # to start can drain the whole queue before the other takes its first turn,
    # and the test would pass without ever having raced anything.
    time.sleep(0.004)

conn.close()
with open(out_file, "w", encoding="utf-8") as stream:
    json.dump(claimed, stream)
'''


def test_two_processes_each_get_a_different_job(tmp_path, queue_home, conn):
    """The real production shape: two processes, two connections, one queue.

    The web server and the worker never share a connection, so the in-process
    lock cannot be what keeps a job from being claimed twice. This is the test
    that says so.
    """
    total = 40
    expected = {
        jobs_module.enqueue(conn, "selftest", params={"n": n})["id"]
        for n in range(total)
    }

    script = tmp_path / "claimer.py"
    script.write_text(_CLAIMER_SOURCE, encoding="utf-8")
    go_file = tmp_path / "go"
    outputs = [tmp_path / "claimed-a.json", tmp_path / "claimed-b.json"]
    ready = [tmp_path / "ready-a", tmp_path / "ready-b"]

    children = [
        subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(REPO_ROOT),
                str(queue_home),
                os.environ["WHYCAST_JOBS_DIR"],
                f"worker-{name}",
                str(go_file),
                str(out),
                str(flag),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(REPO_ROOT),
        )
        for name, out, flag in zip("ab", outputs, ready)
    ]
    per_child = []
    try:
        deadline = time.monotonic() + 120
        while not all(flag.exists() for flag in ready):
            if time.monotonic() > deadline:  # pragma: no cover - a stuck child
                pytest.fail("the claimer processes never reported ready")
            assert all(child.poll() is None for child in children), "a claimer died"
            time.sleep(0.02)
        # Both are past their imports and spinning on this file, so the claims
        # really do overlap rather than running one after the other.
        go_file.write_text("go", encoding="utf-8")
        for child in children:
            output = child.communicate(timeout=180)[0]
            assert child.returncode == 0, f"claimer failed:\n{output}"
    finally:
        for child in children:
            if child.poll() is None:  # pragma: no cover - only on a timeout
                child.kill()
                child.wait(timeout=30)

    claimed = []
    for out in outputs:
        mine = json.loads(out.read_text(encoding="utf-8"))
        per_child.append(len(mine))
        claimed.extend(mine)

    assert len(claimed) == len(set(claimed)), "two processes claimed the same job"
    assert set(claimed) == expected, "a job was never claimed"
    assert set(statuses(conn).values()) == {"running"}
    assert min(per_child) > 0, (
        f"one process drained the whole queue ({per_child}); the claims never "
        f"overlapped, so this run proved nothing about the race"
    )


# ---------------------------------------------------------------------------
# Status transitions: one direction only
# ---------------------------------------------------------------------------


def test_the_normal_path_is_queued_then_running_then_terminal(conn):
    job = jobs_module.enqueue(conn, "selftest")
    assert jobs_module.get_job(conn, job["id"])["status"] == "queued"

    jobs_module.claim_next(conn, "worker-a")
    assert jobs_module.get_job(conn, job["id"])["status"] == "running"

    jobs_module.mark_running(conn, job["id"], 4242)
    assert jobs_module.get_job(conn, job["id"])["pid"] == 4242

    jobs_module.finish(conn, job["id"], "succeeded", exit_code=0)
    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "succeeded"
    assert row["is_terminal"] is True
    assert row["exit_code"] == 0
    assert row["finished_at"] is not None
    assert row["duration_seconds"] is not None


@pytest.mark.parametrize("first", ["succeeded", "failed", "cancelled"])
@pytest.mark.parametrize("second", ["succeeded", "failed", "cancelled"])
def test_a_terminal_job_never_moves_again(conn, first, second):
    """The first verdict stands, and the attempt to overwrite it is announced."""
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.claim_next(conn, "worker-a")
    jobs_module.finish(conn, job["id"], first, exit_code=7, error="the true reason")
    before = jobs_module.get_job(conn, job["id"])

    with use_sink(CollectSink()) as sink:
        jobs_module.finish(conn, job["id"], second, exit_code=0, error="a later guess")

    after = jobs_module.get_job(conn, job["id"])
    assert after["status"] == first
    assert after["exit_code"] == 7
    assert after["error"] == "the true reason"
    assert after["finished_at"] == before["finished_at"]
    assert any(
        event.level == "warning" and "already" in event.message
        for event in sink.events
    ), "silently dropping a second verdict would hide a real conflict"


@pytest.mark.parametrize("status", ["queued", "running", "pending", "", None, 3])
def test_finish_refuses_a_non_terminal_destination(conn, status):
    job = jobs_module.enqueue(conn, "selftest")
    with pytest.raises(ValueError, match="status"):
        jobs_module.finish(conn, job["id"], status)
    assert jobs_module.get_job(conn, job["id"])["status"] == "queued"


def test_finish_refuses_an_unknown_job(conn):
    with pytest.raises(ValueError, match="No such job"):
        jobs_module.finish(conn, "no-such-job", "succeeded")


@pytest.mark.parametrize("exit_code", [True, 1.5, "0"])
def test_finish_demands_an_int_exit_code(conn, exit_code):
    job = jobs_module.enqueue(conn, "selftest")
    with pytest.raises(ValueError, match="exit_code"):
        jobs_module.finish(conn, job["id"], "failed", exit_code=exit_code)


def test_finish_can_settle_a_job_that_never_ran(conn):
    """A queued job cancelled before it started still gets a finished_at."""
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.finish(conn, job["id"], "cancelled", error="never started")

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "cancelled"
    assert row["started_at"] is not None and row["finished_at"] is not None


def test_a_long_error_is_truncated_with_a_pointer_to_the_event_log(conn):
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.finish(conn, job["id"], "failed", exit_code=1, error="x" * 5000)

    error = jobs_module.get_job(conn, job["id"])["error"]
    assert len(error) < 5000
    assert "events.jsonl" in error


def test_mark_running_never_resurrects_a_terminal_job(conn):
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.claim_next(conn, "worker-a")
    jobs_module.mark_running(conn, job["id"], 111)
    jobs_module.finish(conn, job["id"], "succeeded", exit_code=0)

    jobs_module.mark_running(conn, job["id"], 222)

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "succeeded"
    assert row["pid"] == 111, "the pid must keep meaning 'what ran this job'"


def test_mark_running_promotes_a_queued_job(conn):
    """A runner started by hand still leaves a consistent row behind."""
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.mark_running(conn, job["id"], 999)

    row = jobs_module.get_job(conn, job["id"])
    assert row["status"] == "running"
    assert row["pid"] == 999
    assert row["started_at"] is not None


@pytest.mark.parametrize("pid", [0, -3, True, None, "123"])
def test_mark_running_demands_a_real_pid(conn, pid):
    job = jobs_module.enqueue(conn, "selftest")
    with pytest.raises(ValueError, match="pid"):
        jobs_module.mark_running(conn, job["id"], pid)


def test_mark_running_refuses_an_unknown_job(conn):
    with pytest.raises(ValueError, match="No such job"):
        jobs_module.mark_running(conn, "no-such-job", 1234)


def test_the_status_column_itself_refuses_anything_unknown(conn):
    """Belt and braces: the CHECK constraint, not just the Python validation."""
    job = jobs_module.enqueue(conn, "selftest")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE jobs SET status = 'exploded' WHERE id = ?", (job["id"],))


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancelling_a_queued_job_settles_it_outright(conn):
    job = jobs_module.enqueue(conn, "selftest")

    cancelled = jobs_module.request_cancel(conn, job["id"])

    assert cancelled["status"] == "cancelled"
    assert cancelled["is_terminal"] is True
    assert cancelled["cancel_requested"] is True
    assert cancelled["finished_at"] is not None
    assert cancelled["error"], "a cancelled job says why it never ran"
    assert jobs_module.claim_next(conn, "worker-a") is None


def test_cancelling_a_running_job_only_raises_the_flag(conn):
    """The job is only cancelled once the process has actually stopped."""
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.claim_next(conn, "worker-a")

    flagged = jobs_module.request_cancel(conn, job["id"])

    assert flagged["status"] == "running", "nothing may report success early"
    assert flagged["cancel_requested"] is True
    assert flagged["finished_at"] is None
    assert jobs_module.cancel_requested(conn, job["id"]) is True

    # ...and the runner or the worker then settles it.
    jobs_module.finish(conn, job["id"], "cancelled", exit_code=130)
    assert jobs_module.get_job(conn, job["id"])["status"] == "cancelled"


def test_cancelling_a_terminal_job_changes_nothing(conn):
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.claim_next(conn, "worker-a")
    jobs_module.finish(conn, job["id"], "succeeded", exit_code=0)
    before = jobs_module.get_job(conn, job["id"])

    with use_sink(CollectSink()) as sink:
        after = jobs_module.request_cancel(conn, job["id"])

    assert after["status"] == "succeeded"
    assert after["cancel_requested"] is False
    assert after["finished_at"] == before["finished_at"]
    assert any(event.level == "warning" for event in sink.events)


def test_cancel_of_an_unknown_job_raises(conn):
    with pytest.raises(ValueError, match="No such job"):
        jobs_module.request_cancel(conn, "no-such-job")


def test_cancel_requested_is_false_for_a_job_nobody_cancelled(conn):
    job = jobs_module.enqueue(conn, "selftest")
    assert jobs_module.cancel_requested(conn, job["id"]) is False


def test_cancel_requested_raises_for_an_unknown_job(conn):
    """Returning False would let a cancelled run continue to completion."""
    with pytest.raises(ValueError, match="No such job"):
        jobs_module.cancel_requested(conn, "no-such-job")


# ---------------------------------------------------------------------------
# Reads: active_job and list_jobs
# ---------------------------------------------------------------------------


def test_active_job_is_none_when_nothing_runs(conn):
    jobs_module.enqueue(conn, "selftest")
    assert jobs_module.active_job(conn) is None


def test_active_job_is_the_running_one(conn):
    jobs_module.enqueue(conn, "selftest", params={"n": 1})
    second = jobs_module.enqueue(conn, "selftest", params={"n": 2})
    claimed = jobs_module.claim_next(conn, "worker-a")

    active = jobs_module.active_job(conn)

    assert active["id"] == claimed["id"]
    assert active["id"] != second["id"]


def test_active_job_reports_the_oldest_and_warns_when_two_run(conn):
    """Two running jobs means two processes on one GPU. It must be visible."""
    first = jobs_module.enqueue(conn, "selftest", params={"n": 1})
    second = jobs_module.enqueue(conn, "selftest", params={"n": 2})
    conn.execute(
        "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
        (1000.0, first["id"]),
    )
    conn.execute(
        "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
        (2000.0, second["id"]),
    )

    with use_sink(CollectSink()) as sink:
        active = jobs_module.active_job(conn)

    assert active["id"] == first["id"]
    assert any(
        event.level == "warning" and "running at once" in event.message
        for event in sink.events
    )


def test_a_running_job_reports_a_duration_that_is_still_growing(conn):
    jobs_module.enqueue(conn, "selftest")
    claimed = jobs_module.claim_next(conn, "worker-a")

    assert claimed["duration_seconds"] is not None
    assert claimed["finished_at"] is None


def test_list_jobs_is_newest_first(conn):
    created = [
        jobs_module.enqueue(conn, "selftest", params={"n": n})["id"] for n in range(5)
    ]
    assert [job["id"] for job in jobs_module.list_jobs(conn)] == list(reversed(created))


def test_list_jobs_filters_by_one_status(conn):
    queued = jobs_module.enqueue(conn, "selftest", params={"n": 1})
    running = jobs_module.enqueue(conn, "selftest", params={"n": 2})
    conn.execute("UPDATE jobs SET status = 'running' WHERE id = ?", (running["id"],))

    assert [job["id"] for job in jobs_module.list_jobs(conn, status="queued")] == [
        queued["id"]
    ]
    assert [job["id"] for job in jobs_module.list_jobs(conn, status="running")] == [
        running["id"]
    ]


def test_list_jobs_filters_by_several_statuses(conn):
    ids = {}
    for status in ("queued", "running", "succeeded", "failed", "cancelled"):
        job = jobs_module.enqueue(conn, "selftest", params={"s": status})
        ids[status] = job["id"]
        if status != "queued":
            conn.execute(
                "UPDATE jobs SET status = ? WHERE id = ?", (status, job["id"])
            )

    found = jobs_module.list_jobs(conn, status=["failed", "cancelled", "failed"])

    assert {job["id"] for job in found} == {ids["failed"], ids["cancelled"]}


def test_list_jobs_rejects_an_unknown_status(conn):
    with pytest.raises(ValueError, match="Unknown status"):
        jobs_module.list_jobs(conn, status="exploded")


def test_list_jobs_rejects_an_empty_filter(conn):
    """An empty list would silently match nothing; None means 'all'."""
    with pytest.raises(ValueError, match="empty status filter"):
        jobs_module.list_jobs(conn, status=[])


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_list_jobs_rejects_a_bad_limit(conn, limit):
    with pytest.raises(ValueError, match="limit"):
        jobs_module.list_jobs(conn, limit=limit)


def test_list_jobs_honours_the_limit_and_none(conn):
    for n in range(7):
        jobs_module.enqueue(conn, "selftest", params={"n": n})

    assert len(jobs_module.list_jobs(conn, limit=3)) == 3
    assert len(jobs_module.list_jobs(conn, limit=None)) == 7


def test_get_job_is_none_for_an_unknown_id(conn):
    assert jobs_module.get_job(conn, "no-such-job") is None


def test_a_row_of_an_unknown_type_assumes_the_expensive_answer(conn):
    """A row written by a newer build must never make the UI under-warn."""
    job = jobs_module.enqueue(conn, "selftest")
    conn.execute("UPDATE jobs SET type = 'from_the_future' WHERE id = ?", (job["id"],))

    row = jobs_module.get_job(conn, job["id"])

    assert row["known_type"] is False
    assert row["cost"] is True and row["gpu"] is True
    assert row["label"] == "from_the_future"


# ---------------------------------------------------------------------------
# Event log locations - the one place a job id becomes a path
# ---------------------------------------------------------------------------


def test_events_path_lives_under_the_jobs_dir(queue_home):
    path = jobs_module.events_path("abc123")
    assert os.path.dirname(os.path.dirname(path)) == jobs_module.jobs_dir()
    assert os.path.basename(path) == "events.jsonl"
    assert not os.path.exists(os.path.dirname(path)), "asking must create nothing"


@pytest.mark.parametrize(
    "job_id",
    [
        "../../etc/passwd",
        r"..\..\.env",
        "a/b",
        "..",
        ".",
        "",
        "x" * 65,
        "job id with spaces",
        None,
        42,
    ],
)
def test_an_unsafe_job_id_never_becomes_a_path(queue_home, job_id):
    with pytest.raises(ValueError, match="safe path segment"):
        jobs_module.events_path(job_id)
    with pytest.raises(ValueError, match="safe path segment"):
        jobs_module.ensure_job_dir(job_id)


def test_ensure_job_dir_creates_exactly_one_directory(queue_home):
    directory = jobs_module.ensure_job_dir("abc123")

    assert os.path.isdir(directory)
    assert os.path.dirname(directory) == jobs_module.jobs_dir()
    # Idempotent: the runner calls it on every start.
    assert jobs_module.ensure_job_dir("abc123") == directory


# ---------------------------------------------------------------------------
# Sharing a database with the episode index
# ---------------------------------------------------------------------------


def test_the_jobs_table_survives_a_rescan(conn, podcast_dir):
    """Job history is the only record a run happened; a rescan must not touch it."""
    job = jobs_module.enqueue(conn, "postprocess", base_name="episode_1")
    jobs_module.claim_next(conn, "worker-a")
    jobs_module.finish(conn, job["id"], "failed", exit_code=1, error="something broke")

    webui_db.rescan(conn, str(podcast_dir))

    row = jobs_module.get_job(conn, job["id"])
    assert row is not None
    assert row["status"] == "failed"
    assert row["error"] == "something broke"


def test_the_index_survives_the_queue_schema(conn, podcast_dir):
    """And the other way round: creating the queue must not disturb the index."""
    webui_db.rescan(conn, str(podcast_dir))
    before = {ep["base_name"] for ep in webui_db.list_episodes(conn)}
    assert before, "the fixture directory should index at least one episode"

    jobs_module.ensure_job_schema(conn)  # idempotent, called on every startup
    jobs_module.enqueue(conn, "selftest")

    after = {ep["base_name"] for ep in webui_db.list_episodes(conn)}
    assert after == before


def test_rebuilding_the_index_schema_leaves_job_history_alone(
    queue_home, podcast_dir, monkeypatch
):
    """An index version bump drops episodes and keeps jobs. That is the contract.

    ``webui.db`` owns ``episodes``/``artifacts``/``unmatched`` and its own meta
    keys; the queue carries a separate ``jobs_schema_version`` precisely so this
    can never take job history with it.
    """
    first = webui_db.init_db(str(queue_home))
    try:
        jobs_module.ensure_job_schema(first)
        job = jobs_module.enqueue(first, "selftest", params={"keep": "me"})
        webui_db.rescan(first, str(podcast_dir))
        assert webui_db.list_episodes(first)
    finally:
        first.close()

    monkeypatch.setattr(webui_db, "SCHEMA_VERSION", webui_db.SCHEMA_VERSION + 1)
    second = webui_db.init_db(str(queue_home))
    try:
        assert webui_db.list_episodes(second) == [], "the index should be rebuilt"
        row = jobs_module.get_job(second, job["id"])
        assert row is not None, "job history is not rebuildable from disk"
        assert row["params"] == {"keep": "me"}
    finally:
        second.close()


def test_a_newer_queue_schema_is_left_untouched(queue_home, conn):
    """An older build pointed at a newer database must not 'fix' it."""
    job = jobs_module.enqueue(conn, "selftest")
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('jobs_schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(jobs_module.JOBS_SCHEMA_VERSION + 1),),
    )

    with use_sink(CollectSink()) as sink:
        jobs_module.ensure_job_schema(conn)

    assert any(event.level == "warning" for event in sink.events)
    assert jobs_module.get_job(conn, job["id"]) is not None
    stored = conn.execute(
        "SELECT value FROM meta WHERE key = 'jobs_schema_version'"
    ).fetchone()["value"]
    assert stored == str(jobs_module.JOBS_SCHEMA_VERSION + 1)


def test_the_queue_refuses_to_work_inside_someone_elses_transaction(conn):
    """The shared lock is re-entrant, so this is the guard that actually holds."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(jobs_module.JobQueueError, match="transaction"):
            jobs_module.ensure_job_schema(conn)
        with pytest.raises(jobs_module.JobQueueError, match="transaction"):
            jobs_module.claim_next(conn, "worker-a")
    finally:
        conn.execute("ROLLBACK")


# ---------------------------------------------------------------------------
# The verdict sidecar
# ---------------------------------------------------------------------------


def test_a_verdict_the_database_refuses_is_parked_on_disk(conn, monkeypatch):
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.claim_next(conn, "worker-a")
    monkeypatch.setattr(jobs_module, "_FINISH_BACKOFF_SECONDS", (0.0, 0.0))

    def refuse(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(jobs_module, "finish", refuse)

    assert jobs_module.finish_or_record(conn, job["id"], "succeeded", 0, None) is False

    parked = jobs_module.read_pending_verdict(job["id"])
    assert parked["status"] == "succeeded"
    assert parked["exit_code"] == 0
    assert os.path.isfile(
        os.path.join(
            jobs_module.jobs_dir(), job["id"], jobs_module.PENDING_VERDICT_FILENAME
        )
    )


def test_a_verdict_that_lands_leaves_no_sidecar_behind(conn):
    job = jobs_module.enqueue(conn, "selftest")
    jobs_module.claim_next(conn, "worker-a")
    directory = jobs_module.ensure_job_dir(job["id"])
    Path(directory, jobs_module.PENDING_VERDICT_FILENAME).write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )

    assert jobs_module.finish_or_record(conn, job["id"], "succeeded", 0, None) is True
    assert jobs_module.read_pending_verdict(job["id"]) is None


@pytest.mark.parametrize(
    "payload",
    ["not json at all", '"a string"', '{"status": "queued"}', "[]"],
)
def test_a_damaged_sidecar_reads_as_no_verdict(queue_home, payload):
    """Best effort: a corrupt file must never stop a worker from starting."""
    directory = jobs_module.ensure_job_dir("abc123")
    Path(directory, jobs_module.PENDING_VERDICT_FILENAME).write_text(
        payload, encoding="utf-8"
    )

    assert jobs_module.read_pending_verdict("abc123") is None


def test_reading_a_verdict_for_an_unsafe_job_id_is_none_not_a_traversal(queue_home):
    assert jobs_module.read_pending_verdict("../../etc/passwd") is None
    jobs_module.clear_pending_verdict("../../etc/passwd")  # must not raise
