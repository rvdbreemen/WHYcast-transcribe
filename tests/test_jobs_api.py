"""
Tests for the job HTTP API of the WHYcast web UI (ADR-008, TASK-003 phase 2).

The routes in :mod:`webui.app` that create, list, cancel and stream jobs. Two
things this file is careful about:

* **Nothing here spends money or claims the GPU.** The only job type any worker
  in this file runs is ``selftest`` (``cost=False``, ``gpu=False``), and
  :func:`run_the_queue` refuses to start a worker if the queue holds anything
  else. Tests that need an expensive *row* - to check that ``cost`` is reported
  - enqueue one and never run it.
* **The SSE tests use a real job.** ``events.jsonl`` is written by a real
  ``python -m webui.runner`` child, so the sequence numbers, the message
  shapes and the terminal transition are the ones production produces, not a
  fixture's idea of them.

Everything runs against a synthetic podcast directory and temporary SQLite and
job-log directories; the operator's real ``podcasts/``, ``webui/
whycast_webui.db`` and ``logs/jobs`` are never touched.
"""

import json
import os
import socket
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webui import app as app_module  # noqa: E402
from webui import jobs as jobs_module  # noqa: E402
from webui import worker as worker_module  # noqa: E402

FILES = {
    "episode_1.mp3": b"fake-mp3-bytes",
    "episode_1.txt": b"transcript of episode one\n",
    "episode_1_summary.txt": b"summary of episode one\n",
    "episode_2.mp3": b"fake-mp3-bytes",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def podcast_dir(tmp_path):
    root = tmp_path / "podcasts"
    root.mkdir()
    for name, content in FILES.items():
        (root / name).write_bytes(content)
    return root


@pytest.fixture
def job_env(tmp_path, podcast_dir, monkeypatch):
    """Point every ADR-007 variable at this test's temporary directories.

    ``WHYCAST_JOBS_DIR`` is read on every call, so a test that forgot it would
    tail the operator's real ``logs/jobs``. The host variables are cleared
    rather than inherited: an off-loopback bind disables the Host allowlist,
    which would make the guard tests depend on file order.
    """
    jobs_home = tmp_path / "jobs"
    jobs_home.mkdir()
    db_path = tmp_path / "index.db"
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(jobs_home))
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcast_dir))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(db_path))
    monkeypatch.delenv("WHYCAST_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("WHYCAST_WEBUI_HOST", raising=False)
    monkeypatch.delenv("WHYCAST_WEBUI_ALLOWED_HOSTS", raising=False)
    worker_module._ORPHAN_REFUSALS.clear()
    yield db_path
    worker_module._ORPHAN_REFUSALS.clear()


@pytest.fixture
def client(job_env, podcast_dir):
    """A TestClient over a throwaway index, queue and job-log directory."""
    app = app_module.create_app(podcast_dir=str(podcast_dir), db_path=str(job_env))
    with TestClient(app) as test_client:
        test_client.app_object = app  # type: ignore[attr-defined]
        yield test_client


@pytest.fixture
def live_server(job_env, podcast_dir):
    """A real uvicorn on a real socket, for the one test that needs live bytes.

    Yields ``(base_url, app)``. ``TestClient`` runs the ASGI app to completion
    before it hands back a response, so it cannot tell a live stream from a
    buffered one; only a real server and a real socket can.
    """
    uvicorn = pytest.importorskip("uvicorn", reason="the live-stream test needs uvicorn")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    app = app_module.create_app(podcast_dir=str(podcast_dir), db_path=str(job_env))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                break
            time.sleep(0.05)
        else:  # pragma: no cover - only on a machine that cannot bind loopback
            pytest.fail("uvicorn did not start")
        yield f"http://127.0.0.1:{port}", app
    finally:
        server.should_exit = True
        thread.join(timeout=30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def enqueue(client, job_type="selftest", **kwargs):
    """Create a job through the API; fail loudly if it was not accepted."""
    response = client.post("/api/jobs", json={"type": job_type, **kwargs})
    assert response.status_code == 201, response.text
    return response.json()


def run_the_queue(conn, poll_interval=0.05):
    """Run the queue to completion with a real worker, after the cost guard.

    The guard is the important half: ``selftest`` is the only type that spends
    nothing and touches no GPU, and no environment variable can make the rest
    safe - the runner child loads ``.env`` itself. So the queue's contents are
    checked instead.
    """
    expensive = [
        job
        for job in jobs_module.list_jobs(conn, status="queued", limit=None)
        if job["type"] != "selftest"
    ]
    assert not expensive, (
        "COST GUARD: refusing to start a worker while the queue holds "
        + ", ".join(f"{job['type']} ({job['id']})" for job in expensive)
    )
    worker_module.run_worker(poll_interval=poll_interval, once=True)


def parse_sse(text):
    """Split an SSE body into ``(event_name, event_id, payload)`` triples."""
    frames = []
    for block in text.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        name, ident, data = "message", None, []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("id: "):
                ident = int(line[4:])
            elif line.startswith("data: "):
                data.append(line[6:])
        if data:
            frames.append((name, ident, json.loads("\n".join(data))))
    return frames


def wait_until(predicate, timeout=90, message="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail(f"timed out after {timeout}s waiting for {message}")


def finished_selftest(client, seconds=0.2, steps=3, **params):
    """Enqueue a self-test, run it with a real worker, and return the row."""
    job = enqueue(client, "selftest", params={"seconds": seconds, "steps": steps, **params})
    run_the_queue(client.app_object.state.conn)
    row = client.get(f"/api/jobs/{job['id']}").json()
    assert row["is_terminal"], row
    return row


# ---------------------------------------------------------------------------
# POST /api/jobs
# ---------------------------------------------------------------------------


def test_a_valid_request_is_queued_and_nothing_starts(client):
    response = client.post(
        "/api/jobs", json={"type": "selftest", "params": {"seconds": 1}}
    )

    assert response.status_code == 201
    job = response.json()
    assert job["type"] == "selftest"
    assert job["status"] == "queued"
    assert job["is_terminal"] is False
    assert job["params"] == {"seconds": 1}
    assert job["base_name"] is None
    assert job["pid"] is None and job["started_at"] is None
    assert job["cost"] is False and job["gpu"] is False
    assert job["created_at_iso"]
    # Nothing runs in the web process: the worker claims it later.
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "queued"


def test_an_episode_job_is_accepted_and_canonicalised(client):
    job = enqueue(client, "postprocess", base_name="EPISODE_1")

    assert job["base_name"] == "episode_1", (
        "the index's own spelling must be stored, or the episode page - which "
        "compares exactly - cannot find the job it started"
    )
    assert job["cost"] is True


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({}, "job type is required"),
        ({"type": ""}, "job type is required"),
        ({"type": "   "}, "job type is required"),
        ({"type": 42}, "job type is required"),
        ({"type": "mine_dogecoin"}, "Unknown job type"),
        ({"type": "SELFTEST"}, "Unknown job type"),
        ({"type": "selftest", "params": []}, "params must be a JSON object"),
        ({"type": "selftest", "params": "x"}, "params must be a JSON object"),
        ({"type": "selftest", "base_name": 7}, "base_name must be a string"),
        ({"type": "postprocess"}, "needs a base_name"),
    ],
)
def test_a_bad_request_is_a_400_naming_the_problem(client, payload, expected):
    response = client.post("/api/jobs", json=payload)

    assert response.status_code == 400, response.text
    assert expected in response.json()["detail"]
    assert client.get("/api/jobs").json()["count"] == 0, "nothing may be queued"


def test_an_unknown_job_type_lists_the_known_ones(client):
    detail = client.post("/api/jobs", json={"type": "nope"}).json()["detail"]
    for job_type in jobs_module.JOB_TYPES:
        assert job_type in detail


def test_an_unknown_episode_is_a_400_not_a_404(client):
    """The request is wrong, not the URL: nothing about /api/jobs is missing."""
    response = client.post(
        "/api/jobs", json={"type": "postprocess", "base_name": "episode_99"}
    )

    assert response.status_code == 400
    assert "not in the index" in response.json()["detail"]


@pytest.mark.parametrize(
    "base_name",
    [
        "../../.env",
        r"..\..\.env",
        "../episode_1",
        r"..\episode_1",
        "podcasts/episode_1",
        r"podcasts\episode_1",
        "/etc/passwd",
        "C:\\Windows\\System32\\config\\SAM",
        "episode_1/../../secret",
        "..",
        ".",
        "episode_1\0.txt",
        "%2e%2e%2fepisode_1",
    ],
)
def test_a_traversal_in_base_name_is_refused_and_says_nothing_extra(client, base_name):
    """base_name is an index key. A path is simply a name the index does not have.

    The reply is the same for "unknown episode" and "path attempt" on purpose:
    there is nothing here worth distinguishing for a caller, and a different
    message for the second case would confirm which names exist.
    """
    response = client.post(
        "/api/jobs", json={"type": "postprocess", "base_name": base_name}
    )

    assert response.status_code == 400, response.text
    assert "not in the index" in response.json()["detail"]
    assert client.get("/api/jobs").json()["count"] == 0
    # And nothing was created on disk under any interpretation of that name.
    assert sorted(os.listdir(jobs_module.jobs_dir())) == []


@pytest.mark.parametrize("base_name", ["%", "_", "episode__", "epi%", "%%", "\\"])
def test_a_wildcard_base_name_cannot_match_an_episode(client, base_name):
    """The lookup is an equality test, not a pattern match.

    Worth pinning: ``webui.db`` does use ``LIKE`` for the *search* box, so a
    future refactor that routed this through the same query would let ``%``
    queue a paid job against whichever episode happened to come first.
    """
    response = client.post(
        "/api/jobs", json={"type": "postprocess", "base_name": base_name}
    )

    assert response.status_code == 400, response.text
    assert client.get("/api/jobs").json()["count"] == 0


def test_params_are_stored_as_arguments_never_as_a_path(client, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    job = enqueue(
        client,
        "selftest",
        params={"audio_file": str(outside), "output_dir": "C:\\Windows\\System32"},
    )

    assert job["params"]["audio_file"] == str(outside)
    assert outside.read_text(encoding="utf-8") == "secret"


def test_a_form_encoded_body_is_accepted_too(client):
    """The page has to work with scripting off, which means a plain form post."""
    response = client.post(
        "/api/jobs",
        data={"type": "selftest", "params": json.dumps({"seconds": 2})},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 201
    assert response.json()["params"] == {"seconds": 2}


@pytest.mark.parametrize("body", [b"not json", b"[1, 2, 3]", b'"scalar"', b"42"])
def test_a_body_that_is_not_a_json_object_is_a_400(client, body):
    response = client.post(
        "/api/jobs", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 400


def test_an_oversized_body_is_refused_before_it_reaches_the_database(client):
    oversized = "x" * (app_module.MAX_JOB_BODY_BYTES + 1)

    response = client.post(
        "/api/jobs", json={"type": "selftest", "params": {"note": oversized}}
    )

    assert response.status_code == 413
    assert client.get("/api/jobs").json()["count"] == 0


def test_a_cross_origin_post_cannot_queue_anything(client):
    response = client.post(
        "/api/jobs",
        json={"type": "selftest"},
        headers={"origin": "http://attacker.example"},
    )

    assert response.status_code == 403
    assert client.get("/api/jobs").json()["count"] == 0


# ---------------------------------------------------------------------------
# GET /api/jobs
# ---------------------------------------------------------------------------


def test_the_list_is_newest_first_and_reports_its_filter(client):
    created = [enqueue(client, "selftest", params={"n": n})["id"] for n in range(3)]

    payload = client.get("/api/jobs").json()

    assert payload["count"] == 3
    assert payload["status"] is None
    assert payload["limit"] == app_module.JOB_LIST_LIMIT
    assert payload["active"] is None
    assert [job["id"] for job in payload["jobs"]] == list(reversed(created))


def test_the_list_filters_by_status(client):
    queued = enqueue(client, "selftest")
    cancelled = enqueue(client, "selftest")
    client.post(f"/api/jobs/{cancelled['id']}/cancel")

    only_queued = client.get("/api/jobs", params={"status": "queued"}).json()
    only_cancelled = client.get("/api/jobs", params={"status": "cancelled"}).json()

    assert [job["id"] for job in only_queued["jobs"]] == [queued["id"]]
    assert only_queued["status"] == "queued"
    assert [job["id"] for job in only_cancelled["jobs"]] == [cancelled["id"]]


def test_an_unknown_status_filter_is_a_400(client):
    response = client.get("/api/jobs", params={"status": "exploded"})

    assert response.status_code == 400
    assert "Unknown status" in response.json()["detail"]


@pytest.mark.parametrize("limit", [0, -1, app_module.MAX_JOB_LIST_LIMIT + 1, "many"])
def test_a_bad_limit_is_refused(client, limit):
    assert client.get("/api/jobs", params={"limit": limit}).status_code == 422


def test_the_running_job_is_reported_as_active(client):
    job = enqueue(client, "selftest")
    conn = client.app_object.state.conn
    jobs_module.claim_next(conn, "test-worker")

    payload = client.get("/api/jobs").json()

    assert payload["active"]["id"] == job["id"]
    assert payload["active"]["status"] == "running"


def test_one_job_can_be_read_back_and_an_unknown_one_is_a_404(client):
    job = enqueue(client, "selftest")

    assert client.get(f"/api/jobs/{job['id']}").json()["id"] == job["id"]
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeef/events").status_code == 404
    assert client.post("/api/jobs/deadbeef/cancel").status_code == 404


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancelling_a_queued_job_settles_it_outright(client):
    job = enqueue(client, "selftest")

    response = client.post(f"/api/jobs/{job['id']}/cancel")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "cancelled"
    assert body["is_terminal"] is True
    assert body["cancel_requested"] is True


def test_cancelling_a_running_job_only_flags_it(client):
    """The status becomes cancelled once the process has actually stopped."""
    job = enqueue(client, "selftest")
    conn = client.app_object.state.conn
    jobs_module.claim_next(conn, "test-worker")

    body = client.post(f"/api/jobs/{job['id']}/cancel").json()

    assert body["status"] == "running", "nothing may report success early"
    assert body["cancel_requested"] is True
    assert body["finished_at"] is None


def test_cancel_is_a_post_only_route(client):
    job = enqueue(client, "selftest")
    assert client.get(f"/api/jobs/{job['id']}/cancel").status_code == 405
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "queued"


def test_cancelling_a_finished_job_changes_nothing(client):
    row = finished_selftest(client, seconds=0.05, steps=1)

    body = client.post(f"/api/jobs/{row['id']}/cancel").json()

    assert body["status"] == "succeeded"
    assert body["cancel_requested"] is False


# ---------------------------------------------------------------------------
# The cost flag
# ---------------------------------------------------------------------------


def test_the_catalogue_flags_every_type_that_spends_money(client):
    payload = client.get("/api/job-types").json()["job_types"]

    assert {spec["type"] for spec in payload} == set(jobs_module.JOB_TYPES)
    for spec in payload:
        expected = jobs_module.JOB_TYPES[spec["type"]]
        assert spec["cost"] == bool(expected["cost"])
        assert spec["gpu"] == bool(expected["gpu"])
        assert spec["requires_base_name"] == bool(expected["requires_base_name"])
        assert spec["label"] and spec["description"]

    paying = {spec["type"] for spec in payload if spec["cost"]}
    assert paying == {
        "fetch_latest",
        "full_episode",
        "force_episode",
        "postprocess",
        "speakers",
    }
    assert "selftest" not in paying, "the one type the tests run must be free"


def test_a_created_job_carries_the_cost_flag_everywhere_it_appears(client):
    created = enqueue(client, "postprocess", base_name="episode_1")

    assert created["cost"] is True
    assert client.get(f"/api/jobs/{created['id']}").json()["cost"] is True
    listed = client.get("/api/jobs").json()["jobs"][0]
    assert listed["cost"] is True and listed["gpu"] is False
    assert created["id"] in client.get("/jobs", headers={"accept": "text/html"}).text


def test_a_row_of_an_unknown_type_is_reported_as_expensive(client):
    """A row from a newer build must never make the UI under-warn."""
    job = enqueue(client, "selftest")
    client.app_object.state.conn.execute(
        "UPDATE jobs SET type = 'from_the_future' WHERE id = ?", (job["id"],)
    )

    body = client.get(f"/api/jobs/{job['id']}").json()

    assert body["known_type"] is False
    assert body["cost"] is True and body["gpu"] is True


# ---------------------------------------------------------------------------
# The SSE stream, over a real self-test job
# ---------------------------------------------------------------------------


def test_the_stream_replays_a_real_jobs_log_and_ends(client):
    """A finished job's events arrive in full, in order, and the stream closes.

    Real events: this job ran in a real ``python -m webui.runner`` child, so the
    sequence numbers and the message shapes are production's.
    """
    row = finished_selftest(client, seconds=0.2, steps=3)
    on_disk = [
        json.loads(line)
        for line in Path(jobs_module.events_path(row["id"]))
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(on_disk) >= 5

    started = time.monotonic()
    with client.stream("GET", f"/api/jobs/{row['id']}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert stream.headers["cache-control"].startswith("no-cache")
        assert stream.headers["x-accel-buffering"] == "no"
        body = "".join(stream.iter_text())
    assert time.monotonic() - started < 30, "the stream did not end by itself"

    frames = parse_sse(body)
    progress = [frame for frame in frames if frame[0] == "progress"]
    assert [frame[1] for frame in progress] == list(range(1, len(on_disk) + 1))
    assert [frame[2]["message"] for frame in progress] == [
        record["message"] for record in on_disk
    ]
    status = [frame for frame in frames if frame[0] == "status"]
    assert status[0][2]["id"] == row["id"]
    assert frames[-1][0] == "end", "without an end frame EventSource reconnects forever"
    assert frames[-1][2]["status"] == "succeeded"
    assert frames[-1][2]["last_seq"] == len(on_disk)


@pytest.mark.parametrize("via", ["header", "query"])
def test_a_reconnect_resumes_after_the_last_event_id(client, via):
    """``Last-Event-ID: 3`` means "everything after 3", not the whole log again."""
    row = finished_selftest(client, seconds=0.2, steps=4)
    url = f"/api/jobs/{row['id']}/events"
    with client.stream("GET", url) as stream:
        everything = parse_sse("".join(stream.iter_text()))
    seqs = [frame[1] for frame in everything if frame[0] == "progress"]
    assert len(seqs) > 3
    resume_from = seqs[2]

    if via == "header":
        kwargs = {"headers": {"last-event-id": str(resume_from)}}
    else:
        kwargs = {"params": {"last_event_id": resume_from}}
    with client.stream("GET", url, **kwargs) as stream:
        resumed = parse_sse("".join(stream.iter_text()))

    assert [frame[1] for frame in resumed if frame[0] == "progress"] == seqs[3:]
    assert resumed[-1][0] == "end"
    assert resumed[-1][2]["last_seq"] == seqs[-1]


def test_an_unparsable_last_event_id_replays_from_the_start(client):
    """Wrong in the safe direction: replay something twice, never skip it."""
    row = finished_selftest(client, seconds=0.1, steps=2)
    url = f"/api/jobs/{row['id']}/events"

    with client.stream("GET", url, headers={"last-event-id": "not a number"}) as stream:
        frames = parse_sse("".join(stream.iter_text()))

    assert [frame[1] for frame in frames if frame[0] == "progress"][0] == 1


def test_the_stream_of_a_job_with_no_log_yet_is_not_an_error(client):
    """A queued job has written nothing. That is a state, not a failure."""
    job = enqueue(client, "selftest")
    assert not os.path.exists(jobs_module.events_path(job["id"]))
    client.post(f"/api/jobs/{job['id']}/cancel")

    with client.stream("GET", f"/api/jobs/{job['id']}/events") as stream:
        frames = parse_sse("".join(stream.iter_text()))

    assert [frame[0] for frame in frames] == ["status", "end"]
    assert frames[-1][2]["status"] == "cancelled"
    assert not os.path.exists(jobs_module.events_path(job["id"])), (
        "reading a log must not create one"
    )


def test_the_stream_cannot_be_pointed_outside_the_jobs_directory(client, tmp_path):
    """A job id becomes a path in exactly one place, and that place checks it.

    The row is inserted by hand with an id no ``uuid4().hex`` could produce -
    what a hand-edited or corrupted database would hold - so the lookup finds
    it and the *stream* is what has to refuse. It does: ``events_path`` accepts
    a single safe path segment or nothing at all.
    """
    outside = tmp_path / "events.jsonl"
    outside.write_text(json.dumps({"seq": 1, "message": "secret"}) + "\n", encoding="utf-8")
    conn = client.app_object.state.conn
    conn.execute(
        "INSERT INTO jobs (id, type, params, status, cancel_requested, created_at) "
        "VALUES ('a.b', 'selftest', '{}', 'running', 0, ?)",
        (time.time(),),
    )

    with client.stream("GET", "/api/jobs/a.b/events") as stream:
        frames = parse_sse("".join(stream.iter_text()))

    assert frames[0][0] == "error"
    assert "Invalid job id" in frames[0][2]["detail"]
    assert not any(frame[0] == "progress" for frame in frames)
    assert "secret" not in json.dumps(frames)


@pytest.mark.parametrize(
    "job_id",
    ["..%2F..%2Fetc", "..", "%2e%2e", "a%00b", "x" * 65],
)
def test_a_traversal_shaped_job_id_finds_nothing(client, job_id):
    for suffix in ("", "/events"):
        response = client.get(f"/api/jobs/{job_id}{suffix}")
        assert response.status_code in (404, 405), response.text


def test_the_path_helper_itself_refuses_an_unsafe_id():
    """Stated where the guarantee lives, not only where it is used."""
    for job_id in ("../../etc/passwd", r"..\..\.env", "a/b", "a.b", ""):
        with pytest.raises(ValueError):
            jobs_module.events_path(job_id)


@pytest.mark.slow
def test_the_stream_delivers_a_real_job_as_it_runs(live_server):
    """The property that makes this endpoint worth having: live progress.

    A real uvicorn on a real socket, a real worker, and a real runner child.
    ``TestClient`` cannot show this - it runs the app to completion first - so
    this is the one place the stream is watched over HTTP while the job it
    follows is still going.
    """
    httpx = pytest.importorskip("httpx")
    base_url, app = live_server
    conn = app.state.conn

    response = httpx.post(
        f"{base_url}/api/jobs",
        json={"type": "selftest", "params": {"seconds": 1.5, "steps": 6}},
        timeout=30,
    )
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]

    box = {}

    def run():
        try:
            run_the_queue(conn)
        except BaseException as exc:  # pragma: no cover - re-raised below
            box["error"] = exc

    worker = threading.Thread(target=run, name="worker-under-test")
    worker.start()
    arrivals = []
    try:
        with httpx.Client(timeout=60) as http:
            with http.stream("GET", f"{base_url}/api/jobs/{job_id}/events") as stream:
                assert stream.status_code == 200
                for chunk in stream.iter_text():
                    arrivals.append((time.monotonic(), chunk))
                    if "event: end" in chunk:
                        break
    finally:
        worker.join(timeout=120)
    assert not worker.is_alive(), "the worker did not finish"
    assert "error" not in box, box.get("error")

    frames = parse_sse("".join(chunk for _, chunk in arrivals))
    progress = [frame for frame in frames if frame[0] == "progress"]
    assert len(progress) >= 6, "the self-test's own steps should all have arrived"
    assert [frame[1] for frame in progress] == sorted(frame[1] for frame in progress)
    assert frames[-1][0] == "end"
    assert frames[-1][2]["status"] == "succeeded"

    running = [
        frame for frame in frames if frame[0] == "status" and frame[2]["status"] == "running"
    ]
    assert running, "the stream should have shown the job while it was still running"
    span = arrivals[-1][0] - arrivals[0][0]
    assert span > 0.3, (
        f"every chunk arrived within {span:.3f}s: this looks buffered, not streamed"
    )

    row = httpx.get(f"{base_url}/api/jobs/{job_id}", timeout=30).json()
    assert row["status"] == "succeeded"
    assert row["exit_code"] == 0


# ---------------------------------------------------------------------------
# Nothing leaks
# ---------------------------------------------------------------------------


def test_no_route_returns_an_environment_variable_or_a_secret(client, monkeypatch):
    """Not the key, not the token, not the name of the variable holding them."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-CANARY-DO-NOT-LEAK-0123456789")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_CANARY-DO-NOT-LEAK-0123456789")
    monkeypatch.setenv("WHYCAST_CANARY", "CANARY-DO-NOT-LEAK-env")
    # The self-test runs first and the paid row is created after it: the cost
    # guard in run_the_queue would (rightly) refuse to start a worker while a
    # postprocess job sits in the queue.
    free = finished_selftest(client, seconds=0.05, steps=1)
    paid = enqueue(client, "postprocess", base_name="episode_1")

    urls = [
        "/",
        "/jobs",
        f"/jobs/{paid['id']}",
        f"/jobs/{free['id']}",
        "/episodes/episode_1",
        "/unmatched",
        "/api/health",
        "/api/episodes",
        "/api/episodes/episode_1",
        "/api/job-types",
        "/api/jobs",
        f"/api/jobs/{paid['id']}",
        f"/api/jobs/{free['id']}",
        f"/api/jobs/{free['id']}/events",
    ]
    for url in urls:
        response = client.get(url, headers={"accept": "text/html"})
        assert response.status_code == 200, f"{url} -> {response.status_code}"
        haystack = response.text + json.dumps(dict(response.headers))
        assert "CANARY" not in haystack, f"{url} leaked a secret"
        assert "sk-" not in haystack, f"{url} leaked something key-shaped"
        for name in ("OPENAI_API_KEY", "HUGGINGFACE_TOKEN", "WHYCAST_CANARY"):
            assert name not in haystack, f"{url} named {name}"


def test_a_failed_job_reports_its_reason_without_a_stack_trace(client):
    """The error column is a sentence for a human; the stack is in the log."""
    row = finished_selftest(client, seconds=0.05, steps=1, fail=True)

    assert row["status"] == "failed"
    assert row["error"]
    assert "Traceback" not in row["error"]
    assert "\n" not in row["error"]

    with client.stream("GET", f"/api/jobs/{row['id']}/events") as stream:
        frames = parse_sse("".join(stream.iter_text()))
    assert any(
        "Traceback" in json.dumps(frame[2].get("data", {})) for frame in frames
    ), "the stack must still be reachable, in the event log"


def test_the_queue_routes_report_an_unusable_queue_as_503(client):
    """An honest 503 beats a raw sqlite error rendered as a 500.

    ``ensure_job_schema`` only logs when it fails at startup, so a database
    without the queue tables is a state the routes really can be asked to serve.
    Dropping the table produces it exactly.
    """
    job = enqueue(client, "selftest")
    client.app_object.state.conn.execute("DROP TABLE jobs")

    assert client.get("/api/jobs").status_code == 503
    assert client.get(f"/api/jobs/{job['id']}").status_code == 503
    assert client.post("/api/jobs", json={"type": "selftest"}).status_code == 503
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 503
    assert client.get("/jobs", headers={"accept": "text/html"}).status_code == 503
    # The read-only half of the app has nothing to do with the queue and must
    # keep working: an operator whose queue broke still wants the episode list.
    assert client.get("/api/episodes").status_code == 200
