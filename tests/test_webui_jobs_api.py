"""
Tests for the WHYcast web UI job API and SSE stream (ADR-008 / TASK-003).

These cover the HTTP layer only: the routes in :mod:`webui.app` that enqueue a
job, list and cancel jobs, and stream a job's progress. The queue itself
(:mod:`webui.jobs`), the child process (:mod:`webui.runner`) and the worker
(:mod:`webui.worker`) have their own tests.

**No job is ever run here, and no paid call is ever made.** Nothing in this
file starts a worker; a "running" job is a row this test wrote, and a job's
event log is a file this test appended to. That is on purpose: the endpoints
under test read a database and tail a file, so faking the writer tests exactly
what the endpoints do and nothing else. The end-to-end path - real worker, real
child process, real ``selftest`` job - is exercised separately.

As in ``test_webui_api.py``, everything runs against a synthetic podcast
directory and a temporary SQLite database. The operator's real ``podcasts/``
and real ``webui/whycast_webui.db`` are never touched.
"""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webui import app as app_module  # noqa: E402
from webui import jobs as jobs_module  # noqa: E402

#: The synthetic podcast directory. One episode with audio and a transcript is
#: enough: ``base_name`` resolution is a lookup, and what is being tested is
#: that a name the index does not know is refused.
FILES = {
    "episode_1.mp3": b"fake-mp3-bytes",
    "episode_1.txt": b"transcript of episode one\n",
    "episode_1_summary.txt": b"summary of episode one\n",
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
def client(tmp_path, podcast_dir, monkeypatch):
    """A TestClient over a throwaway index, queue and job-log directory.

    ``WHYCAST_JOBS_DIR`` is redirected before the app is built:
    :func:`webui.jobs.events_path` reads it every call, so a test that forgot
    this would tail the operator's real ``logs/jobs``.

    ``TestClient`` as a context manager, always: the index connection is opened
    in the lifespan handler, and the job schema is created there too.
    """
    jobs_home = tmp_path / "jobs"
    jobs_home.mkdir()
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(jobs_home))
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcast_dir))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index.db"))
    # The host guard is read once, in create_app. Clear both variables rather
    # than inheriting whatever the operator - or an earlier test file - left in
    # the environment: an off-loopback bind disables the Host allowlist, which
    # would make the guard tests pass or fail depending on file order.
    monkeypatch.delenv("WHYCAST_WEBUI_HOST", raising=False)
    monkeypatch.delenv("WHYCAST_WEBUI_ALLOWED_HOSTS", raising=False)

    app = app_module.create_app(
        podcast_dir=str(podcast_dir), db_path=str(tmp_path / "index.db")
    )
    with TestClient(app) as test_client:
        test_client.jobs_home = jobs_home  # type: ignore[attr-defined]
        test_client.app_object = app  # type: ignore[attr-defined]
        yield test_client


@pytest.fixture
def live_server(tmp_path, podcast_dir, monkeypatch):
    """A real uvicorn on a real loopback socket, in a background thread.

    Needed for exactly one thing: proving the SSE endpoint streams. Both
    in-process transports (``TestClient`` and ``httpx.ASGITransport``) run the
    ASGI app to completion before handing back a response, so neither can tell
    a live stream from a buffered one - see :func:`test_stream_is_not_buffered`.

    Yields ``(base_url, app)``. The app object comes back too so a test can
    reach ``app.state.conn`` and play the part of the worker.
    """
    uvicorn = pytest.importorskip("uvicorn", reason="the live-stream test needs uvicorn")

    jobs_home = tmp_path / "jobs"
    jobs_home.mkdir()
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(jobs_home))
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcast_dir))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index.db"))
    # The host guard is read once, in create_app. Clear both variables rather
    # than inheriting whatever the operator - or an earlier test file - left in
    # the environment: an off-loopback bind disables the Host allowlist, which
    # would make the guard tests pass or fail depending on file order.
    monkeypatch.delenv("WHYCAST_WEBUI_HOST", raising=False)
    monkeypatch.delenv("WHYCAST_WEBUI_ALLOWED_HOSTS", raising=False)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    app = app_module.create_app(
        podcast_dir=str(podcast_dir), db_path=str(tmp_path / "index.db")
    )
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
        thread.join(timeout=15)


def enqueue(client, job_type="selftest", **kwargs):
    """Create a job through the API and return the row, failing loudly if not 201."""
    response = client.post("/api/jobs", json={"type": job_type, **kwargs})
    assert response.status_code == 201, response.text
    return response.json()


# ---------------------------------------------------------------------------
# The job catalogue: where the cost warning comes from
# ---------------------------------------------------------------------------


def test_job_types_report_gpu_and_cost(client):
    """Every type says whether it costs money and whether it claims the GPU."""
    payload = client.get("/api/job-types").json()["job_types"]
    by_type = {entry["type"]: entry for entry in payload}

    assert set(by_type) == set(jobs_module.JOB_TYPES)
    for entry in payload:
        assert isinstance(entry["cost"], bool)
        assert isinstance(entry["gpu"], bool)
        assert entry["label"] and entry["description"]

    # The two that matter to an operator about to click something.
    assert by_type["selftest"]["cost"] is False
    assert by_type["selftest"]["gpu"] is False
    assert by_type["postprocess"]["cost"] is True
    assert by_type["force_episode"]["cost"] is True
    assert by_type["force_episode"]["gpu"] is True


def test_created_job_carries_the_cost_flag(client):
    """A job response marks cost, so the UI can warn without a second lookup."""
    assert enqueue(client, "selftest")["cost"] is False
    paid = enqueue(client, "postprocess", base_name="episode_1")
    assert paid["cost"] is True
    assert client.get(f"/api/jobs/{paid['id']}").json()["cost"] is True
    assert any(job["cost"] for job in client.get("/api/jobs").json()["jobs"])


# ---------------------------------------------------------------------------
# What a request may say
# ---------------------------------------------------------------------------


def test_unknown_job_type_is_rejected(client):
    """The dispatch table is the allowlist: an unknown type never reaches the queue."""
    response = client.post("/api/jobs", json={"type": "rm_minus_rf"})
    assert response.status_code == 400
    assert "rm_minus_rf" in response.json()["detail"]
    assert client.get("/api/jobs").json()["count"] == 0


@pytest.mark.parametrize(
    "base_name",
    [
        "not_an_episode",
        "../../../etc/passwd",
        "..\\..\\config",
        "episode_1/../../secret",
        "",
    ],
)
def test_base_name_must_resolve_through_the_index(client, base_name):
    """``base_name`` is a lookup key, never a path.

    A name the index does not know is a 400 - including every shape of
    traversal, which is refused for the ordinary reason that no episode is
    called that, not by a special case that could be forgotten.
    """
    response = client.post(
        "/api/jobs", json={"type": "postprocess", "base_name": base_name}
    )
    assert response.status_code == 400
    assert client.get("/api/jobs").json()["count"] == 0


def test_episode_scoped_type_needs_an_episode(client):
    """``postprocess`` without a base_name is a 400, not a job that fails later."""
    response = client.post("/api/jobs", json={"type": "postprocess"})
    assert response.status_code == 400
    assert "base_name" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"type": ""},
        {"type": 42},
        {"type": "selftest", "params": [1, 2, 3]},
        {"type": "selftest", "params": "seconds=1"},
        {"type": "selftest", "base_name": 7},
    ],
)
def test_malformed_bodies_are_400(client, body):
    assert client.post("/api/jobs", json=body).status_code == 400


def test_non_json_body_is_400_not_500(client):
    assert client.post("/api/jobs", content=b"<html>nope</html>").status_code == 400


def test_form_encoded_body_is_accepted(client):
    """A plain HTML form must work: it is the no-JavaScript path.

    ``params`` arrives as a string there and is parsed as JSON, because a form
    field cannot carry a nested object.
    """
    response = client.post(
        "/api/jobs",
        data={"type": "selftest", "params": '{"seconds": 1}'},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["params"] == {"seconds": 1}


def test_valid_request_is_queued_not_started(client):
    """Enqueueing writes a row and returns. The web layer runs nothing."""
    job = enqueue(client, "postprocess", base_name="episode_1")
    assert job["status"] == "queued"
    assert job["base_name"] == "episode_1"
    assert job["pid"] is None
    assert job["started_at"] is None


# ---------------------------------------------------------------------------
# Listing, fetching, cancelling
# ---------------------------------------------------------------------------


def test_list_filters_by_status_and_rejects_unknown_ones(client):
    first = enqueue(client, "selftest")
    enqueue(client, "selftest")
    client.post(f"/api/jobs/{first['id']}/cancel")

    assert client.get("/api/jobs").json()["count"] == 2
    assert client.get("/api/jobs?status=queued").json()["count"] == 1
    assert client.get("/api/jobs?status=cancelled").json()["count"] == 1
    assert client.get("/api/jobs?status=running").json()["count"] == 0
    assert client.get("/api/jobs?status=nonsense").status_code == 400


def test_unknown_job_id_is_404_everywhere(client):
    assert client.get("/api/jobs/nosuchjob").status_code == 404
    assert client.post("/api/jobs/nosuchjob/cancel").status_code == 404
    assert client.get("/api/jobs/nosuchjob/events").status_code == 404
    assert client.get("/jobs/nosuchjob", headers={"accept": "text/html"}).status_code == 404


def test_cancelling_a_queued_job_cancels_it_outright(client):
    """Nothing has started, so there is nothing to kill."""
    job = enqueue(client, "selftest")
    cancelled = client.post(f"/api/jobs/{job['id']}/cancel").json()
    assert cancelled["status"] == "cancelled"
    assert cancelled["is_terminal"] is True


def test_cancelling_a_running_job_only_flags_it(client):
    """A running job is cancelled when it stops, not when the button is pressed.

    Reporting ``cancelled`` here would be a lie the UI would repeat: the child
    process is still on the GPU until the worker's kill lands.
    """
    job = enqueue(client, "selftest")
    conn = client.app_object.state.conn
    jobs_module.mark_running(conn, job["id"], pid=os.getpid())

    response = client.post(f"/api/jobs/{job['id']}/cancel").json()
    assert response["status"] == "running"
    assert response["cancel_requested"] is True
    assert response["is_terminal"] is False


def test_active_job_is_reported(client):
    job = enqueue(client, "selftest")
    assert client.get("/api/jobs").json()["active"] is None
    jobs_module.mark_running(client.app_object.state.conn, job["id"], pid=os.getpid())
    assert client.get("/api/jobs").json()["active"]["id"] == job["id"]


# ---------------------------------------------------------------------------
# The SSE stream
# ---------------------------------------------------------------------------


def write_events(path: Path, records, delay=0.0):
    """Append JSONL event lines the way :class:`webui.runner.JsonlEventSink` does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if delay:
                time.sleep(delay)


def event_record(seq, message, **extra):
    record = {
        "seq": seq,
        "ts": time.time(),
        "step": "selftest",
        "message": message,
        "level": "info",
        "progress": None,
        "data": {},
    }
    record.update(extra)
    return record


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


def test_stream_of_a_terminal_job_replays_and_ends(client):
    """A finished job's log arrives in full and the stream closes by itself.

    The closing matters as much as the content: ``EventSource`` reconnects on
    any clean close, so a stream that just stopped would have the browser
    reopening it forever. The ``end`` frame is what the client closes on.
    """
    job = enqueue(client, "selftest")
    write_events(
        Path(jobs_module.events_path(job["id"])),
        [event_record(i, f"line {i}") for i in range(1, 6)],
    )
    client.post(f"/api/jobs/{job['id']}/cancel")  # queued -> cancelled, terminal

    started = time.monotonic()
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert stream.headers["cache-control"].startswith("no-cache")
        assert stream.headers["x-accel-buffering"] == "no"
        body = "".join(stream.iter_text())
    assert time.monotonic() - started < 10, "the stream did not end on its own"

    frames = parse_sse(body)
    progress = [f for f in frames if f[0] == "progress"]
    assert [f[1] for f in progress] == [1, 2, 3, 4, 5]
    assert [f[2]["message"] for f in progress] == [f"line {i}" for i in range(1, 6)]
    assert frames[-1][0] == "end"
    assert frames[-1][2]["status"] == "cancelled"
    assert frames[-1][2]["last_seq"] == 5


def test_stream_of_a_queued_job_survives_a_missing_log(client):
    """A queued job has written no events.jsonl yet. That is not an error."""
    job = enqueue(client, "selftest")
    assert not os.path.exists(jobs_module.events_path(job["id"]))
    client.post(f"/api/jobs/{job['id']}/cancel")

    with client.stream("GET", f"/api/jobs/{job['id']}/events") as stream:
        frames = parse_sse("".join(stream.iter_text()))
    assert [f[0] for f in frames] == ["status", "end"]
    assert frames[-1][2]["last_seq"] == 0


def test_stream_tails_a_job_while_it_is_written(client):
    """A job written while the stream is open is followed to the end.

    A writer thread stands in for the runner child: it appends flushed lines
    with a gap between them, then marks the job finished. What this proves is
    *content*: every line written after the stream opened arrives, in order,
    and the stream ends on the status change rather than on end-of-file.

    It cannot prove the stream is live. ``TestClient`` runs the app to
    completion before it returns a response (so does ``httpx.ASGITransport`` -
    both end with ``assert response_complete.is_set()``), so an endpoint that
    buffered everything until the last byte would pass this test unchanged.
    :func:`test_stream_is_not_buffered` runs a real server for that.
    """
    job = enqueue(client, "selftest")
    conn = client.app_object.state.conn
    jobs_module.mark_running(conn, job["id"], pid=os.getpid())
    path = Path(jobs_module.events_path(job["id"]))

    def write():
        time.sleep(0.2)
        write_events(path, [event_record(i, f"line {i}") for i in range(1, 6)], delay=0.1)
        jobs_module.finish(conn, job["id"], "succeeded", exit_code=0)

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    try:
        with client.stream("GET", f"/api/jobs/{job['id']}/events") as stream:
            body = "".join(stream.iter_text())
    finally:
        writer.join(timeout=10)

    frames = parse_sse(body)
    assert [f[1] for f in frames if f[0] == "progress"] == [1, 2, 3, 4, 5]
    assert frames[-1][0] == "end"
    assert frames[-1][2]["status"] == "succeeded"
    assert frames[-1][2]["last_seq"] == 5


def test_stream_is_not_buffered(live_server):
    """Events reach a real HTTP client as they are written.

    The one property the in-process tests cannot check, and the one that makes
    this endpoint worth having: a 40-minute transcription must show its
    progress while it runs, not arrive as a single burst at the end.

    So: a real uvicorn, a real socket, real ``httpx`` streaming, and arrival
    times recorded per chunk. Five lines 0.6 s apart span about 2.4 s; a
    buffered response delivers all five at the same instant and fails here.

    The spacing is deliberately four times the margin asserted below. At 0.3 s
    this test failed intermittently on a loaded machine: the writer thread
    stalls, several events land in one server-side poll, and they then arrive
    together - which looks exactly like buffering without being it. The wide
    gap keeps the real property (progress arrives while the job runs) provable
    without the clock deciding the verdict.
    """
    base_url, app = live_server
    created = httpx.post(
        f"{base_url}/api/jobs", json={"type": "selftest"}, timeout=10.0
    )
    assert created.status_code == 201, created.text
    job = created.json()

    conn = app.state.conn
    jobs_module.mark_running(conn, job["id"], pid=os.getpid())
    path = Path(jobs_module.events_path(job["id"]))

    def write():
        time.sleep(0.3)
        write_events(path, [event_record(i, f"line {i}") for i in range(1, 6)], delay=0.6)
        jobs_module.finish(conn, job["id"], "succeeded", exit_code=0)

    writer = threading.Thread(target=write, daemon=True)
    writer.start()
    try:
        start = time.monotonic()
        arrivals = []
        with httpx.stream(
            "GET", f"{base_url}/api/jobs/{job['id']}/events", timeout=60.0
        ) as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            for chunk in response.iter_text():
                arrivals.append((time.monotonic() - start, chunk))
    finally:
        writer.join(timeout=15)

    frames = parse_sse("".join(chunk for _, chunk in arrivals))
    assert [f[1] for f in frames if f[0] == "progress"] == [1, 2, 3, 4, 5]
    assert frames[-1][0] == "end"

    first = next(t for t, chunk in arrivals if '"line 1"' in chunk)
    last = next(t for t, chunk in arrivals if '"line 5"' in chunk)
    assert last - first > 0.5, (
        f"the stream looks buffered: line 1 arrived at {first:.2f}s, "
        f"line 5 at {last:.2f}s"
    )


def test_heartbeat_keeps_an_idle_stream_open(client, monkeypatch):
    """A stream with nothing to say still says something.

    A queued job has no log file at all, and a transcription step can run for
    many minutes in silence. Without traffic an intermediary or an idle browser
    drops the connection and the log appears to have frozen, so the tailer
    sends a ``: heartbeat`` comment - valid SSE, ignored by ``EventSource``.

    The interval is monkeypatched down from 15 s: it is read from the module
    global on every loop, so the test does not have to wait out the real one.
    """
    monkeypatch.setattr(app_module, "SSE_HEARTBEAT_SECONDS", 0.2)
    job = enqueue(client, "selftest")
    assert not os.path.exists(jobs_module.events_path(job["id"]))
    conn = client.app_object.state.conn

    def stop_it():
        time.sleep(1.5)
        # queued -> cancelled, which is terminal, so the stream ends by itself.
        jobs_module.request_cancel(conn, job["id"])

    stopper = threading.Thread(target=stop_it, daemon=True)
    stopper.start()
    try:
        with client.stream("GET", f"/api/jobs/{job['id']}/events") as stream:
            body = "".join(stream.iter_text())
    finally:
        stopper.join(timeout=10)

    assert body.count(": heartbeat") >= 3, f"no heartbeats in:\n{body}"
    assert "event: end" in body
    frames = parse_sse(body)
    assert frames[-1][0] == "end"
    assert frames[-1][2]["status"] == "cancelled"


@pytest.mark.parametrize("via", ["header", "query"])
def test_reconnect_resumes_after_last_event_id(client, via):
    """A reconnect gets what it has not seen, not the whole log again."""
    job = enqueue(client, "selftest")
    write_events(
        Path(jobs_module.events_path(job["id"])),
        [event_record(i, f"line {i}") for i in range(1, 9)],
    )
    client.post(f"/api/jobs/{job['id']}/cancel")

    url = f"/api/jobs/{job['id']}/events"
    if via == "header":
        response = client.get(url, headers={"Last-Event-ID": "5"})
    else:
        response = client.get(f"{url}?last_event_id=5")

    seqs = [f[1] for f in parse_sse(response.text) if f[0] == "progress"]
    assert seqs == [6, 7, 8], f"expected 6..8, got {seqs}"


def test_unparsable_last_event_id_replays_from_the_start(client):
    """Wrong in the safe direction: replay rather than silently skip events."""
    job = enqueue(client, "selftest")
    write_events(
        Path(jobs_module.events_path(job["id"])),
        [event_record(i, f"line {i}") for i in range(1, 4)],
    )
    client.post(f"/api/jobs/{job['id']}/cancel")

    response = client.get(
        f"/api/jobs/{job['id']}/events", headers={"Last-Event-ID": "not-a-number"}
    )
    assert [f[1] for f in parse_sse(response.text) if f[0] == "progress"] == [1, 2, 3]


def test_a_half_written_line_is_not_delivered(client):
    """A line without its newline is still being written; it must not be parsed.

    The writer flushes per line, but a reader can land between the write and
    the newline. Half a JSON object would be a dropped event, which is worse
    than waiting one poll for the rest.
    """
    job = enqueue(client, "selftest")
    path = Path(jobs_module.events_path(job["id"]))
    write_events(path, [event_record(1, "complete line")])
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"seq": 2, "message": "torn')  # no newline
        handle.flush()
    client.post(f"/api/jobs/{job['id']}/cancel")

    response = client.get(f"/api/jobs/{job['id']}/events")
    frames = parse_sse(response.text)
    seqs = [f[1] for f in frames if f[0] == "progress"]
    assert seqs == [1]
    assert "torn" not in response.text


def test_malformed_lines_are_skipped_not_fatal(client):
    """Junk in the log costs that line, not the rest of the stream."""
    job = enqueue(client, "selftest")
    path = Path(jobs_module.events_path(job["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event_record(1, "first")) + "\n")
        handle.write("this is not json at all\n")
        handle.write(json.dumps(["a list, not an object"]) + "\n")
        handle.write(json.dumps({"no_seq": True}) + "\n")
        handle.write(json.dumps(event_record(2, "second")) + "\n")
    client.post(f"/api/jobs/{job['id']}/cancel")

    frames = parse_sse(client.get(f"/api/jobs/{job['id']}/events").text)
    assert [f[1] for f in frames if f[0] == "progress"] == [1, 2]
    assert frames[-1][0] == "end"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def test_dashboard_and_detail_pages_render(client):
    job = enqueue(client, "postprocess", base_name="episode_1")

    dashboard = client.get("/jobs", headers={"accept": "text/html"})
    assert dashboard.status_code == 200
    assert "text/html" in dashboard.headers["content-type"]
    assert job["id"][:8] in dashboard.text

    detail = client.get(f"/jobs/{job['id']}", headers={"accept": "text/html"})
    assert detail.status_code == 200
    # The live-log hook /static/app.js binds to. Renamed here without renaming
    # it there, the log silently stops arriving.
    assert f'data-job-id="{job["id"]}"' in detail.text
    assert "data-job-log" in detail.text


def test_pages_keep_the_strict_csp(client):
    """No inline script anywhere: the app's CSP forbids it (see _PAGE_HEADERS)."""
    for url in ("/jobs", f"/jobs/{enqueue(client, 'selftest')['id']}"):
        response = client.get(url, headers={"accept": "text/html"})
        csp = response.headers["content-security-policy"]
        assert "script-src 'self'" in csp
        assert "'unsafe-inline'" not in csp.split("style-src")[0]
        assert "<script>" not in response.text


def test_episode_page_offers_the_enqueue_actions(client):
    """The episode page gets the action catalogue, cheapest first."""
    response = client.get("/episodes/episode_1", headers={"accept": "text/html"})
    assert response.status_code == 200
    assert 'action="/api/jobs"' in response.text
    for job_type in app_module.EPISODE_ACTION_TYPES:
        assert job_type in response.text
    # force_episode is the expensive one and must come last.
    assert app_module.EPISODE_ACTION_TYPES[-1] == "force_episode"


def test_fallback_pages_carry_the_same_hooks(client):
    """A missing template must not take the job pages down.

    These are what ``_render_or_fallback`` serves when a template is not
    installed, so they have to carry the attributes ``/static/app.js`` binds
    to, and no inline script.
    """
    job = enqueue(client, "selftest")
    detail = app_module._job_detail_fallback(job).body.decode()
    for hook in (
        'class="joblog"',
        f'data-job-id="{job["id"]}"',
        "data-job-log",
        "data-events-url=",
        "data-terminal=",
        "data-job-status-badge",
    ):
        assert hook in detail, hook
    assert "<script>" not in detail

    dashboard = app_module._jobs_fallback([job], None).body.decode()
    assert job["id"][:8] in dashboard
    assert "<script>" not in dashboard


# ---------------------------------------------------------------------------
# Nothing leaks, and the phase 1 guards still hold
# ---------------------------------------------------------------------------


def test_no_route_returns_a_secret(client, monkeypatch):
    """No job response may carry a credential, masked or otherwise (ADR-008)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-DO-NOT-LEAK-THIS-VALUE")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_DO-NOT-LEAK-THIS-EITHER")
    job = enqueue(client, "postprocess", base_name="episode_1")

    bodies = [
        client.get("/api/job-types").text,
        client.get("/api/jobs").text,
        client.get(f"/api/jobs/{job['id']}").text,
        client.get("/jobs", headers={"accept": "text/html"}).text,
        client.get(f"/jobs/{job['id']}", headers={"accept": "text/html"}).text,
    ]
    for body in bodies:
        assert "DO-NOT-LEAK" not in body
        assert "sk-" not in body
        assert "OPENAI_API_KEY" not in body


def test_params_are_never_used_to_build_a_path(client, tmp_path):
    """``params`` is arguments for the runner, never a filename.

    A job whose params name a path outside the podcast directory is still just
    a queued row with those params in it; nothing in the web layer opens them.
    """
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    job = enqueue(
        client,
        "selftest",
        params={"audio_file": str(outside), "output_dir": "C:\\Windows\\System32"},
    )
    assert job["params"]["audio_file"] == str(outside)
    assert job["status"] == "queued"
    assert outside.read_text(encoding="utf-8") == "secret"


def test_cross_origin_post_is_still_refused(client):
    """The phase 1 CSRF guard covers the new write routes too."""
    response = client.post(
        "/api/jobs",
        json={"type": "selftest"},
        headers={"origin": "http://attacker.example"},
    )
    assert response.status_code == 403
    assert client.get("/api/jobs").json()["count"] == 0


def test_bad_host_header_is_still_refused(client):
    """DNS rebinding cannot reach the queue either."""
    assert client.get("/api/jobs", headers={"host": "attacker.example"}).status_code == 421


# ---------------------------------------------------------------------------
# Phase 2 review findings (minor)
# ---------------------------------------------------------------------------


def test_base_name_is_stored_as_the_index_spells_it(client):
    """A job for ``EPISODE_1`` must show up on the page for ``episode_1``.

    Index lookups are case-insensitive, so the mixed-case name was accepted and
    ran correctly - and was then invisible in the episode page's job history,
    which filters with an exact string compare. Storing the canonical key is
    what makes "this job belongs to this episode" one answer instead of two.
    """
    response = client.post(
        "/api/jobs", json={"type": "postprocess", "base_name": "EPISODE_1"}
    )
    assert response.status_code == 201
    assert response.json()["base_name"] == "episode_1"

    page = client.get("/episodes/episode_1")
    assert page.status_code == 200
    assert response.json()["id"] in page.text


def test_an_oversized_job_request_is_refused(client):
    """The body is buffered whole and then written into the queue database,

    which shares its SQLite file with the episode index. A 32 MB parameter
    string grew that file to 41 MB. Nothing legitimate comes near the cap.
    """
    oversized = "x" * (app_module.MAX_JOB_BODY_BYTES + 1)
    response = client.post(
        "/api/jobs", json={"type": "selftest", "params": {"note": oversized}}
    )
    assert response.status_code == 413
    assert client.get("/api/jobs").json()["count"] == 0

    ok = client.post("/api/jobs", json={"type": "selftest", "params": {"note": "x"}})
    assert ok.status_code == 201
