"""
Episode download tests for ADR-009: a partial download is worth nothing.

Everything here runs against a ``http.server`` bound to 127.0.0.1 on an
ephemeral port. No internet, no real feed, no mocking of ``requests`` - the
bytes really travel over a socket, because the failures being pinned down are
framing failures and a mock has no framing.

The three ways a download can be worthless while still looking like an HTTP
success, and the witness that catches each:

* half a body under HTTP/1.1 with a ``Content-Length``  -> the client itself
  notices the short read (this already worked; the test pins that nothing
  lands at the target path);
* half a body with *close-delimited* framing - HTTP/1.0, or ``Connection:
  close`` with no ``Content-Length`` - which ends exactly like a complete one
  -> only the RSS enclosure's ``length`` can tell;
* a ``200 text/html`` error page from a CDN, or an empty body, both perfectly
  self-consistent -> the enclosure length and the zero-byte rule.

The feed itself is handed to feedparser as a string; only the audio is served.
That keeps the RSS out of the directory that gets scanned, so a phantom-episode
assertion cannot be muddied by the feed file itself.
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from whycast.episodes import scan_podcasts
from whycast.io_utils import TEMP_SUFFIX
from whycast.pipeline import feed as feed_module
from whycast.pipeline.feed import (
    _download_audio,
    _enclosure_length,
    _header_length,
    _verify_download,
    delete_episode_files,
    download_all_episodes_from_rssfeed,
    podcast_fetching_workflow,
)

# Deterministic and mp3-shaped enough to be recognisable in a hexdump; 40 KB is
# five 8 KB chunks, so a "half of it" truncation lands mid-chunk.
PAYLOAD = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(range(256)) * 160
HTML_ERROR = b"<html><body>503 Backend unavailable</body></html>"


# ---------------------------------------------------------------------------
# A local server with per-test routes
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 by default so a route can choose its framing deliberately; the
    # close-delimited routes downgrade themselves by omitting Content-Length.
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        route = self.server.routes.get(self.path)
        self.server.hits.append(self.path)
        if route is None:
            self.send_error(404)
            return
        try:
            route(self)
        except OSError:
            # The client hung up first (a timeout test); not this test's news.
            self.close_connection = True

    def log_message(self, *args):
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        pass  # a deliberately aborted request is not a test failure


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.routes = {}
    srv.hits = []
    srv.released = threading.Event()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.released.set()
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def url_for(server, path):
    host, port = server.server_address[:2]
    return f"http://{host}:{port}{path}"


# --- route builders --------------------------------------------------------

def whole_body(payload, content_type="audio/mpeg"):
    """A correct, framed, complete response."""
    def route(h):
        h.send_response(200)
        h.send_header("Content-Type", content_type)
        h.send_header("Content-Length", str(len(payload)))
        h.end_headers()
        h.wfile.write(payload)
    return route


def framed_but_short(payload):
    """Promises the full length, sends half, drops the connection."""
    def route(h):
        h.send_response(200)
        h.send_header("Content-Type", "audio/mpeg")
        h.send_header("Content-Length", str(len(payload)))
        h.end_headers()
        h.wfile.write(payload[: len(payload) // 2])
        h.wfile.flush()
        h.close_connection = True
    return route


def close_delimited_short(payload):
    """No Content-Length at all: the body ends when the connection does.

    This is the framing that hides a truncation. The client cannot tell "the
    server finished" from "the server gave up" - both are just EOF.
    """
    def route(h):
        h.send_response(200)
        h.send_header("Content-Type", "audio/mpeg")
        h.send_header("Connection", "close")
        h.end_headers()
        h.wfile.write(payload[: len(payload) // 2])
        h.wfile.flush()
        h.close_connection = True
    return route


def stalls_after(payload, released):
    """Sends half, then goes quiet until the fixture releases it."""
    def route(h):
        h.send_response(200)
        h.send_header("Content-Type", "audio/mpeg")
        h.send_header("Content-Length", str(len(payload)))
        h.end_headers()
        h.wfile.write(payload[: len(payload) // 2])
        h.wfile.flush()
        released.wait(timeout=30)
        h.close_connection = True
    return route


def rss(title, audio_url, length=None):
    """A minimal feed with one enclosure, as feedparser hands it to feed.py."""
    declared = "" if length is None else f' length="{length}"'
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0"><channel><title>WHYcast</title>'
        f"<item><title>{title}</title>"
        f'<enclosure url="{audio_url}" type="audio/mpeg"{declared}/>'
        "</item></channel></rss>"
    )


def leftovers(directory):
    return sorted(n for n in os.listdir(directory) if n.endswith(TEMP_SUFFIX))


def episode_bases(directory):
    return [e.base_name for e in scan_podcasts(str(directory)).episodes]


# ---------------------------------------------------------------------------
# The happy path: what "complete" has to mean
# ---------------------------------------------------------------------------

def test_a_completed_download_is_byte_identical(tmp_path, server):
    server.routes["/episode_42.mp3"] = whole_body(PAYLOAD)
    feed_xml = rss("episode_42", url_for(server, "/episode_42.mp3"), len(PAYLOAD))

    result = podcast_fetching_workflow(feed_xml, str(tmp_path), return_base_name=True)

    target = tmp_path / "episode_42.mp3"
    assert result == (str(target), "episode_42")
    assert target.read_bytes() == PAYLOAD, "the file on disk must be the file sent"
    assert leftovers(tmp_path) == []
    assert episode_bases(tmp_path) == ["episode_42"]


def test_a_completed_download_survives_a_generous_feed_length(tmp_path, server):
    """A feed that understates the size is not a reason to throw the file away.

    Hosts re-encode and inject after publishing the feed, which makes the file
    *longer* than declared. Only shorter is a truncation.
    """
    server.routes["/episode_42.mp3"] = whole_body(PAYLOAD)
    feed_xml = rss("episode_42", url_for(server, "/episode_42.mp3"), len(PAYLOAD) - 1000)

    result = podcast_fetching_workflow(feed_xml, str(tmp_path), return_base_name=True)

    assert result == (str(tmp_path / "episode_42.mp3"), "episode_42")
    assert (tmp_path / "episode_42.mp3").read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# Aborted and truncated: nothing may land, nothing may be indexed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, route, declared",
    [
        # The client notices this one itself: the promised length never arrives.
        ("framed", framed_but_short(PAYLOAD), len(PAYLOAD)),
        # Nothing but the feed's own length can notice this one.
        ("close_delimited", close_delimited_short(PAYLOAD), len(PAYLOAD)),
        # 200 OK, honest Content-Length, and 49 bytes of CDN apology.
        ("html_error_page", whole_body(HTML_ERROR, "text/html"), len(PAYLOAD)),
        # Self-consistent and completely empty.
        ("empty", whole_body(b""), len(PAYLOAD)),
    ],
)
def test_a_broken_download_leaves_no_file_and_no_episode(
    tmp_path, server, name, route, declared
):
    server.routes[f"/{name}.mp3"] = route
    feed_xml = rss(name, url_for(server, f"/{name}.mp3"), declared)

    result = podcast_fetching_workflow(feed_xml, str(tmp_path), return_base_name=True)

    assert result == (None, None), "a partial download must report failure"
    assert not (tmp_path / f"{name}.mp3").exists(), "nothing may appear at the target"
    assert leftovers(tmp_path) == [], "the temp file must be discarded too"
    assert episode_bases(tmp_path) == [], "and the scanner must see no episode"
    assert os.listdir(tmp_path) == [], "not a single file, useful-looking or not"


def test_a_failed_download_is_retried_on_the_next_run(tmp_path, server):
    """The point of leaving nothing behind: the retry is not blocked.

    The old behaviour committed the stump, and ``os.path.exists`` then made
    every later run skip it - the corruption was permanent.
    """
    server.routes["/episode_42.mp3"] = close_delimited_short(PAYLOAD)
    feed_xml = rss("episode_42", url_for(server, "/episode_42.mp3"), len(PAYLOAD))

    assert podcast_fetching_workflow(feed_xml, str(tmp_path)) is None

    server.routes["/episode_42.mp3"] = whole_body(PAYLOAD)
    second = podcast_fetching_workflow(feed_xml, str(tmp_path))

    assert second == str(tmp_path / "episode_42.mp3")
    assert (tmp_path / "episode_42.mp3").read_bytes() == PAYLOAD
    assert server.hits.count("/episode_42.mp3") == 2, "the second run must re-fetch"


def test_download_all_episodes_discards_a_truncated_body_too(tmp_path, server):
    """The second call site gets the same treatment as the first."""
    server.routes["/episode_43.mp3"] = close_delimited_short(PAYLOAD)
    feed_xml = rss("episode_43", url_for(server, "/episode_43.mp3"), len(PAYLOAD))

    download_all_episodes_from_rssfeed(feed_xml, str(tmp_path))

    assert os.listdir(tmp_path) == []
    assert episode_bases(tmp_path) == []


def test_download_all_episodes_keeps_a_complete_body(tmp_path, server):
    server.routes["/episode_43.mp3"] = whole_body(PAYLOAD)
    feed_xml = rss("episode_43", url_for(server, "/episode_43.mp3"), len(PAYLOAD))

    download_all_episodes_from_rssfeed(feed_xml, str(tmp_path))

    assert (tmp_path / "episode_43.mp3").read_bytes() == PAYLOAD


# ---------------------------------------------------------------------------
# An existing file is not collateral damage
# ---------------------------------------------------------------------------

def test_a_failed_re_download_leaves_an_existing_file_byte_identical(tmp_path, server):
    """The real claim: the download machinery cannot damage what is already there.

    Deliberately *not* routed through ``podcast_fetching_workflow``, which
    short-circuits on ``os.path.exists`` and would never attempt the download -
    that test passes for the wrong reason. This drives the streaming write onto
    an existing path and truncates it mid-body.
    """
    target = tmp_path / "episode_42.mp3"
    target.write_bytes(PAYLOAD)

    server.routes["/episode_42.mp3"] = close_delimited_short(PAYLOAD)

    with pytest.raises(Exception):
        _download_audio(url_for(server, "/episode_42.mp3"), str(target), len(PAYLOAD))

    assert target.read_bytes() == PAYLOAD, "the previous recording must be untouched"
    assert leftovers(tmp_path) == []


def test_an_existing_episode_is_returned_without_a_request(tmp_path, server):
    """The other half: the feed path does not even ask when the file is there."""
    target = tmp_path / "episode_42.mp3"
    target.write_bytes(PAYLOAD)
    server.routes["/episode_42.mp3"] = whole_body(b"REPLACEMENT")
    feed_xml = rss("episode_42", url_for(server, "/episode_42.mp3"), len(PAYLOAD))

    result = podcast_fetching_workflow(feed_xml, str(tmp_path), return_base_name=True)

    assert result == (str(target), "episode_42")
    assert target.read_bytes() == PAYLOAD
    assert server.hits == [], "an existing episode is never re-fetched"


# ---------------------------------------------------------------------------
# A stalled server must not hang the job forever
# ---------------------------------------------------------------------------

def test_a_stalled_download_gives_up_and_leaves_nothing(tmp_path, server, monkeypatch):
    """Without a timeout this test never returns; that was the bug.

    The timeout is shortened rather than waited out, so this stays a fast test:
    what is being pinned is that a per-chunk read timeout is configured and that
    hitting it goes down the normal failure path.
    """
    monkeypatch.setattr(feed_module, "DOWNLOAD_TIMEOUT", (5, 0.3))
    server.routes["/stall.mp3"] = stalls_after(PAYLOAD, server.released)
    feed_xml = rss("stall", url_for(server, "/stall.mp3"), len(PAYLOAD))

    result = podcast_fetching_workflow(feed_xml, str(tmp_path), return_base_name=True)

    assert result == (None, None)
    assert os.listdir(tmp_path) == []


def test_the_download_timeout_is_a_connect_read_pair():
    connect, read = feed_module.DOWNLOAD_TIMEOUT
    assert connect > 0 and read > 0


# ---------------------------------------------------------------------------
# The size checks themselves, at close range
# ---------------------------------------------------------------------------

def test_verify_accepts_a_body_that_matches_both_witnesses():
    _verify_download("http://x/a.mp3", 1000, 1000, 1000)  # must not raise


def test_verify_rejects_a_body_shorter_than_content_length():
    with pytest.raises(Exception) as excinfo:
        _verify_download("http://x/a.mp3", 999, 1000, None)
    assert "Content-Length" in str(excinfo.value)


def test_verify_rejects_an_empty_body_even_when_the_header_agrees():
    with pytest.raises(Exception):
        _verify_download("http://x/a.mp3", 0, 0, None)


def test_verify_rejects_a_body_shorter_than_the_feed_declared():
    with pytest.raises(Exception) as excinfo:
        _verify_download("http://x/a.mp3", 500, None, 1000)
    assert "500" in str(excinfo.value)


def test_verify_accepts_a_body_longer_than_the_feed_declared():
    _verify_download("http://x/a.mp3", 1500, None, 1000)  # must not raise


def test_verify_accepts_anything_non_empty_when_nobody_declared_a_size():
    _verify_download("http://x/a.mp3", 1, None, None)  # must not raise


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("409610", 409610),
        (" 409610 ", 409610),
        ("0", None),          # "no information", never "expect zero bytes"
        ("-1", None),
        ("not a number", None),
        (None, None),
    ],
)
def test_enclosure_length_is_parsed_defensively(raw, expected):
    link = {"type": "audio/mpeg", "href": "http://x/a.mp3"}
    if raw is not None:
        link["length"] = raw
    assert _enclosure_length(link) == expected


class _FakeResponse:
    def __init__(self, headers):
        self.headers = headers


def test_header_length_reads_content_length():
    assert _header_length(_FakeResponse({"Content-Length": "1234"})) == 1234


def test_header_length_ignores_a_compressed_body():
    """requests hands back decoded bytes; the header counts encoded ones."""
    response = _FakeResponse({"Content-Length": "1234", "Content-Encoding": "gzip"})
    assert _header_length(response) is None


def test_header_length_is_none_when_the_server_does_not_say():
    assert _header_length(_FakeResponse({})) is None


# ---------------------------------------------------------------------------
# Deleting one episode's files must not touch another's
# ---------------------------------------------------------------------------

def test_deleting_episode_1_spares_episode_10_through_19(tmp_path):
    """The prefix glob this replaced destroyed nineteen other episodes.

    ``episode_1`` and ``episode_10``..``episode_19`` all exist in podcasts/,
    artifacts carry no .bak (ADR-009) and the directory is gitignored, so the
    old ``episode_1*.*`` glob made one "reprocess (force)" click terminal for
    all of them.
    """
    names = [
        "episode_1.mp3", "episode_1_transcript.txt", "episode_1_summary.txt",
        "episode_10.mp3", "episode_10_transcript.txt",
        "episode_10_speaker_assignment.txt",
        "episode_19_speaker_assignment.txt", "episode_2_summary.txt",
        "episode_100_blog.txt",
    ]
    for name in names:
        (tmp_path / name).write_text("HAND CORRECTED CONTENT", encoding="utf-8")

    delete_episode_files(
        "episode_1", str(tmp_path), exclude_files=[str(tmp_path / "episode_1.mp3")]
    )

    assert sorted(os.listdir(tmp_path)) == sorted([
        "episode_1.mp3",
        "episode_10.mp3", "episode_10_transcript.txt",
        "episode_10_speaker_assignment.txt",
        "episode_19_speaker_assignment.txt", "episode_2_summary.txt",
        "episode_100_blog.txt",
    ])


def test_deleting_an_episode_still_removes_its_own_artifacts(tmp_path):
    """The boundary rule must not have made the function useless."""
    for name in [
        "episode_1.mp3", "episode_1.txt", "episode_1_transcript.txt",
        "episode_1_transcript_ts.txt", "episode_1_summary.html",
        "episode_1_summary.wiki", "episode_1_merged.txt",
        "episode_1_summary.txt.bak",
    ]:
        (tmp_path / name).write_text("generated", encoding="utf-8")

    delete_episode_files("episode_1", str(tmp_path))

    assert os.listdir(tmp_path) == ["episode_1.mp3"], "audio is never deleted"


def test_deleting_an_episode_spares_the_excluded_input(tmp_path):
    (tmp_path / "episode_1.wav").write_bytes(b"source recording")
    (tmp_path / "episode_1_summary.txt").write_text("x", encoding="utf-8")

    delete_episode_files(
        "episode_1", str(tmp_path), exclude_files=[str(tmp_path / "episode_1.wav")]
    )

    assert os.listdir(tmp_path) == ["episode_1.wav"]


def test_deleting_an_episode_leaves_a_live_temp_file_alone(tmp_path):
    """A ``.tmp`` from another writer is not this episode's to remove."""
    (tmp_path / "episode_1_summary.txt").write_text("x", encoding="utf-8")
    stray = tmp_path / (".episode_1_summary.txt.abcd1234" + TEMP_SUFFIX)
    stray.write_bytes(b"half an artifact")

    delete_episode_files("episode_1", str(tmp_path))

    assert os.listdir(tmp_path) == [stray.name]
