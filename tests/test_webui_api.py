"""
Tests for the WHYcast web UI HTTP layer (ADR-008 / TASK-002 phase 1).

Everything runs against a **synthetic** podcast directory in ``tmp_path`` and a
**temporary** SQLite index. The operator's real ``podcasts/`` and real
``webui/whycast_webui.db`` are never touched: the module-level
``webui.app.app`` object is built at import time from the live environment, so
these tests deliberately never use it and always build their own app with
:func:`webui.app.create_app` after pointing the ADR-007 environment variables
at the temporary directories.

``TestClient`` is always used as a context manager. The index connection is
opened in the FastAPI lifespan handler; without ``with``, ``app.state.conn``
stays ``None`` and every route answers 503 instead of doing its job.

Covered here, mapped to the TASK-002 acceptance criteria:

* AC #1 - every episode visible with the right artifact status (``/``,
  ``/api/episodes``)
* AC #2 - artifact content served, audio streamable
  (``/api/episodes/{base}/artifacts/{kind}``, ``/media/{base}/audio``)
* AC #3 - unmatched files visible rather than silently ignored (``/unmatched``)
* AC #4 - the index is disposable: delete the file, restart, it rebuilds
* AC #5 - the server binds to 127.0.0.1 by default

Plus the path-traversal guard that TASK-002's plan asks for on artifact
serving, which is the one place a request could otherwise reach a file.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from webui import app as app_module  # noqa: E402
from webui import db as db_module  # noqa: E402

#: Written outside the podcast directory. No response body may ever contain it.
SECRET_MARKER = "TOP-SECRET-OUTSIDE-THE-PODCAST-DIRECTORY"

#: The synthetic podcast directory: a small, exactly-known version of the
#: naming chaos in the real one. Contents double as expected response bodies.
FILES = {
    "episode_1.mp3": "fake-mp3-bytes-for-one",
    "episode_1.txt": "bare transcript of episode one\n",
    "episode_1_summary.txt": "summary of episode one\n",
    "episode_1_blog.html": "<p>blog of episode one</p>\n",
    "episode_1_blog.txt": "blog of episode one, as text\n",
    # Two files in one (kind, fmt) slot; mtimes are set in the fixture so the
    # "newest wins" rule has a known answer.
    "episode_1_speaker_assignment.txt": "SUPERSEDED assignment\n",
    "episode_1_ts_speaker_assignment.txt": "CURRENT assignment\n",
    # The prefix collision, so the API is exercised on a directory where it
    # matters and not only the scanner.
    "episode_10.mp3": "fake-mp3-bytes-for-ten",
    "episode_10_blog.txt": "blog of episode ten\n",
    # A base name with a space: it has to survive URL round-tripping.
    "Episode 28.mp3": "fake-mp3-bytes-for-twenty-eight",
    "Episode 28_summary.txt": "summary of episode twenty-eight\n",
    # Audio-less episode: artifacts without an mp3.
    "episode_42_summary.txt": "summary of episode forty-two\n",
    "episode_42_ts.txt": "timestamped forty-two\n",
    # Junk, so /unmatched has something to show (AC #3).
    "E99 Thumbnail.jpg": "not an artifact",
}

EXPECTED_BASE_NAMES = [
    "episode_1",
    "episode_10",
    "Episode 28",
    "episode_42",
]

#: Keys every episode dict in the API must carry. The overview template reads
#: `formats` and `kinds`; a rename upstream must fail here, not silently draw
#: an empty status matrix.
EPISODE_KEYS = {
    "base_key",
    "base_name",
    "number",
    "audio_path",
    "audio_size",
    "has_audio",
    "mtime",
    "mtime_iso",
    "position",
    "artifact_count",
    "kinds",
    "formats",
}

ARTIFACT_KEYS = {"kind", "fmt", "path", "size", "mtime", "position"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def write(path: Path, content: str) -> None:
    """Write ``content`` byte for byte.

    Not ``Path.write_text``: on Windows that opens in text mode and rewrites
    every ``\\n`` as ``\\r\\n``, so the bytes on disk stop matching the string
    the test compares the response against. The pipeline writes artifacts
    verbatim and the API serves them as bytes; the fixture has to do the same
    or the test measures Python's newline translation instead of the app.
    """
    path.write_bytes(content.encode("utf-8"))


@pytest.fixture
def podcast_dir(tmp_path):
    """A synthetic podcasts directory, plus a secret file outside of it."""
    root = tmp_path / "podcasts"
    root.mkdir()
    for name, content in FILES.items():
        write(root / name, content)

    # Superseded first, current second: the API must serve the newer file.
    os.utime(root / "episode_1_speaker_assignment.txt", (1_700_000_000, 1_700_000_000))
    os.utime(
        root / "episode_1_ts_speaker_assignment.txt", (1_700_086_400, 1_700_086_400)
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    write(outside / "secret.txt", SECRET_MARKER)
    return root


@pytest.fixture
def db_file(tmp_path):
    return tmp_path / "index" / "whycast_webui_test.db"


@pytest.fixture
def configured_env(monkeypatch, podcast_dir, db_file):
    """Point the ADR-007 environment variables at the temporary directories.

    ``create_app()`` is then called with no arguments on purpose: that is the
    configuration path the server itself uses, so this exercises
    ``podcast_dir_from_env`` / ``db_path_from_env`` rather than bypassing them.
    """
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcast_dir))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(db_file))
    monkeypatch.delenv("WHYCAST_WEBUI_HOST", raising=False)
    monkeypatch.delenv("WHYCAST_WEBUI_PORT", raising=False)
    monkeypatch.delenv("WHYCAST_WEBUI_ALLOWED_HOSTS", raising=False)
    return {"podcast_dir": podcast_dir, "db_file": db_file}


@pytest.fixture
def client(configured_env):
    """A started TestClient over a freshly built app.

    The ``with`` block is what runs the lifespan handler, which opens the index
    and rescans it. Without it every route returns 503.
    """
    app = app_module.create_app()
    with TestClient(app) as test_client:
        yield test_client


def _dump_index(client) -> dict:
    """Every row of the three index tables, in scan order.

    Used to prove rescan idempotency by rows rather than by counts: "two runs
    over an unchanged directory produce identical rows" is what
    :func:`webui.db.rescan` documents.
    """
    conn = client.app.state.conn
    tables = {}
    # artifacts.position numbers files WITHIN an episode, so it is not unique
    # across the table; ordering by it alone would leave ties to whatever
    # SQLite's page layout happens to return and could fail spuriously.
    order = {
        "episodes": "position",
        "artifacts": "base_key, position",
        "unmatched": "position",
    }
    for table, by in order.items():
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY {by}").fetchall()
        tables[table] = [tuple(row) for row in rows]
    return tables


# ---------------------------------------------------------------------------
# Pages (AC #1, #2, #3)
# ---------------------------------------------------------------------------


class TestPages:
    def test_index_lists_every_synthetic_episode(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        for base_name in EXPECTED_BASE_NAMES:
            assert base_name in response.text, f"{base_name} missing from the overview"

    def test_index_reports_the_episode_count(self, client):
        response = client.get("/")
        assert f"{len(EXPECTED_BASE_NAMES)} episodes" in response.text

    def test_status_matrix_marks_present_and_missing_kinds(self, client):
        """AC #1: the matrix must be *right*, not merely rendered.

        The overview calls ``db.list_episodes()``, which does not nest artifact
        rows; a template that reached for ``episode.artifacts`` here would find
        Undefined - falsy, never an error - and silently draw every cell as
        missing while still answering 200. A status-code assertion cannot see
        that, so the cell titles are asserted instead.
        """
        body = client.get("/").text
        assert 'title="blog: txt, html"' in body
        assert 'title="transcript: txt"' in body
        assert 'title="summary: txt"' in body
        assert 'title="merged: missing"' in body
        # One filled cell per (episode, kind) that exists: 4 for episode_1,
        # 1 for episode_10, 1 for "Episode 28", 2 for episode_42.
        assert body.count('class="cell on"') == 8

    def test_audio_column_distinguishes_audio_less_episodes(self, client):
        body = client.get("/").text
        assert body.count('class="cell audio on"') == 3
        assert body.count('class="cell audio off"') == 1

    def test_index_search_filters(self, client):
        response = client.get("/", params={"search": "episode_10"})
        assert response.status_code == 200
        assert "episode_10" in response.text
        # "episode_1" is a substring of "episode_10", so it may appear; the
        # unrelated episodes must not.
        assert "Episode 28" not in response.text
        assert "episode_42" not in response.text

    def test_episode_detail_page_renders(self, client):
        response = client.get("/episodes/episode_1")
        assert response.status_code == 200
        assert "episode_1" in response.text

    def test_detail_page_links_each_present_artifact_and_marks_the_rest(self, client):
        """AC #2: every artifact the episode has is reachable from the page."""
        body = client.get("/episodes/episode_1").text
        for kind in ("transcript", "summary", "blog", "speaker_assignment"):
            assert f'href="/api/episodes/episode_1/artifacts/{kind}"' in body
        # Kinds this episode lacks are shown as absent, not as dead links.
        assert 'href="/api/episodes/episode_1/artifacts/merged"' not in body
        assert 'title="merged: missing"' in body

    def test_detail_page_offers_the_audio_element(self, client):
        """AC #2: the mp3 must be playable from the page."""
        body = client.get("/episodes/episode_1").text
        assert 'src="/media/episode_1/audio"' in body
        assert "<audio" in body
        assert 'preload="none"' in body

    def test_audio_less_episode_says_so_instead_of_offering_a_dead_player(self, client):
        body = client.get("/episodes/episode_42").text
        assert "/media/episode_42/audio" not in body
        assert "No audio file for this episode" in body

    def test_artifact_content_is_never_merged_into_the_page(self, client):
        """Artifacts are prompt-injectable LLM output; they load sandboxed.

        A previous version swapped artifact bodies in with htmx
        ``hx-swap="innerHTML"``, which makes the API's ``sandbox`` CSP inert
        because there is no browsing context to apply it to. The tab links must
        stay plain navigations into the sandboxed frame.
        """
        body = client.get("/episodes/episode_1").text
        assert 'id="artifact-frame"' in body
        assert "sandbox" in body
        assert 'target="artifact-frame"' in body
        # Narrow on purpose: htmx elsewhere on the page is fine (the overview
        # already uses it for search). What must never exist is an htmx fetch
        # of artifact content, because a response CSP cannot govern a string
        # handed to innerHTML.
        assert "hx-get" not in body or "/artifacts/" not in body.split("hx-get")[1]
        assert 'hx-swap="innerHTML"' not in body
        assert FILES["episode_1_summary.txt"].strip() not in body

    def test_detail_page_lists_the_files_on_disk(self, client):
        body = client.get("/episodes/episode_1").text
        assert "Files on disk" in body
        assert "6 artifacts" in body

    def test_episode_detail_page_for_a_name_with_a_space(self, client):
        response = client.get("/episodes/Episode%2028")
        assert response.status_code == 200
        assert "Episode 28" in response.text

    def test_unknown_episode_page_is_404(self, client):
        response = client.get(
            "/episodes/no-such-episode", headers={"accept": "text/html"}
        )
        assert response.status_code == 404
        assert "404" in response.text

    def test_unmatched_page_shows_ignored_files(self, client):
        """AC #3: junk is visible, not silently dropped."""
        response = client.get("/unmatched")
        assert response.status_code == 200
        assert "E99 Thumbnail.jpg" in response.text
        assert "1 entry" in response.text

    def test_unmatched_page_renders_fields_not_a_python_dict(self, client):
        """``list_unmatched`` yields dicts; iterating them as strings printed
        the repr into the cell once. Pin the fields instead."""
        body = client.get("/unmatched").text
        assert "{'path':" not in body
        assert '"path":' not in body
        assert ".jpg" in body  # the ext column

    def test_pages_carry_the_security_headers(self, client):
        response = client.get("/")
        assert "default-src 'self'" in response.headers["content-security-policy"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"


# ---------------------------------------------------------------------------
# GET /api/episodes  (AC #1)
# ---------------------------------------------------------------------------


class TestEpisodesApi:
    def test_shape_matches_the_contract(self, client):
        payload = client.get("/api/episodes").json()
        assert set(payload) == {"count", "search", "episodes"}
        assert payload["search"] is None
        assert payload["count"] == len(payload["episodes"]) == len(EXPECTED_BASE_NAMES)

        for episode in payload["episodes"]:
            assert set(episode) == EPISODE_KEYS
            assert isinstance(episode["kinds"], list)
            assert isinstance(episode["formats"], dict)
            assert episode["base_key"] == episode["base_name"].lower()
            assert episode["has_audio"] == (episode["audio_path"] is not None)
            assert episode["artifact_count"] >= 0

    def test_episodes_are_in_scan_order(self, client):
        payload = client.get("/api/episodes").json()
        assert [e["base_name"] for e in payload["episodes"]] == EXPECTED_BASE_NAMES
        assert [e["position"] for e in payload["episodes"]] == [0, 1, 2, 3]

    def test_artifact_status_per_kind_is_correct(self, client):
        """AC #1: the status matrix has to be right, not merely present."""
        by_name = {
            e["base_name"]: e for e in client.get("/api/episodes").json()["episodes"]
        }

        one = by_name["episode_1"]
        assert set(one["kinds"]) == {
            "transcript",
            "summary",
            "blog",
            "speaker_assignment",
        }
        # ARTIFACT_KINDS order, not alphabetical or insertion order.
        assert one["kinds"] == ["transcript", "summary", "blog", "speaker_assignment"]
        assert one["formats"]["blog"] == ["txt", "html"]  # preference order
        assert one["formats"]["transcript"] == ["txt"]
        # Six files across four kinds: the count is per file, not per kind,
        # and the two speaker_assignment files both count.
        assert one["artifact_count"] == 6

        ten = by_name["episode_10"]
        assert ten["kinds"] == ["blog"]
        assert "blog" not in one["formats"].get("nonexistent", [])

        forty_two = by_name["episode_42"]
        assert forty_two["has_audio"] is False
        assert forty_two["audio_path"] is None
        assert set(forty_two["kinds"]) == {"ts", "summary"}

    def test_the_prefix_collision_survives_the_index(self, client):
        """episode_10's blog must not show up under episode_1."""
        by_name = {
            e["base_name"]: e for e in client.get("/api/episodes").json()["episodes"]
        }
        paths = client.get("/api/episodes/episode_1").json()["artifacts"]
        assert all("episode_10" not in os.path.basename(a["path"]) for a in paths)
        assert by_name["episode_10"]["artifact_count"] == 1

    def test_search_parameter_is_echoed_and_applied(self, client):
        payload = client.get("/api/episodes", params={"search": "42"}).json()
        assert payload["search"] == "42"
        assert [e["base_name"] for e in payload["episodes"]] == ["episode_42"]

    def test_search_underscore_is_literal_not_a_wildcard(self, client):
        """``_`` is a LIKE wildcard; base names are full of them."""
        payload = client.get("/api/episodes", params={"search": "episode_4"}).json()
        assert [e["base_name"] for e in payload["episodes"]] == ["episode_42"]


# ---------------------------------------------------------------------------
# GET /api/episodes/{base_name}
# ---------------------------------------------------------------------------


class TestEpisodeDetailApi:
    def test_returns_the_artifact_rows(self, client):
        episode = client.get("/api/episodes/episode_1").json()
        assert episode["base_name"] == "episode_1"
        assert episode["artifacts"], "the detail endpoint must nest artifact rows"
        for artifact in episode["artifacts"]:
            assert ARTIFACT_KEYS <= set(artifact)
            assert os.path.isabs(artifact["path"])
        kinds = {a["kind"] for a in episode["artifacts"]}
        assert kinds == {"transcript", "summary", "blog", "speaker_assignment"}
        assert len(episode["artifacts"]) == 6

    def test_lookup_is_case_insensitive(self, client):
        """The scanner groups case-insensitively; the API must agree."""
        assert client.get("/api/episodes/EPISODE_1").status_code == 200
        assert client.get("/api/episodes/Episode_1").json()["base_name"] == "episode_1"

    def test_name_with_a_space(self, client):
        episode = client.get("/api/episodes/Episode%2028").json()
        assert episode["base_name"] == "Episode 28"
        assert episode["number"] == 28

    def test_unknown_base_name_is_404(self, client):
        response = client.get("/api/episodes/nope")
        assert response.status_code == 404
        assert response.json()["detail"] == "Episode not found"

    def test_api_404_answers_json_even_to_a_browser(self, client):
        response = client.get("/api/episodes/nope", headers={"accept": "text/html"})
        assert response.status_code == 404
        assert response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/episodes/{base_name}/artifacts/{kind}   (AC #2)
# ---------------------------------------------------------------------------


class TestArtifactApi:
    def test_returns_the_file_content(self, client):
        response = client.get("/api/episodes/episode_1/artifacts/summary")
        assert response.status_code == 200
        assert response.text == FILES["episode_1_summary.txt"]
        assert response.headers["content-type"].startswith("text/plain")

    def test_bare_transcript_is_served_as_the_transcript(self, client):
        response = client.get("/api/episodes/episode_1/artifacts/transcript")
        assert response.status_code == 200
        assert response.text == FILES["episode_1.txt"]

    def test_format_preference_picks_txt_over_html(self, client):
        response = client.get("/api/episodes/episode_1/artifacts/blog")
        assert response.text == FILES["episode_1_blog.txt"]

    def test_explicit_format_is_honoured(self, client):
        response = client.get(
            "/api/episodes/episode_1/artifacts/blog", params={"fmt": "html"}
        )
        assert response.status_code == 200
        assert response.text == FILES["episode_1_blog.html"]
        assert response.headers["content-type"].startswith("text/html")

    def test_html_artifacts_are_sandboxed(self, client):
        """Artifacts are LLM output: never trusted, never merged into a page."""
        response = client.get(
            "/api/episodes/episode_1/artifacts/blog", params={"fmt": "html"}
        )
        csp = response.headers["content-security-policy"]
        assert "sandbox" in csp
        assert "default-src 'none'" in csp
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_newest_file_wins_when_two_share_a_slot(self, client):
        """``_speaker_assignment`` vs ``_ts_speaker_assignment``: serve the new one.

        Ordering by path alone would return the superseded file forever, since
        ``_speaker`` sorts before ``_ts_speaker``. This is also the one place
        ``app._pick_artifact`` and ``episodes.Episode.artifact`` could drift.
        """
        response = client.get("/api/episodes/episode_1/artifacts/speaker_assignment")
        assert response.status_code == 200
        assert response.text == FILES["episode_1_ts_speaker_assignment.txt"]

    def test_unknown_kind_is_404(self, client):
        response = client.get("/api/episodes/episode_1/artifacts/definitely_not_a_kind")
        assert response.status_code == 404
        assert "artifact kind" in response.text

    def test_known_kind_the_episode_does_not_have_is_404(self, client):
        response = client.get("/api/episodes/episode_1/artifacts/merged")
        assert response.status_code == 404
        assert "merged" in response.text and "episode_1" in response.text

    def test_a_missing_artifact_says_so_inside_the_frame(self, client):
        """These 404s are read in an iframe, so they must be framable.

        A JSON HTTPException got the *page* headers - ``X-Frame-Options: DENY``
        and ``frame-ancestors 'none'`` - so the browser refused to render it in
        the artifact frame and showed its own "can't open this page" instead.
        The reader was told a security policy had intervened when a job had
        simply moved the file aside a moment earlier.
        """
        response = client.get("/api/episodes/episode_1/artifacts/merged")

        assert response.status_code == 404
        assert "x-frame-options" not in {k.lower() for k in response.headers}
        policy = response.headers["content-security-policy"]
        assert "frame-ancestors 'none'" not in policy
        assert response.headers["content-type"].startswith("text/plain")
        # Says what to do, not just that something is missing.
        assert "rescan" in response.text.lower()

    def test_unknown_format_is_404(self, client):
        response = client.get(
            "/api/episodes/episode_1/artifacts/blog", params={"fmt": "exe"}
        )
        assert response.status_code == 404

    def test_unknown_episode_is_404(self, client):
        response = client.get("/api/episodes/nope/artifacts/summary")
        assert response.status_code == 404

    def test_head_is_allowed(self, client):
        """The episode page probes artifacts with HEAD before embedding them."""
        response = client.head("/api/episodes/episode_1/artifacts/summary")
        assert response.status_code == 200

    def test_a_file_removed_behind_the_index_is_404_not_a_500(self, client, podcast_dir):
        (podcast_dir / "episode_1_summary.txt").unlink()
        response = client.get("/api/episodes/episode_1/artifacts/summary")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Audio (AC #2)
# ---------------------------------------------------------------------------


class TestAudio:
    def test_audio_streams(self, client):
        response = client.get("/media/episode_1/audio")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.text == FILES["episode_1.mp3"]

    def test_head_is_allowed(self, client):
        assert client.head("/media/episode_1/audio").status_code == 200

    def test_range_requests_are_answered(self, client):
        """Without 206 the browser downloads the whole file before it can seek."""
        response = client.get("/media/episode_1/audio", headers={"range": "bytes=0-3"})
        assert response.status_code == 206
        assert response.text == FILES["episode_1.mp3"][:4]

    def test_audio_less_episode_is_404(self, client):
        response = client.get("/media/episode_42/audio")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/rescan  (AC #4)
# ---------------------------------------------------------------------------


class TestRescan:
    def test_is_idempotent(self, client):
        first = client.post("/api/rescan")
        assert first.status_code == 200
        rows_after_first = _dump_index(client)

        second = client.post("/api/rescan")
        assert second.status_code == 200
        rows_after_second = _dump_index(client)

        count_keys = (
            "episodes",
            "episodes_with_audio",
            "episodes_without_audio",
            "artifacts",
            "unmatched",
            "podcast_dir",
        )
        a = first.json()["rescan"]
        b = second.json()["rescan"]
        assert {k: a[k] for k in count_keys} == {k: b[k] for k in count_keys}
        # Stronger than counts: the documented guarantee is identical rows.
        assert rows_after_first == rows_after_second

    def test_reports_the_synthetic_directory_accurately(self, client, podcast_dir):
        summary = client.post("/api/rescan").json()["rescan"]
        assert summary["episodes"] == len(EXPECTED_BASE_NAMES)
        assert summary["episodes_with_audio"] == 3
        assert summary["episodes_without_audio"] == 1
        assert summary["unmatched"] == 1
        assert os.path.normcase(summary["podcast_dir"]) == os.path.normcase(
            str(podcast_dir)
        )

    def test_picks_up_a_newly_created_artifact(self, client, podcast_dir):
        before = client.post("/api/rescan").json()["rescan"]
        assert client.get("/api/episodes/episode_10/artifacts/summary").status_code == 404

        write(podcast_dir / "episode_10_summary.txt", "brand new summary\n")

        after = client.post("/api/rescan").json()["rescan"]
        assert after["artifacts"] == before["artifacts"] + 1
        assert after["episodes"] == before["episodes"]

        response = client.get("/api/episodes/episode_10/artifacts/summary")
        assert response.status_code == 200
        assert response.text == "brand new summary\n"

    def test_picks_up_a_newly_created_episode(self, client, podcast_dir):
        before = client.post("/api/rescan").json()["rescan"]
        write(podcast_dir / "episode_99.mp3", "fake-mp3")

        after = client.post("/api/rescan").json()["rescan"]
        assert after["episodes"] == before["episodes"] + 1
        assert client.get("/api/episodes/episode_99").json()["number"] == 99

    def test_drops_a_deleted_artifact(self, client, podcast_dir):
        before = client.post("/api/rescan").json()["rescan"]
        (podcast_dir / "episode_1_summary.txt").unlink()

        after = client.post("/api/rescan").json()["rescan"]
        assert after["artifacts"] == before["artifacts"] - 1
        assert client.get("/api/episodes/episode_1/artifacts/summary").status_code == 404

    def test_reports_meta_from_the_allowlist_only(self, client):
        """ADR-008 Must: configuration surfaces go through an allowlist."""
        meta = client.post("/api/rescan").json()["meta"]
        assert set(meta) <= set(app_module.META_KEYS)
        assert "OPENAI_API_KEY" not in json.dumps(meta)
        assert meta["episode_count"] == len(EXPECTED_BASE_NAMES)


# ---------------------------------------------------------------------------
# Index rebuildability (AC #4)
# ---------------------------------------------------------------------------


class TestIndexIsDisposable:
    def test_deleting_the_database_and_restarting_rebuilds_the_index(
        self, configured_env
    ):
        db_file = configured_env["db_file"]

        app = app_module.create_app()
        with TestClient(app) as client:
            before = client.get("/api/episodes").json()
            unmatched_before = client.get("/api/health").json()["meta"][
                "unmatched_count"
            ]
        assert before["count"] == len(EXPECTED_BASE_NAMES)
        assert db_file.exists()

        # Remove the WAL siblings too, so this proves a rebuild from disk and
        # not a journal recovery.
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(db_file) + suffix)
            if candidate.exists():
                candidate.unlink()
        assert not db_file.exists()

        restarted = app_module.create_app()
        with TestClient(restarted) as client:
            after = client.get("/api/episodes").json()
            unmatched_after = client.get("/api/health").json()["meta"]["unmatched_count"]

        assert db_file.exists(), "the index file must be recreated"
        assert after["count"] == before["count"]
        assert [e["base_name"] for e in after["episodes"]] == [
            e["base_name"] for e in before["episodes"]
        ]
        # "Without data loss" is about content, not merely about row counts.
        for old, new in zip(before["episodes"], after["episodes"]):
            assert old == new
        assert unmatched_after == unmatched_before

    def test_the_index_never_writes_to_the_podcast_directory(self, configured_env):
        """ADR-008: the web layer must not write episode artifacts."""
        podcast_dir = configured_env["podcast_dir"]
        before = {
            name: os.stat(podcast_dir / name).st_mtime
            for name in os.listdir(podcast_dir)
        }
        app = app_module.create_app()
        with TestClient(app) as client:
            client.get("/")
            client.post("/api/rescan")
            client.get("/api/episodes/episode_1/artifacts/summary")
        after = {
            name: os.stat(podcast_dir / name).st_mtime
            for name in os.listdir(podcast_dir)
        }
        assert before == after

    def test_the_database_lives_where_the_environment_says(self, configured_env):
        app = app_module.create_app()
        assert os.path.normcase(app.state.db_path) == os.path.normcase(
            str(configured_env["db_file"])
        )
        assert os.path.normcase(app.state.podcast_dir) == os.path.normcase(
            str(configured_env["podcast_dir"])
        )


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------


#: Hostile values for the ``base_name`` path segment. Each must produce a 4xx
#: and, more importantly, must never return the content of a file outside the
#: podcast directory.
TRAVERSAL_BASE_NAMES = [
    pytest.param("../../outside/secret", id="literal-dotdot"),
    pytest.param("..%2f..%2foutside%2fsecret", id="encoded-slash"),
    pytest.param("%2e%2e%2f%2e%2e%2foutside%2fsecret", id="encoded-dotdot-and-slash"),
    pytest.param("....//....//outside//secret", id="doubled-dotdot"),
    pytest.param("..%5c..%5coutside%5csecret", id="encoded-backslash"),
    pytest.param("%252e%252e%252foutside", id="double-encoded"),
]


class TestPathTraversal:
    """No request may reach a file outside the podcast directory.

    Both halves of each assertion matter. The status code proves the request
    was refused; the body check proves it was refused *before* reading
    anything, which is the part that would actually leak. The exact code is
    left loose (404 from the router, 405, 421 from the Host guard) because it
    shifts with framework versions, while "no secret bytes" never should.
    """

    @pytest.mark.parametrize("hostile", TRAVERSAL_BASE_NAMES)
    def test_hostile_base_name_on_the_artifact_route(self, client, hostile):
        response = client.get(f"/api/episodes/{hostile}/artifacts/transcript")
        assert 400 <= response.status_code < 500, response.status_code
        assert SECRET_MARKER not in response.text

    @pytest.mark.parametrize("hostile", TRAVERSAL_BASE_NAMES)
    def test_hostile_base_name_on_the_episode_route(self, client, hostile):
        response = client.get(f"/api/episodes/{hostile}")
        assert 400 <= response.status_code < 500, response.status_code
        assert SECRET_MARKER not in response.text

    @pytest.mark.parametrize("hostile", TRAVERSAL_BASE_NAMES)
    def test_hostile_base_name_on_the_audio_route(self, client, hostile):
        response = client.get(f"/media/{hostile}/audio")
        assert 400 <= response.status_code < 500, response.status_code
        assert SECRET_MARKER not in response.text

    @pytest.mark.parametrize(
        "hostile",
        [
            "../../outside/secret.txt",
            "..%2f..%2foutside%2fsecret.txt",
            "%2e%2e%2foutside",
            "transcript%00.txt",
        ],
    )
    def test_hostile_kind(self, client, hostile):
        response = client.get(f"/api/episodes/episode_1/artifacts/{hostile}")
        assert 400 <= response.status_code < 500, response.status_code
        assert SECRET_MARKER not in response.text

    @pytest.mark.parametrize(
        "hostile",
        ["../../outside/secret", "txt/../../../outside/secret", "..\\..\\outside", "\x00"],
    )
    def test_hostile_fmt_query_parameter(self, client, hostile):
        response = client.get(
            "/api/episodes/episode_1/artifacts/blog", params={"fmt": hostile}
        )
        assert response.status_code == 404
        assert SECRET_MARKER not in response.text

    @pytest.mark.parametrize(
        "hostile",
        [
            "C:\\Windows\\win.ini",
            "D:/Windows/win.ini",
            "\\\\server\\share\\secret",
            "..\\..\\outside\\secret",
        ],
    )
    def test_absolute_and_unc_paths_are_looked_up_not_opened(self, client, hostile):
        """No forward slash, so the route matches and the value reaches the index.

        This is the interesting case: ``base_name`` is only ever a SQLite
        lookup key, so an absolute path simply misses. If it were ever joined
        into a filesystem path, this test is where that would show.
        """
        response = client.get(f"/api/episodes/{hostile}/artifacts/transcript")
        assert response.status_code == 404
        assert SECRET_MARKER not in response.text
        # Two different refusals, both correct, and it is worth knowing which
        # is which: a value carrying "/" never matches the {base_name}
        # converter and is refused by the router ("Not Found"); a backslash or
        # UNC form does reach the handler and simply misses the index
        # ("Episode not found"). Neither ever opens a file.
        assert response.json()["detail"] in {"Not Found", "Episode not found"}
        expected = "Not Found" if "/" in hostile else "Episode not found"
        assert response.json()["detail"] == expected

    def test_null_byte_in_base_name(self, client):
        """Either the client refuses to send it or the server refuses to serve it.

        httpx rejects some control characters before they reach the app; that
        is a fine outcome, but it must not be mistaken for the app accepting
        them, so both branches are spelled out.
        """
        try:
            response = client.get("/api/episodes/episode%001/artifacts/transcript")
        except Exception:
            return  # refused client-side: nothing reached the server
        assert 400 <= response.status_code < 500, response.status_code
        assert SECRET_MARKER not in response.text

    def test_a_symlink_out_of_the_directory_is_not_served(self, client, podcast_dir, tmp_path):
        """The index is re-checked against the podcast directory before opening.

        A path can be inside the directory and still resolve outside it.
        ``_safe_file`` calls ``realpath`` on both sides for exactly this.
        """
        target = tmp_path / "outside" / "secret.txt"
        link = podcast_dir / "episode_10_summary.txt"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError, AttributeError):
            pytest.skip("creating symlinks requires privileges on this machine")

        client.post("/api/rescan")
        response = client.get("/api/episodes/episode_10/artifacts/summary")
        assert response.status_code == 404
        assert SECRET_MARKER not in response.text

    def test_the_static_mount_does_not_escape(self, client):
        response = client.get("/static/..%2f..%2foutside%2fsecret.txt")
        assert 400 <= response.status_code < 500
        assert SECRET_MARKER not in response.text


# ---------------------------------------------------------------------------
# Binding and host guard  (AC #5)
# ---------------------------------------------------------------------------


class TestBindingDefaults:
    """AC #5 / ADR-008 Must: "The web server must bind to 127.0.0.1 by default".

    Asserted against the configuration, not against a live socket: binding a
    port in a test proves nothing about the default and makes the suite
    dependent on a free port.
    """

    def test_default_host_constant_is_loopback(self):
        assert app_module.DEFAULT_HOST == "127.0.0.1"
        assert app_module.is_loopback_host(app_module.DEFAULT_HOST)

    def test_host_from_env_defaults_to_loopback(self, monkeypatch):
        monkeypatch.delenv("WHYCAST_WEBUI_HOST", raising=False)
        assert app_module.host_from_env() == "127.0.0.1"

    def test_empty_host_variable_falls_back_to_loopback(self, monkeypatch):
        monkeypatch.setenv("WHYCAST_WEBUI_HOST", "")
        assert app_module.host_from_env() == "127.0.0.1"

    def test_the_cli_does_not_supply_its_own_default_host(self):
        """The flag defaults to None so the env/default answer stays the one answer."""
        from webui.__main__ import build_parser

        assert build_parser().get_default("host") is None

    def test_widening_the_bind_is_a_deliberate_act(self, monkeypatch):
        monkeypatch.setenv("WHYCAST_WEBUI_HOST", "0.0.0.0")
        assert app_module.host_from_env() == "0.0.0.0"
        assert not app_module.is_loopback_host("0.0.0.0")

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", ""])
    def test_loopback_detection(self, host):
        assert app_module.is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com"])
    def test_non_loopback_detection(self, host):
        assert not app_module.is_loopback_host(host)

    def test_default_port_is_in_range(self, monkeypatch):
        monkeypatch.delenv("WHYCAST_WEBUI_PORT", raising=False)
        assert 1 <= app_module.port_from_env() <= 65535
        assert app_module.port_from_env() == app_module.DEFAULT_PORT

    def test_a_bad_port_raises_rather_than_exits(self, monkeypatch):
        from whycast.errors import ConfigurationError

        monkeypatch.setenv("WHYCAST_WEBUI_PORT", "not-a-port")
        with pytest.raises(ConfigurationError):
            app_module.port_from_env()

    def test_loopback_bind_keeps_a_host_allowlist(self, monkeypatch):
        monkeypatch.delenv("WHYCAST_WEBUI_ALLOWED_HOSTS", raising=False)
        allowed = app_module.allowed_hosts_from_env("127.0.0.1")
        assert "*" not in allowed
        assert "127.0.0.1" in allowed and "localhost" in allowed

    def test_a_foreign_host_header_is_rejected(self, client):
        """DNS rebinding: a bind address alone does not stop it."""
        response = client.get("/", headers={"host": "evil.example.com"})
        assert response.status_code == 421

    def test_cross_origin_writes_are_refused(self, client):
        response = client.post(
            "/api/rescan", headers={"origin": "https://evil.example.com"}
        )
        assert response.status_code == 403

    def test_same_origin_writes_are_allowed(self, client):
        response = client.post("/api/rescan", headers={"origin": "http://testserver"})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_reports_allowlisted_metadata_only(self, client):
        payload = client.get("/api/health").json()
        assert payload["status"] == "ok"
        assert set(payload["meta"]) <= set(app_module.META_KEYS)
        assert payload["meta"]["schema_version"] == db_module.SCHEMA_VERSION
        assert payload["meta"]["artifact_kinds"]


# ---------------------------------------------------------------------------
# The path guard itself
# ---------------------------------------------------------------------------


class TestSafeFileGuard:
    """Direct tests of :func:`webui.app._safe_file`.

    This is the only gate between an index row and :func:`open`, and the
    symlink test above skips on machines without the privilege to create one.
    Calling the function directly checks the same rule without needing it.
    """

    def test_a_file_inside_the_root_is_returned_resolved(self, podcast_dir):
        target = str(podcast_dir / "episode_1_summary.txt")
        result = app_module._safe_file(str(podcast_dir), target)
        assert result is not None
        assert os.path.normcase(result) == os.path.normcase(os.path.realpath(target))

    def test_a_file_outside_the_root_is_refused(self, podcast_dir, tmp_path):
        outside = str(tmp_path / "outside" / "secret.txt")
        assert os.path.isfile(outside)
        assert app_module._safe_file(str(podcast_dir), outside) is None

    def test_a_traversal_out_of_the_root_is_refused(self, podcast_dir, tmp_path):
        escape = os.path.join(str(podcast_dir), "..", "outside", "secret.txt")
        assert os.path.isfile(escape)  # the path really does reach the secret
        assert app_module._safe_file(str(podcast_dir), escape) is None

    def test_the_root_itself_is_not_a_file_to_serve(self, podcast_dir):
        assert app_module._safe_file(str(podcast_dir), str(podcast_dir)) is None

    def test_a_directory_inside_the_root_is_refused(self, podcast_dir):
        (podcast_dir / "subdir").mkdir()
        assert app_module._safe_file(str(podcast_dir), str(podcast_dir / "subdir")) is None

    def test_a_sibling_directory_with_the_same_prefix_is_refused(self, tmp_path):
        """``podcasts_evil`` must not pass a check meant for ``podcasts``.

        A prefix comparison without the trailing separator would let it
        through; this is why the root is normalised to exactly one ``os.sep``.
        """
        root = tmp_path / "podcasts"
        root.mkdir(exist_ok=True)
        evil = tmp_path / "podcasts_evil"
        evil.mkdir()
        write(evil / "secret.txt", SECRET_MARKER)
        assert app_module._safe_file(str(root), str(evil / "secret.txt")) is None

    def test_a_missing_file_is_refused(self, podcast_dir):
        candidate = str(podcast_dir / "not-there.txt")
        assert app_module._safe_file(str(podcast_dir), candidate) is None

    @pytest.mark.parametrize("candidate", [None, ""])
    def test_no_candidate_is_refused(self, podcast_dir, candidate):
        assert app_module._safe_file(str(podcast_dir), candidate) is None

    def test_a_different_drive_does_not_raise(self, podcast_dir):
        """``commonpath`` raises across drives; the prefix test must not.

        Live case on this machine: podcasts/ is on D:, %TEMP% on C:.
        """
        other_drive = r"Z:\definitely\not\here.txt" if os.name == "nt" else "/etc/hosts"
        assert app_module._safe_file(str(podcast_dir), other_drive) is None


# ---------------------------------------------------------------------------
# Degraded configuration
# ---------------------------------------------------------------------------


class TestMissingPodcastDirectory:
    """A missing directory must not stop the server from booting.

    The operator's next move is to look at the UI; an empty page plus a logged
    warning explains more than a stack trace on the console.
    """

    @pytest.fixture
    def broken_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(tmp_path / "does-not-exist"))
        monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index.db"))
        monkeypatch.delenv("WHYCAST_WEBUI_ALLOWED_HOSTS", raising=False)
        monkeypatch.delenv("WHYCAST_WEBUI_HOST", raising=False)

    def test_the_app_still_boots_and_serves_an_empty_overview(self, broken_env):
        with TestClient(app_module.create_app()) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert client.get("/api/episodes").json()["count"] == 0

    def test_rescan_reports_the_failure_instead_of_hiding_it(self, broken_env):
        with TestClient(app_module.create_app()) as client:
            response = client.post("/api/rescan")
            assert response.status_code == 500
            assert response.json()["detail"]

    def test_an_empty_directory_is_not_an_error(self, monkeypatch, tmp_path):
        empty = tmp_path / "empty-podcasts"
        empty.mkdir()
        monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(empty))
        monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index.db"))
        monkeypatch.delenv("WHYCAST_WEBUI_ALLOWED_HOSTS", raising=False)
        with TestClient(app_module.create_app()) as client:
            assert client.get("/").status_code == 200
            assert client.post("/api/rescan").json()["rescan"]["episodes"] == 0
            assert client.get("/unmatched").status_code == 200


class TestSchemaVersioning:
    """A schema bump must rebuild the index tables, not corrupt the database."""

    def test_a_stale_schema_version_is_rebuilt_and_refilled(
        self, configured_env, monkeypatch
    ):
        db_file = configured_env["db_file"]
        with TestClient(app_module.create_app()) as client:
            before = client.get("/api/episodes").json()["count"]
        assert before == len(EXPECTED_BASE_NAMES)

        # Pretend the file was written by an older build of the index.
        conn = db_module.init_db(str(db_file))
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
        conn.close()

        with TestClient(app_module.create_app()) as client:
            payload = client.get("/api/episodes").json()
            assert payload["count"] == before
            assert client.get("/api/health").json()["meta"]["schema_version"] == (
                db_module.SCHEMA_VERSION
            )

    def test_a_foreign_table_in_the_same_database_survives_a_rebuild(
        self, configured_env
    ):
        """The phase-2 job queue will share this file and is NOT rebuildable."""
        db_file = configured_env["db_file"]
        conn = db_module.init_db(str(db_file))
        conn.execute("CREATE TABLE jobs (id INTEGER PRIMARY KEY, payload TEXT)")
        conn.execute("INSERT INTO jobs (payload) VALUES ('do not lose me')")
        conn.execute("UPDATE meta SET value = '0' WHERE key = 'schema_version'")
        conn.close()

        with TestClient(app_module.create_app()) as client:
            assert client.get("/api/episodes").json()["count"] == len(
                EXPECTED_BASE_NAMES
            )
            rows = client.app.state.conn.execute("SELECT payload FROM jobs").fetchall()
        assert [tuple(r) for r in rows] == [("do not lose me",)]


# ---------------------------------------------------------------------------
# The actual bind, through the entry point  (AC #5)
# ---------------------------------------------------------------------------


class TestServerBind:
    """AC #5 through ``python -m webui``, not just through a constant.

    ``uvicorn.run`` is replaced with a recorder: the test is about which host
    the entry point asks for, and binding a real socket would prove nothing
    extra while making the suite depend on a free port.
    """

    @pytest.fixture
    def recorded_run(self, monkeypatch, tmp_path):
        import uvicorn

        calls = []
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
        monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(tmp_path))
        monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index.db"))
        monkeypatch.delenv("WHYCAST_WEBUI_HOST", raising=False)
        monkeypatch.delenv("WHYCAST_WEBUI_PORT", raising=False)
        yield calls
        # ``main`` writes the flags into os.environ by design, and
        # ``monkeypatch.delenv(..., raising=False)`` on a variable that was not
        # set records *no* undo - so a value main() writes afterwards survives
        # the test. That is how ``--port 70000`` escaped into the rest of the
        # session and decided the outcome of a test in another file.
        for name in ("WHYCAST_WEBUI_HOST", "WHYCAST_WEBUI_PORT"):
            os.environ.pop(name, None)

    def test_no_flags_binds_loopback(self, recorded_run):
        from webui.__main__ import main

        assert main([]) == 0
        (_args, kwargs) = recorded_run[0]
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == app_module.DEFAULT_PORT

    def test_the_environment_can_widen_the_bind(self, recorded_run, monkeypatch):
        from webui.__main__ import main

        monkeypatch.setenv("WHYCAST_WEBUI_HOST", "0.0.0.0")
        assert main([]) == 0
        assert recorded_run[0][1]["host"] == "0.0.0.0"

    def test_a_flag_and_the_variable_end_in_the_same_state(
        self, recorded_run, monkeypatch
    ):
        """The flags are a convenience over ADR-007 variables, not a second store."""
        from webui.__main__ import main

        # ``main`` writes the flags into os.environ on purpose (a --reload child
        # re-imports the app and has to see them), which is exactly what this
        # test asserts. Touching the variables through monkeypatch first records
        # their previous state, so the write is undone at teardown instead of
        # leaking "0.0.0.0" into every test that runs after this file.
        monkeypatch.setenv("WHYCAST_WEBUI_HOST", "127.0.0.1")
        monkeypatch.setenv("WHYCAST_WEBUI_PORT", str(app_module.DEFAULT_PORT))

        main(["--host", "0.0.0.0", "--port", "9000"])
        assert recorded_run[0][1]["host"] == "0.0.0.0"
        assert recorded_run[0][1]["port"] == 9000
        assert os.environ["WHYCAST_WEBUI_HOST"] == "0.0.0.0"
        assert os.environ["WHYCAST_WEBUI_PORT"] == "9000"

    def test_an_out_of_range_port_is_refused_before_binding(self, recorded_run):
        from webui.__main__ import main

        with pytest.raises(SystemExit) as excinfo:
            main(["--port", "70000"])
        assert excinfo.value.code == 2
        assert recorded_run == [], "the server must not start on a bad port"


# ---------------------------------------------------------------------------
# Index freshness across a restart
# ---------------------------------------------------------------------------


class TestIndexFreshness:
    """A restart must reflect the disk, including files rewritten in place.

    ``_newest_change`` walks the entries rather than trusting the directory's
    own mtime, because NTFS updates that only when an entry is added, renamed
    or removed - not when the pipeline rewrites an artifact in place, which is
    the common case. Mtimes are set explicitly here so the test does not depend
    on filesystem timestamp granularity.
    """

    def test_a_rewritten_artifact_is_picked_up_on_restart(self, configured_env):
        podcast_dir = configured_env["podcast_dir"]
        target = podcast_dir / "episode_1_summary.txt"

        with TestClient(app_module.create_app()) as client:
            before = client.get("/api/episodes/episode_1").json()
        old_size = next(
            a["size"] for a in before["artifacts"] if a["kind"] == "summary"
        )

        # Same filename, different content: the directory listing is unchanged.
        replacement = "a considerably longer summary than the original one\n"
        write(target, replacement)
        future = os.path.getmtime(target) + 10_000
        os.utime(target, (future, future))

        with TestClient(app_module.create_app()) as client:
            after = client.get("/api/episodes/episode_1").json()
            served = client.get("/api/episodes/episode_1/artifacts/summary").text
        new_size = next(a["size"] for a in after["artifacts"] if a["kind"] == "summary")

        assert served == replacement
        assert new_size == len(replacement.encode("utf-8"))
        assert new_size != old_size, "the index still reports the old file size"

    def test_a_new_episode_appears_after_a_restart(self, configured_env):
        podcast_dir = configured_env["podcast_dir"]
        with TestClient(app_module.create_app()) as client:
            before = client.get("/api/episodes").json()["count"]

        write(podcast_dir / "episode_77.mp3", "fake-mp3")
        future = os.path.getmtime(podcast_dir / "episode_77.mp3") + 10_000
        os.utime(podcast_dir / "episode_77.mp3", (future, future))

        with TestClient(app_module.create_app()) as client:
            assert client.get("/api/episodes").json()["count"] == before + 1
            assert client.get("/api/episodes/episode_77").status_code == 200


class TestArtifactFormats:
    """Every format in the scanner's vocabulary must be servable."""

    @pytest.fixture
    def client_with_all_formats(self, configured_env, podcast_dir):
        write(podcast_dir / "episode_1_history.wiki", "== history ==\n")
        write(podcast_dir / "episode_1_cleaned.md", "# cleaned\n")
        app = app_module.create_app()
        with TestClient(app) as test_client:
            yield test_client

    @pytest.mark.parametrize(
        "kind, expected",
        [("history", "== history ==\n"), ("cleaned", "# cleaned\n")],
    )
    def test_wiki_and_markdown_are_served_as_plain_text(
        self, client_with_all_formats, kind, expected
    ):
        response = client_with_all_formats.get(
            f"/api/episodes/episode_1/artifacts/{kind}"
        )
        assert response.status_code == 200
        assert response.text == expected
        # Never text/html: a .wiki or .md file rendered as HTML would execute
        # whatever markup the LLM happened to write into it.
        assert response.headers["content-type"].startswith("text/plain")


class TestAwkwardInput:
    """Values that are legal but awkward: wildcards, non-ASCII, empty files."""

    @pytest.mark.parametrize("term", ["100%", "\\", "%", "'", '"', "a" * 300])
    def test_search_terms_with_sql_metacharacters_match_nothing(self, client, term):
        """None of these characters occurs in a base name, so none may match.

        ``%`` is the interesting one: unescaped it is the LIKE wildcard and
        would return every episode.
        """
        response = client.get("/api/episodes", params={"search": term})
        assert response.status_code == 200
        assert response.json()["episodes"] == [], (
            f"{term!r} matched something it should not"
        )

    def test_a_lone_underscore_is_literal_not_a_wildcard(self, client):
        """``_`` is the single-character LIKE wildcard, and base names are full
        of literal underscores - so this is the case that tells the two apart.

        Escaped (correct): the three names that really contain ``_`` match.
        Unescaped: ``_`` matches any single character, so all four would.
        """
        payload = client.get("/api/episodes", params={"search": "_"}).json()
        matched = [e["base_name"] for e in payload["episodes"]]
        assert matched == ["episode_1", "episode_10", "episode_42"]
        assert "Episode 28" not in matched, "'_' is being treated as a wildcard"

    @pytest.fixture
    def awkward_client(self, configured_env, podcast_dir):
        write(podcast_dir / "aflevering_63_café.mp3", "audio")
        write(podcast_dir / "aflevering_63_café_summary.txt", "café summary\n")
        (podcast_dir / "episode_10_cleaned.txt").write_bytes(b"")
        app = app_module.create_app()
        with TestClient(app) as test_client:
            yield test_client

    def test_a_non_ascii_episode_round_trips(self, awkward_client):
        listing = awkward_client.get("/api/episodes").json()
        names = [e["base_name"] for e in listing["episodes"]]
        assert "aflevering_63_café" in names

        detail = awkward_client.get("/api/episodes/aflevering_63_café")
        assert detail.status_code == 200
        assert detail.json()["number"] == 63

        content = awkward_client.get(
            "/api/episodes/aflevering_63_café/artifacts/summary"
        )
        assert content.status_code == 200
        assert content.text == "café summary\n"

    def test_a_non_ascii_episode_renders_on_the_overview_and_detail_pages(
        self, awkward_client
    ):
        assert "café" in awkward_client.get("/").text
        assert awkward_client.get("/episodes/aflevering_63_café").status_code == 200

    def test_a_zero_byte_artifact_is_served_as_an_empty_file_not_a_404(
        self, awkward_client
    ):
        """The pipeline wrote an empty file; that is a result, not a gap."""
        episode = awkward_client.get("/api/episodes/episode_10").json()
        cleaned = [a for a in episode["artifacts"] if a["kind"] == "cleaned"]
        assert cleaned and cleaned[0]["size"] == 0
        assert "cleaned" in episode["kinds"]

        response = awkward_client.get("/api/episodes/episode_10/artifacts/cleaned")
        assert response.status_code == 200
        assert response.text == ""

    def test_a_zero_byte_artifact_shows_as_present_in_the_status_matrix(
        self, awkward_client
    ):
        body = awkward_client.get("/").text
        assert 'title="cleaned: txt"' in body


class TestHostGuard:
    """The Host allowlist, which is what actually stops DNS rebinding.

    A loopback bind does not: a remote page can point its own hostname at
    127.0.0.1 and reach this app as same-origin. With no authentication in
    phase 1, this check is the only thing in the way.
    """

    def test_an_ipv6_loopback_host_header_is_accepted(self, client):
        """``--host ::1`` must keep working.

        ``TrustedHostMiddleware`` derives the hostname with
        ``host.split(":")[0]``, which turns ``[::1]:8420`` into ``[``. The
        hand-written guard uses ``request.url.hostname`` instead; this is the
        request that tells the two apart.
        """
        response = client.get("/", headers={"host": "[::1]:8420"})
        assert response.status_code == 200

    @pytest.mark.parametrize("host", ["127.0.0.1:8420", "localhost:8420", "testserver"])
    def test_loopback_names_are_accepted(self, client, host):
        assert client.get("/", headers={"host": host}).status_code == 200

    @pytest.mark.parametrize(
        "host", ["evil.example.com", "attacker.test:8420", "192.168.1.10"]
    )
    def test_foreign_names_are_rejected(self, client, host):
        response = client.get("/", headers={"host": host})
        assert response.status_code == 421

    def test_the_operator_can_add_a_name(self, monkeypatch, configured_env):
        """Needed when the UI is reached by a machine name or through a proxy."""
        monkeypatch.setenv("WHYCAST_WEBUI_ALLOWED_HOSTS", "whycast.local, testserver")
        with TestClient(app_module.create_app()) as client:
            assert client.get("/", headers={"host": "whycast.local"}).status_code == 200
            assert client.get("/", headers={"host": "other.local"}).status_code == 421

    def test_an_off_loopback_bind_disables_the_check_rather_than_guessing(
        self, monkeypatch, configured_env
    ):
        """The operator published the UI under names we cannot enumerate.

        ``python -m webui`` warns loudly about this; silently breaking every
        request with a 421 would look like a bug instead.
        """
        monkeypatch.setenv("WHYCAST_WEBUI_HOST", "0.0.0.0")
        assert app_module.allowed_hosts_from_env() == ["*"]
        with TestClient(app_module.create_app()) as client:
            assert client.get("/", headers={"host": "whatever.example"}).status_code == 200

    def test_an_explicit_allowlist_beats_an_off_loopback_bind(self, monkeypatch):
        monkeypatch.setenv("WHYCAST_WEBUI_HOST", "0.0.0.0")
        monkeypatch.setenv("WHYCAST_WEBUI_ALLOWED_HOSTS", "whycast.local")
        assert app_module.allowed_hosts_from_env() == ["whycast.local"]

    def test_a_get_is_never_blocked_by_the_origin_check(self, client):
        """Artifact links must keep working from anywhere; only writes are gated."""
        response = client.get("/", headers={"origin": "https://evil.example.com"})
        assert response.status_code == 200

    def test_a_write_without_an_origin_header_is_allowed(self, client):
        """That is curl, not a browser: no browser omits Origin on a POST."""
        assert client.post("/api/rescan").status_code == 200
