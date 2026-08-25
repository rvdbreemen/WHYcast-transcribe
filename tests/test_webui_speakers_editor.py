"""
Tests for the speaker mapping editor in the web UI (ADR-010, TASK-004).

Everything runs against a **synthetic** podcast directory in ``tmp_path`` and a
**temporary** SQLite index, exactly like ``tests/test_webui_api.py``: the real
``podcasts/`` and the operator's real index are never touched, and the
module-level ``webui.app.app`` object is deliberately never used.

NO PAID CALLS. ``_no_paid_calls`` replaces every OpenAI entry point in
:mod:`whycast.pipeline.speakers` with something that fails loudly, for the whole
module. Nothing in this feature is supposed to reach a model - the editor reads
and writes a JSON file - and a test that quietly spent money while proving that
would be a poor sort of proof.

What is covered, against the TASK-004 acceptance criteria:

* the editor renders every ``SPEAKER_xx`` label with context lines;
* saving writes the file, and the second save leaves a ``.bak`` (ADR-009: human
  input keeps a backup, generated artifacts do not);
* the fingerprint the editor records is the one the runner will compute - the
  single assertion this whole feature stands on, verified without reusing the
  app's own transcript-picking helper;
* a label that is not in the transcript is refused;
* a hostile ``base_name`` touches nothing outside the podcast directory;
* the stale banner appears when the transcript changes underneath a mapping;
* a malformed mapping is reported, not repaired;
* only the explicit discard route deletes a mapping, and it leaves the ``.bak``.
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

#: Written outside the podcast directory. Nothing may create, change or remove
#: a file here, whatever a request asks for.
OUTSIDE_MARKER = "OUTSIDE-THE-PODCAST-DIRECTORY-DO-NOT-TOUCH"

#: A merged transcript in the shape the pipeline actually writes: one paragraph
#: per turn, ``[SPEAKER_xx]`` at the start, blank line between. SPEAKER_UNKNOWN
#: is in there because pyannote emits it and ``\d+`` would miss it.
MERGED = (
    "[SPEAKER_01] Hi and welcome to the WHYcast episode 7. I'm Nancy.\n"
    "\n"
    "[SPEAKER_03] So Ad, what are we talking about today?\n"
    "\n"
    "[SPEAKER_UNKNOWN] We have a vacancy of the week.\n"
    "\n"
    "[SPEAKER_01] Yes, it's like we planned this.\n"
)

#: The bare transcript. Different text on purpose: if anything picks this file
#: instead of the merged one, the fingerprint test fails and says so.
BARE = "[SPEAKER_01] a different transcript entirely\n"

FILES = {
    "episode_7.mp3": "fake-mp3-bytes",
    "episode_7_merged.txt": MERGED,
    "episode_7.txt": BARE,
    # An episode with no transcript at all, for the "nothing to name" branch.
    "episode_8_summary.txt": "summary only\n",
}


def write(path: Path, content: str) -> None:
    """Write bytes, not text: Path.write_text rewrites \\n as \\r\\n here."""
    path.write_bytes(content.encode("utf-8"))


def snapshot(root: Path) -> dict:
    """Every file under ``root``, mapped to its bytes. Used to prove absence."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(autouse=True)
def _no_paid_calls(monkeypatch):
    """Make any OpenAI call in the speakers module an immediate test failure."""
    from whycast.pipeline import speakers as speakers_module

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "A test tried to call the paid OpenAI API. Nothing in the speaker "
            "editor may do that."
        )

    for name in (
        "analyze_speakers_with_o4",
        "process_with_openai",
        "speaker_assignment_fallback",
    ):
        if hasattr(speakers_module, name):
            monkeypatch.setattr(speakers_module, name, forbidden)


@pytest.fixture
def tree(tmp_path):
    """The podcast directory, plus a sibling directory that must stay untouched."""
    root = tmp_path / "podcasts"
    root.mkdir()
    for name, content in FILES.items():
        write(root / name, content)

    outside = tmp_path / "outside"
    outside.mkdir()
    write(outside / "secret.txt", OUTSIDE_MARKER)
    return {"root": root, "outside": outside, "tmp": tmp_path}


@pytest.fixture
def client(monkeypatch, tree, tmp_path):
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(tree["root"]))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index" / "test.db"))
    monkeypatch.delenv("WHYCAST_WEBUI_HOST", raising=False)
    monkeypatch.delenv("WHYCAST_WEBUI_PORT", raising=False)
    monkeypatch.delenv("WHYCAST_WEBUI_ALLOWED_HOSTS", raising=False)
    app = app_module.create_app()
    with TestClient(app) as test_client:
        yield test_client


def map_file(tree) -> Path:
    return tree["root"] / "episode_7_speakers.json"


def save(client, speakers, base="episode_7"):
    return client.post(f"/api/episodes/{base}/speakers", json={"speakers": speakers})


# ---------------------------------------------------------------------------
# The editor page
# ---------------------------------------------------------------------------


def test_editor_lists_every_label_with_context(client):
    """AC: every SPEAKER_xx label, with lines a person can recognise a voice by."""
    response = client.get("/episodes/episode_7/speakers", headers={"accept": "text/html"})
    assert response.status_code == 200, response.text
    body = response.text

    for label in ("SPEAKER_01", "SPEAKER_03", "SPEAKER_UNKNOWN"):
        assert label in body, f"{label} is missing from the editor"
        assert f'name="speaker:{label}"' in body, f"no input for {label}"

    # Context, so the label means something without opening the transcript.
    assert "So Ad, what are we talking about today?" in body
    # And it came from the merged artifact, not the bare one.
    assert "episode_7_merged.txt" in body
    assert "a different transcript entirely" not in body

    # State is visible: no mapping yet, so the next run pays a model.
    assert "The next speakers run will ask the model" in body


def test_editor_survives_an_episode_without_a_transcript(client):
    response = client.get("/episodes/episode_8/speakers", headers={"accept": "text/html"})
    assert response.status_code == 200, response.text
    assert "no transcript" in response.text.lower()


def test_a_label_cap_that_bites_says_so(client, tree):
    """A cap that drops labels silently is worse than no cap at all."""
    limit = app_module.MAX_SPEAKER_LABELS
    write(
        tree["root"] / "episode_7_merged.txt",
        "".join(f"[SPEAKER_{i:03d}] line {i}\n\n" for i in range(limit + 5)),
    )
    assert client.post("/api/rescan").status_code == 200

    state = client.get("/api/episodes/episode_7/speakers").json()
    assert len(state["labels"]) == limit
    assert state["labels_truncated"] is True

    page = client.get("/episodes/episode_7/speakers", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert f"Only the first {limit} labels are shown" in page.text


def test_the_normal_case_is_not_flagged_as_truncated(client):
    assert client.get("/api/episodes/episode_7/speakers").json()["labels_truncated"] is False


def test_unknown_episode_is_404(client):
    response = client.get("/episodes/nope/speakers", headers={"accept": "text/html"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_save_writes_the_file_and_the_second_save_keeps_a_backup(client, tree):
    """ADR-009: human input keeps a .bak. Generated artifacts do not."""
    path = map_file(tree)
    backup = Path(str(path) + ".bak")
    assert not path.exists()

    first = save(client, {"SPEAKER_01": "Nancy", "SPEAKER_03": "Ad"})
    assert first.status_code == 200, first.text
    assert first.json()["saved"] is True
    assert path.is_file()
    assert not backup.exists(), "a first save must not invent a backup"

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["speakers"] == {"SPEAKER_01": "Nancy", "SPEAKER_03": "Ad"}
    assert record["source"] == "human"
    assert record["transcript_fingerprint"]
    assert record["updated_at"]

    second = save(client, {"SPEAKER_01": "Nancy", "SPEAKER_03": "Ad Bakker"})
    assert second.status_code == 200, second.text
    assert backup.is_file(), "the second save must leave the previous version behind"
    assert json.loads(backup.read_text(encoding="utf-8"))["speakers"]["SPEAKER_03"] == "Ad"
    assert second.json()["backup_file"] == "episode_7_speakers.json.bak"


def test_saved_fingerprint_is_the_one_the_runner_will_compute(client, tree):
    """The assertion this whole feature stands on.

    If the editor hashes a different transcript than the runner reads, every
    saved mapping reads as stale the moment it is applied and the feature is
    worse than useless. So this replicates ``webui.runner._read_transcript``
    independently - candidates from ("merged", "transcript"), ordered by format
    preference, read as UTF-8 - rather than calling the app's own helper, which
    would only prove that the helper agrees with itself.
    """
    from whycast.episodes import ARTIFACT_FORMATS
    from whycast.pipeline.speakers import (
        fingerprint_transcript,
        load_speaker_map_record,
    )

    assert save(client, {"SPEAKER_01": "Nancy"}).status_code == 200

    episode = client.get("/api/episodes/episode_7").json()
    order = {fmt: i for i, fmt in enumerate(ARTIFACT_FORMATS)}
    candidates = [
        artifact
        for kind in ("merged", "transcript")
        for artifact in sorted(
            (a for a in episode["artifacts"] if a["kind"] == kind),
            key=lambda a: order.get(a["fmt"], len(order)),
        )
    ]
    assert candidates, "the fixture has no transcript to read"
    with open(candidates[0]["path"], "r", encoding="utf-8") as handle:
        runner_text = handle.read()

    runners_fingerprint = fingerprint_transcript(runner_text)

    # The load the pipeline performs: with the fingerprint, which RAISES on a
    # stale record. Getting the names back is the proof.
    record = load_speaker_map_record(
        "episode_7",
        str(tree["root"]),
        transcript_fingerprint=runners_fingerprint,
    )
    assert record is not None
    assert record["speakers"] == {"SPEAKER_01": "Nancy"}
    assert record["transcript_fingerprint"] == runners_fingerprint


def test_a_label_that_is_not_in_the_transcript_is_refused(client, tree):
    before = snapshot(tree["root"])
    response = save(client, {"SPEAKER_99": "Zaphod"})
    assert response.status_code == 400, response.text
    assert "SPEAKER_99" in response.json()["detail"]
    assert snapshot(tree["root"]) == before, "a refused save must write nothing"


def test_a_form_post_refuses_the_same_label(client, tree):
    """Both body shapes fail on the same input, not one silently ignoring it."""
    response = client.post(
        "/api/episodes/episode_7/speakers",
        data={"speaker:SPEAKER_99": "Zaphod", "not-a-speaker-field": "ignored"},
    )
    assert response.status_code == 400, response.text
    assert not map_file(tree).exists()


def test_blank_names_are_refused_with_a_pointer_to_discard(client, tree):
    response = save(client, {"SPEAKER_01": "   ", "SPEAKER_03": ""})
    assert response.status_code == 400, response.text
    assert "discard" in response.json()["detail"].lower()
    assert not map_file(tree).exists()


def test_control_characters_in_a_name_are_refused(client, tree):
    response = save(client, {"SPEAKER_01": "Nancy\x00Marvin"})
    assert response.status_code == 400, response.text
    assert not map_file(tree).exists()


def test_an_overlong_name_is_refused(client, tree):
    response = save(client, {"SPEAKER_01": "N" * 200})
    assert response.status_code == 400, response.text
    assert not map_file(tree).exists()


def test_names_are_escaped_where_they_are_rendered(client):
    """A name is text. It is never markup, on any page that shows it."""
    hostile = '<img src=x onerror="alert(1)">'
    assert save(client, {"SPEAKER_01": hostile}).status_code == 200

    page = client.get("/episodes/episode_7/speakers", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert hostile not in page.text
    assert "&lt;img src=x onerror=" in page.text


def test_saving_enqueues_nothing(client):
    """Saving is free. It must not start the paid job on the user's behalf."""
    assert save(client, {"SPEAKER_01": "Nancy"}).status_code == 200
    queue = client.get("/api/jobs").json()
    assert queue["count"] == 0, queue
    assert queue["active"] is None


def test_the_save_response_offers_the_paid_job_without_starting_it(client):
    body = save(client, {"SPEAKER_01": "Nancy"}).json()
    assert body["apply_job"]["type"] == "speakers"
    assert body["apply_job"]["cost"] is True, "the offer must carry the cost flag"
    assert body["next_run"] == "saved"


# ---------------------------------------------------------------------------
# Staleness (ADR-010 precedence 2)
# ---------------------------------------------------------------------------


def test_a_changed_transcript_makes_the_mapping_stale_and_says_so(client, tree):
    assert save(client, {"SPEAKER_01": "Nancy", "SPEAKER_03": "Ad"}).status_code == 200

    # A re-transcription: same labels, different words.
    write(
        tree["root"] / "episode_7_merged.txt",
        MERGED.replace("episode 7", "episode 7, remastered"),
    )
    assert client.post("/api/rescan").status_code == 200

    state = client.get("/api/episodes/episode_7/speakers").json()
    assert state["stale"] is True
    assert state["next_run"] == "stale"
    assert state["speakers"] == {"SPEAKER_01": "Nancy", "SPEAKER_03": "Ad"}, (
        "a stale mapping must still be readable - checking the names is the "
        "whole point of the page"
    )

    page = client.get("/episodes/episode_7/speakers", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert "written for a different transcript" in page.text
    assert "save them again" in page.text.lower()
    assert "discard" in page.text.lower()
    assert map_file(tree).is_file(), "nothing may be deleted automatically"


def test_saving_again_clears_a_stale_mapping(client, tree):
    """"Check the names" resolves to a plain re-save. No field editing required."""
    assert save(client, {"SPEAKER_01": "Nancy"}).status_code == 200
    write(tree["root"] / "episode_7_merged.txt", MERGED.replace("Nancy", "Nancy B"))
    assert client.post("/api/rescan").status_code == 200
    assert client.get("/api/episodes/episode_7/speakers").json()["stale"] is True

    assert save(client, {"SPEAKER_01": "Nancy"}).status_code == 200
    after = client.get("/api/episodes/episode_7/speakers").json()
    assert after["stale"] is False
    assert after["next_run"] == "saved"


# ---------------------------------------------------------------------------
# Malformed (ADR-010 precedence 4)
# ---------------------------------------------------------------------------


def test_a_malformed_mapping_is_reported_not_repaired(client, tree):
    write(map_file(tree), "{ this is not json")
    assert client.post("/api/rescan").status_code == 200

    state = client.get("/api/episodes/episode_7/speakers").json()
    assert state["next_run"] == "malformed"
    assert "episode_7_speakers.json" in state["mapping_error"]

    page = client.get("/episodes/episode_7/speakers", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert "cannot be read" in page.text
    assert map_file(tree).read_text(encoding="utf-8") == "{ this is not json", (
        "a malformed file must be left exactly as the person typed it"
    )


def test_a_hand_written_bare_mapping_is_honoured(client, tree):
    """The obvious thing a person writes by hand has to work."""
    write(map_file(tree), '{"SPEAKER_01": "Nancy"}\n')
    assert client.post("/api/rescan").status_code == 200

    state = client.get("/api/episodes/episode_7/speakers").json()
    assert state["next_run"] == "saved"
    assert state["speakers"] == {"SPEAKER_01": "Nancy"}
    assert state["source"] == "human"
    assert state["stale"] is False, "a file with no fingerprint applies to any transcript"


# ---------------------------------------------------------------------------
# Discard (the only route that deletes)
# ---------------------------------------------------------------------------


def test_only_the_explicit_route_deletes_the_mapping(client, tree):
    path = map_file(tree)
    assert save(client, {"SPEAKER_01": "Nancy"}).status_code == 200
    assert save(client, {"SPEAKER_01": "Nancy B"}).status_code == 200  # makes the .bak
    backup = Path(str(path) + ".bak")
    assert backup.is_file()

    # Reading, in every shape, leaves it alone.
    client.get("/episodes/episode_7/speakers", headers={"accept": "text/html"})
    client.get("/api/episodes/episode_7/speakers")
    client.get("/episodes/episode_7", headers={"accept": "text/html"})
    client.post("/api/rescan")
    assert path.is_file(), "reading the page must never delete the mapping"

    response = client.post("/api/episodes/episode_7/speakers/discard")
    assert response.status_code == 200, response.text
    assert response.json()["discarded"] is True
    assert response.json()["next_run"] == "model"
    assert not path.exists()
    assert backup.is_file(), "discard removes the mapping, not the backup"


def test_the_delete_verb_does_the_same_thing(client, tree):
    assert save(client, {"SPEAKER_01": "Nancy"}).status_code == 200
    response = client.delete("/api/episodes/episode_7/speakers")
    assert response.status_code == 200, response.text
    assert response.json()["discarded"] is True
    assert not map_file(tree).exists()


def test_discarding_nothing_is_honest_about_it(client, tree):
    response = client.post("/api/episodes/episode_7/speakers/discard")
    assert response.status_code == 200, response.text
    assert response.json()["discarded"] is False
    assert "nothing changed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# The episode page
# ---------------------------------------------------------------------------


def test_the_episode_page_links_here_and_says_whether_a_mapping_exists(client):
    before = client.get("/episodes/episode_7", headers={"accept": "text/html"})
    assert before.status_code == 200
    assert "/episodes/episode_7/speakers" in before.text
    assert "No saved speaker mapping" in before.text

    assert save(client, {"SPEAKER_01": "Nancy"}).status_code == 200

    after = client.get("/episodes/episode_7", headers={"accept": "text/html"})
    assert "A saved speaker mapping exists" in after.text


# ---------------------------------------------------------------------------
# Path safety (ADR-008)
# ---------------------------------------------------------------------------

#: Names that must never reach the filesystem: traversal in three spellings,
#: an absolute path, and a Windows drive-relative one.
HOSTILE = [
    "../secret",
    "..%2Fsecret",
    "..%252Fsecret",
    "%2e%2e%2fsecret",
    "....//secret",
    "/etc/passwd",
    "C:/Windows/System32/drivers/etc/hosts",
    "episode_7/../../outside/secret",
    "\\\\server\\share\\secret",
]


@pytest.mark.parametrize("hostile", HOSTILE)
def test_a_hostile_base_name_touches_nothing(client, tree, hostile, capsys):
    """The load-bearing property is not the status code; it is that nothing moved.

    A path attempt can legitimately end up as a 404 (the route converter refuses
    a slash), a 400 (the index has no such episode) or a 307 (Starlette
    redirecting a trailing-slash form). Which one it is depends on the URL
    shape, so the assertion is about the filesystem, and the observed statuses
    are printed for the record.
    """
    before_root = snapshot(tree["root"])
    before_outside = snapshot(tree["outside"])

    statuses = {
        "GET page": client.get(
            f"/episodes/{hostile}/speakers", headers={"accept": "text/html"}
        ).status_code,
        "GET api": client.get(f"/api/episodes/{hostile}/speakers").status_code,
        "POST save": client.post(
            f"/api/episodes/{hostile}/speakers", json={"speakers": {"SPEAKER_01": "x"}}
        ).status_code,
        "POST discard": client.post(
            f"/api/episodes/{hostile}/speakers/discard"
        ).status_code,
        "DELETE": client.delete(f"/api/episodes/{hostile}/speakers").status_code,
    }
    print(f"{hostile!r}: {statuses}")

    assert snapshot(tree["root"]) == before_root
    assert snapshot(tree["outside"]) == before_outside
    assert (tree["outside"] / "secret.txt").read_text(encoding="utf-8") == OUTSIDE_MARKER
    for label, status in statuses.items():
        # A path attempt must never look like it worked, and must never be a
        # server fault either - a 500 here would mean the name reached code
        # that was not expecting it.
        assert status != 200, f"{label} answered 200 for {hostile!r}"
        assert status < 500, f"{label} answered {status} for {hostile!r}"


def test_no_response_leaks_a_path_outside_the_podcast_directory(client, tree):
    """The API reports file names, never absolute paths (see _public_speaker_state)."""
    assert save(client, {"SPEAKER_01": "Nancy"}).status_code == 200
    body = client.get("/api/episodes/episode_7/speakers").text
    assert OUTSIDE_MARKER not in body
    assert "map_path" not in body
    assert str(tree["root"]) not in body
