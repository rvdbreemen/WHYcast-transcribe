"""
The three human-input editors, as an HTTP contract (TASK-004 phase 3).

``tests/test_webui_editors.py`` covers the vocabulary and prompt editors as
features. This module covers all three editors - vocabulary, prompts and the
speaker mapping - as a *boundary*: what a request may name, how much of it may
arrive, what comes back out into a page, and what is still on disk afterwards
when a save is refused.

The findings pinned here were each measured against a real server:

* every body cap was bypassed by a chunked, form-encoded body, because the
  form branch called ``request.form()`` (unlimited) and returned before the
  ``len(body) > cap`` check that the code called "the check that always
  holds";
* a vocabulary value was used as an ``re.sub`` replacement *template*, so a
  saved ``{"WAIcast": "\\1"}`` killed every later transcription that contained
  the term - after the GPU had been paid for - and an ordinary Windows path
  silently expanded ``\\t`` to a tab;
* deeply nested JSON and a lone UTF-16 surrogate escaped their validators and
  became opaque 500s from the one editor whose job is to explain a bad paste;
* the prompt editor accepted control characters that its sibling validator
  refuses in a speaker name, and wrote them to a file sent to the paid API.

Isolation, and why it is not optional: ``whycast.config`` computes
``VOCABULARY_FILE`` and every ``PROMPT_*_FILE`` at import time from the
repository root, with no environment override. Every test here repoints those
module attributes at ``tmp_path``; :mod:`webui.app` reads them through the
module object per request, which is what makes the patch bite. Without it these
tests would edit the operator's live vocabulary and prompts.

NO PAID CALLS. None of these routes imports the pipeline or starts a worker.
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
from whycast import config as whycast_config  # noqa: E402
from whycast.io_utils import BACKUP_SUFFIX  # noqa: E402
from whycast.pipeline.vocabulary import apply_vocabulary_corrections  # noqa: E402

PROMPT_ATTRS = {spec["name"]: spec["config_attr"] for spec in app_module.PROMPT_SPECS}

#: A transcript with two labels, so the speaker editor has something to name.
MERGED = (
    "[SPEAKER_00] Welcome to the WHYcast, episode seven.\n"
    "\n"
    "[SPEAKER_01] Glad to be here.\n"
)

FILES = {
    "episode_7.mp3": "fake-mp3-bytes",
    "episode_7_merged.txt": MERGED,
}

#: The payload that must never come back unescaped, in any of the three editors.
XSS = "<script>alert('pwned')</script>"


@pytest.fixture
def podcast_dir(tmp_path):
    directory = tmp_path / "podcasts"
    directory.mkdir()
    for name, content in FILES.items():
        (directory / name).write_bytes(content.encode("utf-8"))
    return directory


@pytest.fixture
def inputs_dir(tmp_path, monkeypatch):
    """Point every editable input file at ``tmp_path``."""
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "prompts").mkdir()

    vocab = directory / "vocabulary.json"
    vocab.write_bytes(b'{\n  "WAIcast": "WHYcast"\n}\n')
    monkeypatch.setattr(whycast_config, "VOCABULARY_FILE", str(vocab))

    for name, attr in PROMPT_ATTRS.items():
        monkeypatch.setattr(
            whycast_config, attr, str(directory / "prompts" / f"{name}_prompt.txt")
        )
    (directory / "prompts" / "summary_prompt.txt").write_bytes(b"Summarise it.\n")
    return directory


@pytest.fixture
def client(podcast_dir, inputs_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcast_dir))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index.db"))
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(tmp_path / "jobs"))
    app = app_module.create_app(
        podcast_dir=str(podcast_dir), db_path=str(tmp_path / "index.db")
    )
    with TestClient(app) as test_client:
        yield test_client


def vocab_path():
    return Path(whycast_config.VOCABULARY_FILE)


def prompt_path(name):
    return Path(getattr(whycast_config, PROMPT_ATTRS[name]))


def map_path(podcast_dir):
    return podcast_dir / "episode_7_speakers.json"


# ---------------------------------------------------------------------------
# Body caps hold on every content type (the chunked-form bypass)
# ---------------------------------------------------------------------------


def chunked(payload: bytes):
    """An iterable body, which makes httpx send ``Transfer-Encoding: chunked``.

    That is the whole trick: no ``Content-Length`` header, so the cheap header
    check cannot see the size, and the body has to be refused while it arrives.
    """
    yield payload


#: ``(url, field prefix, cap)`` per route that reads a body. The oversized
#: payload is built inside the test, never in a parameter: pytest derives the
#: ``tmp_path`` directory name from the test id, and a half-megabyte id is not
#: a directory name any filesystem will accept.
CAPPED_ROUTES = {
    "vocabulary": ("/api/vocabulary", b"text=", app_module.MAX_EDITOR_BODY_BYTES),
    "prompt": ("/api/prompts/summary", b"text=", app_module.MAX_EDITOR_BODY_BYTES),
    "jobs": ("/api/jobs", b"type=rescan&params=", app_module.MAX_JOB_BODY_BYTES),
    "speakers": (
        "/api/episodes/episode_7/speakers",
        b"speaker:SPEAKER_00=",
        app_module.MAX_JOB_BODY_BYTES,
    ),
}


@pytest.mark.parametrize("route", sorted(CAPPED_ROUTES))
def test_a_chunked_form_body_cannot_slip_past_the_cap(client, route):
    """The bypass, closed. A form body is capped exactly like a JSON one."""
    url, prefix, cap = CAPPED_ROUTES[route]
    body = prefix + b"A" * (cap + 1024)

    response = client.post(
        url,
        content=chunked(body),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 413, (
        f"{url} accepted {len(body)} bytes past a {cap}-byte cap"
    )


@pytest.mark.parametrize("route", sorted(CAPPED_ROUTES))
def test_a_chunked_json_body_cannot_slip_past_the_cap_either(client, route):
    """The branch that already held, kept honest by the same reader."""
    url, _, cap = CAPPED_ROUTES[route]
    body = b'{"text": "' + b"A" * (cap + 1024) + b'"}'

    response = client.post(
        url, content=chunked(body), headers={"content-type": "application/json"}
    )

    assert response.status_code == 413


def test_the_oversized_prompt_never_reaches_disk(client, inputs_dir):
    """The consequence that outlived the request: a prompt sent to the paid API.

    ``read_prompt_file`` does a bare ``file.read()`` and ``process_with_openai``
    concatenates the prompt ahead of the transcript while truncating only the
    transcript - so an 8 MiB prompt is billed in full, once per chunk in the
    recursive summarisation path.
    """
    before = prompt_path("summary").read_bytes()
    body = b"text=" + b"A" * (app_module.MAX_EDITOR_BODY_BYTES + 1024)

    response = client.post(
        "/api/prompts/summary",
        content=chunked(body),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 413
    assert prompt_path("summary").read_bytes() == before, "the file must be untouched"
    assert not prompt_path("summary").with_suffix(".txt" + BACKUP_SUFFIX).exists()


def test_the_oversized_job_params_never_reach_the_database(client, tmp_path):
    """The other consequence: the ``jobs`` table shares a file with the index."""
    body = b"type=rescan&params=" + b"A" * (app_module.MAX_JOB_BODY_BYTES + 1024)

    response = client.post(
        "/api/jobs",
        content=chunked(body),
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 413
    assert client.get("/api/jobs").json()["jobs"] == []


def test_a_form_body_within_the_cap_still_works(client):
    """The control. A cap that refuses everything proves nothing."""
    response = client.post(
        "/api/vocabulary",
        data={"text": '{"WAIcast": "WHYcast", "wy2025": "WHY2025"}'},
    )

    assert response.status_code == 200, response.text
    assert json.loads(vocab_path().read_text(encoding="utf-8")) == {
        "WAIcast": "WHYcast",
        "wy2025": "WHY2025",
    }


def test_a_blank_form_field_still_arrives_as_a_blank(client):
    """``parse_qsl`` drops blanks by default; the editors need them kept.

    A blank speaker name means "leave this one as it is". Dropping it would
    make that unreachable from a browser form while it kept working over JSON.
    """
    response = client.post(
        "/api/episodes/episode_7/speakers",
        data={"speaker:SPEAKER_00": "Nancy", "speaker:SPEAKER_01": ""},
    )

    assert response.status_code == 200, response.text
    assert response.json()["speakers"] == {"SPEAKER_00": "Nancy"}, (
        "a blank name means 'leave this one as it is', not an error"
    )

    # It has to reach the validator to be treated that way. A dropped field is
    # indistinguishable from one that was never rendered - and a form carrying
    # *only* blanks must be refused with the pointer to discard, which is only
    # possible if the blanks arrive.
    only_blanks = client.post(
        "/api/episodes/episode_7/speakers",
        data={"speaker:SPEAKER_00": "", "speaker:SPEAKER_01": ""},
    )

    assert only_blanks.status_code == 400, only_blanks.text
    assert "discard" in only_blanks.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Traversal and allowlist rejection, on every editor route
# ---------------------------------------------------------------------------


HOSTILE_NAMES = [
    "../../../../etc/passwd",
    "..\\..\\..\\windows\\win.ini",
    "summary/../../secret",
    "%2e%2e%2fsummary",
    "nonexistent_prompt",
]


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_a_prompt_name_outside_the_allowlist_is_refused(client, name, inputs_dir):
    """The prompt editor's whole path-safety story: a key, never a path."""
    before = sorted(p.name for p in (inputs_dir / "prompts").iterdir())

    read = client.get(f"/api/prompts/{name}")
    write = client.post(f"/api/prompts/{name}", json={"text": "malicious\n"})

    assert read.status_code in (404, 400, 405), read.text
    assert write.status_code in (404, 400, 405), write.text
    assert sorted(p.name for p in (inputs_dir / "prompts").iterdir()) == before


def snapshot(root: Path) -> dict:
    """Every file under ``root``, mapped to its bytes.

    The SQLite index is skipped: it is a rebuildable cache (ADR-008), and its
    ``-wal`` and ``-shm`` companions change on any read at all, which would
    make this a test of SQLite rather than of path safety.
    """
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".db" not in p.name
    }


@pytest.mark.parametrize("base_name", HOSTILE_NAMES + ["episode_7/../episode_9"])
def test_a_hostile_base_name_writes_no_speaker_mapping(client, base_name, tmp_path):
    """``base_name`` is an opaque index key, never joined onto a directory."""
    before = snapshot(tmp_path)

    save = client.post(
        f"/api/episodes/{base_name}/speakers", json={"speakers": {"SPEAKER_00": "X"}}
    )
    discard = client.post(f"/api/episodes/{base_name}/speakers/discard")

    assert save.status_code in (400, 404, 405), save.text
    assert discard.status_code in (400, 404, 405), discard.text
    assert snapshot(tmp_path) == before, (
        "a refused request must leave the filesystem alone"
    )


def test_the_prompt_allowlist_is_the_whole_editable_set(client):
    """Every name the API offers resolves; nothing else does."""
    listed = {entry["name"] for entry in client.get("/api/prompts").json()["prompts"]}

    assert listed == set(app_module.PROMPT_NAMES)
    for name in listed:
        assert client.get(f"/api/prompts/{name}").status_code == 200


# ---------------------------------------------------------------------------
# Escaping: nothing a person types comes back as markup
# ---------------------------------------------------------------------------


def test_a_speaker_name_is_escaped_in_the_page(client, podcast_dir):
    """Saved first, so this cannot pass by the name having been rejected."""
    saved = client.post(
        "/api/episodes/episode_7/speakers",
        json={"speakers": {"SPEAKER_00": XSS}},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["speakers"]["SPEAKER_00"] == XSS, "stored verbatim"

    page = client.get(
        "/episodes/episode_7/speakers", headers={"accept": "text/html"}
    )

    assert page.status_code == 200
    assert XSS not in page.text, "the raw script tag must never reach the page"
    assert "&lt;script&gt;" in page.text, "it must be there, escaped"


def test_a_vocabulary_key_and_value_are_escaped_in_the_page(client):
    """The vocabulary editor renders the file into a textarea."""
    saved = client.post(
        "/api/vocabulary",
        json={"text": json.dumps({XSS: "harmless", "safe": XSS})},
    )
    assert saved.status_code == 200, saved.text

    page = client.get("/vocabulary", headers={"accept": "text/html"})

    assert page.status_code == 200
    assert "<script>alert" not in page.text
    assert "&lt;script&gt;" in page.text


def test_prompt_content_is_escaped_in_the_page(client):
    """A prompt is free-form prose, so it is the easiest place to hide markup."""
    saved = client.post(
        "/api/prompts/summary", json={"text": f"Summarise this.\n{XSS}\n"}
    )
    assert saved.status_code == 200, saved.text
    assert XSS in prompt_path("summary").read_text(encoding="utf-8")

    page = client.get("/prompts/summary", headers={"accept": "text/html"})

    assert page.status_code == 200
    assert XSS not in page.text
    assert "&lt;script&gt;" in page.text


def test_an_error_message_quoting_the_input_is_escaped(client):
    """The validators quote what was submitted; that is user input too."""
    response = client.post(
        "/api/vocabulary",
        json={"text": json.dumps({"key": {"nested": XSS}})},
        headers={"accept": "text/html"},
    )

    assert response.status_code == 400
    assert "<script>alert" not in response.text


# ---------------------------------------------------------------------------
# Invalid vocabulary: refused, with the file byte-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("{not json at all", "not valid JSON"),
        ("[1, 2, 3]", "must be a JSON object"),
        ("", "empty"),
        ('{"": "everywhere"}', "empty"),
        ('{"   ": "everywhere"}', "empty"),
        ('{"ok": 42}', "not text"),
        ('{"dup": "one", "dup": "two"}', "duplicate"),
        ('{"a": "b"' + "}" * 1, ""),  # control: this one is valid
    ],
)
def test_a_rejected_vocabulary_leaves_the_file_byte_identical(client, text, expected):
    before = vocab_path().read_bytes()

    response = client.post("/api/vocabulary", json={"text": text})

    if expected == "":
        assert response.status_code == 200, response.text
        return
    assert response.status_code == 400, response.text
    assert expected.lower() in response.json()["detail"].lower()
    assert vocab_path().read_bytes() == before, "a refused save must write nothing"
    assert not Path(str(vocab_path()) + BACKUP_SUFFIX).exists()


def test_deeply_nested_json_is_a_400_not_a_500(client):
    """``RecursionError`` is neither ``ValueError`` nor ``JSONDecodeError``.

    It used to escape the validator entirely and reach the operator as
    "Internal Server Error", from the editor that exists to tell them the line
    and column of their mistake.
    """
    before = vocab_path().read_bytes()

    response = client.post(
        "/api/vocabulary", json={"text": "[" * 60000 + "]" * 60000}
    )

    assert response.status_code == 400, response.text
    assert "nested too deeply" in response.json()["detail"]
    assert vocab_path().read_bytes() == before


def test_a_lone_surrogate_is_a_400_not_a_500(client):
    """Valid JSON source, but no UTF-8 exists for it.

    It passed every validator and failed inside ``atomic_write_text`` as a
    ``UnicodeEncodeError``, which ``_write_input_file``'s ``except OSError``
    does not catch.
    """
    before = vocab_path().read_bytes()

    response = client.post(
        "/api/vocabulary", json={"text": '{"A": "\\ud800"}'}
    )

    assert response.status_code == 400, response.text
    assert "UTF-8" in response.json()["detail"]
    assert vocab_path().read_bytes() == before
    assert [
        p.name for p in vocab_path().parent.iterdir() if p.is_file()
    ] == [vocab_path().name], "no orphan .tmp may be left behind"


def test_the_app_still_serves_after_a_rejected_save(client):
    """Whatever a bad paste does, it must not take the server with it."""
    client.post("/api/vocabulary", json={"text": "[" * 60000 + "]" * 60000})
    client.post("/api/vocabulary", json={"text": '{"A": "\\ud800"}'})

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/vocabulary").status_code == 200


# ---------------------------------------------------------------------------
# A vocabulary value is data, not a regular-expression template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "replacement",
    [r"\1", r"\g<x>", r"C:\temp", r"\\", "WHY\\ncast"],
)
def test_a_saved_vocabulary_value_cannot_break_a_later_transcription(
    client, replacement
):
    """The blocker: the value was used as an ``re.sub`` replacement template.

    ``{"WAIcast": "\\1"}`` saved with HTTP 200 and then raised "invalid group
    reference" on every later transcription containing the term - i.e. exactly
    when the entry was doing its job, and after the GPU work was already paid
    for. ``C:\\temp`` was worse: no error, just a tab in the transcript.
    """
    response = client.post(
        "/api/vocabulary", json={"text": json.dumps({"WAIcast": replacement})}
    )
    assert response.status_code == 200, response.text

    mapping = json.loads(vocab_path().read_text(encoding="utf-8"))
    out = apply_vocabulary_corrections("Welcome to the WAIcast.", mapping)

    assert out == f"Welcome to the {replacement}.", "substituted literally"
    assert "\t" not in out, "a Windows path must not expand into a tab"


def test_an_ordinary_correction_still_round_trips(client):
    """The control for the four above."""
    response = client.post(
        "/api/vocabulary", json={"text": json.dumps({"WAIcast": "WHYcast"})}
    )
    assert response.status_code == 200, response.text

    mapping = json.loads(vocab_path().read_text(encoding="utf-8"))
    assert (
        apply_vocabulary_corrections("Welcome to the WAIcast.", mapping)
        == "Welcome to the WHYcast."
    )


# ---------------------------------------------------------------------------
# Prompts are plain text, the same plain text speaker names are
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", ["\x00 nul\n", "bell \x07 here\n", "\x1b[31mred\n"])
def test_a_control_character_in_a_prompt_is_refused(client, payload):
    """The two human-input editors used to disagree about "plain text".

    The speaker editor refused a NUL in a name; the prompt editor wrote one to
    a file that is then sent verbatim to the paid OpenAI API, where it fails -
    after a GPU transcription may already have been paid for.
    """
    before = prompt_path("summary").read_bytes()

    response = client.post("/api/prompts/summary", json={"text": payload})

    assert response.status_code == 400, response.text
    assert "control character" in response.json()["detail"]
    assert prompt_path("summary").read_bytes() == before


def test_tabs_and_newlines_are_still_ordinary_prompt_text(client):
    """The control: rejecting all C0 controls would reject every prompt."""
    response = client.post(
        "/api/prompts/summary",
        json={"text": "Summarise.\n\tIndented point.\n\nAnd a blank line.\n"},
    )

    assert response.status_code == 200, response.text
    assert "\tIndented point." in prompt_path("summary").read_text(encoding="utf-8")


def test_a_control_character_in_a_speaker_name_is_still_refused(client):
    """The sibling rule the prompt editor now agrees with."""
    response = client.post(
        "/api/episodes/episode_7/speakers",
        json={"speakers": {"SPEAKER_00": "Nan\x00cy"}},
    )

    assert response.status_code == 400, response.text
    assert "control character" in response.json()["detail"]


# ---------------------------------------------------------------------------
# ADR-009: every human-input write keeps exactly one backup
# ---------------------------------------------------------------------------


def test_the_vocabulary_keeps_one_backup(client):
    backup = Path(str(vocab_path()) + BACKUP_SUFFIX)

    first = client.post("/api/vocabulary", json={"text": '{"one": "1"}'})
    assert first.status_code == 200, first.text
    assert backup.read_bytes() == b'{\n  "WAIcast": "WHYcast"\n}\n'

    second = client.post("/api/vocabulary", json={"text": '{"two": "2"}'})
    assert second.status_code == 200, second.text
    assert json.loads(backup.read_text(encoding="utf-8")) == {"one": "1"}
    assert json.loads(vocab_path().read_text(encoding="utf-8")) == {"two": "2"}


def test_a_prompt_keeps_one_backup(client):
    backup = Path(str(prompt_path("summary")) + BACKUP_SUFFIX)

    assert client.post(
        "/api/prompts/summary", json={"text": "Version one.\n"}
    ).status_code == 200
    assert backup.read_text(encoding="utf-8") == "Summarise it.\n"

    assert client.post(
        "/api/prompts/summary", json={"text": "Version two.\n"}
    ).status_code == 200
    assert backup.read_text(encoding="utf-8") == "Version one.\n"
    assert prompt_path("summary").read_text(encoding="utf-8") == "Version two.\n"


def test_a_speaker_mapping_keeps_one_backup(client, podcast_dir):
    path = map_path(podcast_dir)
    backup = Path(str(path) + BACKUP_SUFFIX)

    first = client.post(
        "/api/episodes/episode_7/speakers", json={"speakers": {"SPEAKER_00": "Nancy"}}
    )
    assert first.status_code == 200, first.text
    assert first.json()["backup_file"] is None, "nothing to back up on a first save"
    assert not backup.exists()

    second = client.post(
        "/api/episodes/episode_7/speakers", json={"speakers": {"SPEAKER_00": "Marvin"}}
    )
    assert second.status_code == 200, second.text
    assert second.json()["backup_file"] == "episode_7_speakers.json.bak"
    assert json.loads(backup.read_text(encoding="utf-8"))["speakers"] == {
        "SPEAKER_00": "Nancy"
    }
    assert json.loads(path.read_text(encoding="utf-8"))["speakers"] == {
        "SPEAKER_00": "Marvin"
    }


def test_a_new_prompt_file_needs_no_backup(client):
    """``cleanup_prompt.txt`` does not exist in this fixture. Creating it is not
    an overwrite, so there is nothing to keep."""
    assert not prompt_path("cleanup").exists()

    response = client.post("/api/prompts/cleanup", json={"text": "Clean it up.\n"})

    assert response.status_code == 200, response.text
    assert prompt_path("cleanup").read_text(encoding="utf-8") == "Clean it up.\n"
    assert not Path(str(prompt_path("cleanup")) + BACKUP_SUFFIX).exists()


# ---------------------------------------------------------------------------
# Discard says what it actually leaves behind
# ---------------------------------------------------------------------------


def test_discard_does_not_claim_the_discarded_version_is_recoverable(
    client, podcast_dir
):
    """The ``.bak`` holds the version *before* the one just discarded.

    Leaving it is right - ADR-010 does not destroy human work on the way past -
    but reporting it under ``backup_file`` alone reads as "your discarded
    mapping is recoverable", which it never is.
    """
    assert client.post(
        "/api/episodes/episode_7/speakers", json={"speakers": {"SPEAKER_00": "v1"}}
    ).status_code == 200
    assert client.post(
        "/api/episodes/episode_7/speakers", json={"speakers": {"SPEAKER_00": "v2"}}
    ).status_code == 200

    response = client.post("/api/episodes/episode_7/speakers/discard")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["discarded"] is True
    assert payload["backup_is_older_version"] is True
    assert "not an undo" in payload["detail"]
    assert not map_path(podcast_dir).exists()

    backup = Path(str(map_path(podcast_dir)) + BACKUP_SUFFIX)
    assert json.loads(backup.read_text(encoding="utf-8"))["speakers"] == {
        "SPEAKER_00": "v1"
    }, "the .bak is v1, never the discarded v2"


def test_discard_after_a_single_save_admits_there_is_no_undo(client, podcast_dir):
    assert client.post(
        "/api/episodes/episode_7/speakers", json={"speakers": {"SPEAKER_00": "only"}}
    ).status_code == 200

    payload = client.post("/api/episodes/episode_7/speakers/discard").json()

    assert payload["discarded"] is True
    assert payload["backup_file"] is None
    assert payload["backup_is_older_version"] is False
    assert "not recoverable" in payload["detail"]


# ---------------------------------------------------------------------------
# The real repository files are never touched by any of this
# ---------------------------------------------------------------------------


def test_the_real_input_files_are_untouched(client):
    """The safety net for the whole module. Paths here are all in ``tmp_path``."""
    real_vocab = REPO_ROOT / "vocabulary.json"
    before = real_vocab.read_bytes() if real_vocab.exists() else None

    client.post("/api/vocabulary", json={"text": '{"scratch": "value"}'})
    client.post("/api/prompts/summary", json={"text": "scratch\n"})

    after = real_vocab.read_bytes() if real_vocab.exists() else None
    assert after == before, "the operator's real vocabulary.json must not move"
    assert not (REPO_ROOT / ("vocabulary.json" + BACKUP_SUFFIX)).exists()
