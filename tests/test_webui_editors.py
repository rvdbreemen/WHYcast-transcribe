"""
Tests for the vocabulary and prompt editors (ADR-008 phase 3, TASK-004).

These routes are the only ones in the web UI that *write*. Everything here is
therefore about two questions: does a save land on disk with a backup, and can
a request reach a file it was never allowed to name.

Isolation, and why it is not optional
-------------------------------------
``whycast.config`` computes ``VOCABULARY_FILE`` and every ``PROMPT_*_FILE`` at
import time from the repository root, with no environment override. A test that
did not redirect them would edit the operator's live ``vocabulary.json`` and the
real prompts - which is exactly the accident these tests exist to prevent. So:

* every test patches the ``whycast.config`` *module attributes* at a temporary
  directory. :mod:`webui.app` reads them through the module object per request
  (never a from-import), which is what makes the patch effective;
* :func:`test_real_repository_files_are_never_touched` measures the real files
  before and after the whole editor suite and fails if a byte moved.

No OpenAI call happens anywhere in here. These routes never import the
pipeline, and the two enqueue checks write a queue row and stop - no worker is
started, so nothing is ever executed or paid for.
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

#: Put in the environment as OPENAI_API_KEY. No response body may contain it.
SECRET_MARKER = "sk-TOP-SECRET-EDITOR-TEST-KEY-DO-NOT-LEAK"

#: A tiny synthetic podcast directory: one episode with audio (so the
#: vocabulary's "reprocess" picker has a target) and one without.
FILES = {
    "episode_1.mp3": "fake-mp3-bytes",
    "episode_1.txt": "bare transcript of episode one\n",
    "episode_1_summary.txt": "summary of episode one\n",
    "episode_2_summary.txt": "summary of episode two\n",
}

#: Every prompt name in the allowlist, and the config attribute behind it.
PROMPT_ATTRS = {spec["name"]: spec["config_attr"] for spec in app_module.PROMPT_SPECS}


@pytest.fixture
def podcast_dir(tmp_path):
    directory = tmp_path / "podcasts"
    directory.mkdir()
    for name, content in FILES.items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


@pytest.fixture
def inputs_dir(tmp_path, monkeypatch):
    """Redirect every editable input file into ``tmp_path``.

    Returns the directory. ``vocabulary.json`` starts with one correction;
    ``summary_prompt.txt`` with one line; the rest are absent, which is a real
    state (``blog_alt1_prompt.txt`` does not exist in this repository either).

    The prompt files are named ``<name>_prompt.txt``, which is *not* always the
    real basename - the real history prompt is ``history_extract_prompt.txt``.
    Nothing depends on the basename (it is shown, never parsed), so the tests
    stay valid; but they do not exercise the real filenames, which is what
    :func:`test_the_real_vocabulary_and_prompts_pass_their_own_validators`
    covers instead.
    """
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "prompts").mkdir()

    vocab = directory / "vocabulary.json"
    vocab.write_text('{\n  "WAIcast": "WHYcast"\n}\n', encoding="utf-8")
    monkeypatch.setattr(whycast_config, "VOCABULARY_FILE", str(vocab))

    for name, attr in PROMPT_ATTRS.items():
        target = directory / "prompts" / f"{name}_prompt.txt"
        monkeypatch.setattr(whycast_config, attr, str(target))
    (directory / "prompts" / "summary_prompt.txt").write_text(
        "Summarise the transcript.\n", encoding="utf-8"
    )
    return directory


@pytest.fixture
def client(podcast_dir, inputs_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcast_dir))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index.db"))
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(tmp_path / "jobs"))
    # A believable secret in the environment, so the leak sweep is meaningful.
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_MARKER)
    app = app_module.create_app(
        podcast_dir=str(podcast_dir), db_path=str(tmp_path / "index.db")
    )
    with TestClient(app) as test_client:
        yield test_client


def vocab_path():
    return Path(whycast_config.VOCABULARY_FILE)


def prompt_path(name):
    return Path(getattr(whycast_config, PROMPT_ATTRS[name]))


# ---------------------------------------------------------------------------
# Pages render
# ---------------------------------------------------------------------------


def test_vocabulary_page_renders(client):
    response = client.get("/vocabulary")
    assert response.status_code == 200
    body = response.text
    assert "vocabulary.json" in body
    assert "WAIcast" in body  # the current file is in the textarea
    # It must say, in words, that saving starts nothing.
    assert "No pipeline step runs" in body
    # And the re-run it offers must be marked as paid.
    assert "$ paid API" in body


def test_prompts_pages_render(client):
    listing = client.get("/prompts")
    assert listing.status_code == 200
    for name in PROMPT_ATTRS:
        assert name in listing.text or name.replace("_", " ") in listing.text

    one = client.get("/prompts/summary")
    assert one.status_code == 200
    assert "Summarise the transcript." in one.text
    assert "No pipeline step runs" in one.text
    assert "$ paid API" in one.text


def test_absent_prompt_file_renders_as_empty_and_says_the_step_is_off(client):
    assert not prompt_path("blog_alt1").exists()
    response = client.get("/prompts/blog_alt1")
    assert response.status_code == 200
    assert "does not exist yet" in response.text


# ---------------------------------------------------------------------------
# Saving: the file, and the .bak on the second save
# ---------------------------------------------------------------------------


def test_saving_vocabulary_writes_the_file_and_backs_up_the_previous_version(client):
    path = vocab_path()
    first = json.dumps({"WAIcast": "WHYcast", "Y2025": "WHY2025"})
    response = client.post("/api/vocabulary", json={"text": first})
    assert response.status_code == 200, response.text
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(first)

    # First save backs up the *original* file, which existed.
    backup = Path(str(path) + ".bak")
    assert backup.is_file()
    assert json.loads(backup.read_text(encoding="utf-8")) == {"WAIcast": "WHYcast"}

    second = json.dumps({"WAIcast": "WHYcast", "Y2025": "WHY2025", "Rai": "WHY"})
    response = client.post("/api/vocabulary", json={"text": second})
    assert response.status_code == 200, response.text
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(second)
    # The backup now holds save #1: the previous version, not the original.
    assert json.loads(backup.read_text(encoding="utf-8")) == json.loads(first)


def test_saving_a_prompt_writes_the_file_and_backs_up_the_previous_version(client):
    path = prompt_path("summary")
    backup = Path(str(path) + ".bak")

    first = client.post("/api/prompts/summary", json={"text": "Version one."})
    assert first.status_code == 200, first.text
    assert path.read_text(encoding="utf-8") == "Version one.\n"
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == "Summarise the transcript.\n"

    second = client.post("/api/prompts/summary", json={"text": "Version two."})
    assert second.status_code == 200, second.text
    assert path.read_text(encoding="utf-8") == "Version two.\n"
    assert backup.read_text(encoding="utf-8") == "Version one.\n"


def test_saving_an_absent_prompt_creates_it(client):
    path = prompt_path("blog_alt1")
    assert not path.exists()
    response = client.post("/api/prompts/blog_alt1", json={"text": "Write it again."})
    assert response.status_code == 200, response.text
    assert path.read_text(encoding="utf-8") == "Write it again.\n"


def test_a_textarea_crlf_does_not_grow_carriage_returns(client):
    """A browser posts CRLF; the file must not collect a \\r per save.

    ``atomic_write_text`` defaults to the platform newline translation, so
    without normalising the input first, ``a\\r\\nb`` would be written as
    ``a\\r\\r\\nb`` on Windows - and again on the next save.
    """
    path = prompt_path("cleanup")
    client.post("/api/prompts/cleanup", json={"text": "line one\r\nline two\r\n"})
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw == b"line one\nline two\n"

    client.post("/api/prompts/cleanup", json={"text": raw.decode() + "line three\r\n"})
    assert b"\r" not in path.read_bytes()


# ---------------------------------------------------------------------------
# Rejections: 4xx, and the file on disk is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, because",
    [
        ('{"WAIcast": "WHYcast",}', "trailing comma is not JSON"),
        ("not json at all", "not JSON"),
        ('["WAIcast", "WHYcast"]', "a list, not an object"),
        ('"just a string"', "a scalar, not an object"),
        ('{"": "WHYcast"}', "empty key matches every word boundary"),
        ('{"   ": "WHYcast"}', "whitespace-only key"),
        ('{"WAIcast": 42}', "value is not text"),
        ('{"WAIcast": null}', "value is not text"),
        ('{"WAIcast": {"nested": "object"}}', "value is not text"),
        ('{"WAIcast": "WHYcast", "WAIcast": "Something else"}', "duplicate key"),
        ("", "an empty file is not valid JSON"),
        ("   \n  ", "whitespace only"),
    ],
)
def test_invalid_vocabulary_is_rejected_and_the_file_is_unchanged(client, text, because):
    path = vocab_path()
    before = path.read_bytes()

    response = client.post("/api/vocabulary", json={"text": text})
    assert 400 <= response.status_code < 500, (
        f"{because}: expected a 4xx, got {response.status_code}"
    )
    assert response.json()["detail"], "a rejection must explain itself"

    assert path.read_bytes() == before, f"{because}: the file on disk changed"
    # A rejected save must not leave a half-written temp file behind either.
    leftovers = [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers


def test_an_empty_prompt_is_rejected_because_it_silently_disables_the_step(client):
    path = prompt_path("summary")
    before = path.read_bytes()
    response = client.post("/api/prompts/summary", json={"text": "   \n  "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    assert path.read_bytes() == before


def test_a_save_without_a_text_field_is_a_400(client):
    assert client.post("/api/vocabulary", json={}).status_code == 400
    assert client.post("/api/vocabulary", json={"text": 42}).status_code == 400
    assert client.post("/api/prompts/summary", json={"nope": "x"}).status_code == 400


def test_an_oversized_body_is_refused(client):
    big = "x" * (app_module.MAX_EDITOR_BODY_BYTES + 1)
    response = client.post("/api/prompts/summary", json={"text": big})
    assert response.status_code == 413


# ---------------------------------------------------------------------------
# The allowlist: no request parameter becomes a path
# ---------------------------------------------------------------------------


#: Names that must never resolve to a prompt.
#:
#: A *bare* ``..`` is deliberately not in this list, and the reason is worth
#: recording: httpx (and every browser) collapses ``/prompts/..`` to ``/`` in
#: the client, so that request never reaches the server at all and asserting a
#: 4xx on it would be testing the URL normaliser rather than this app. The
#: percent-encoded spellings below are *not* normalised, so they are the ones
#: that actually arrive here and actually prove something.
NOT_PROMPTS = (
    "not_a_prompt",
    "config",
    ".env",
    "%2e%2e",
    "%2e%2e%2f%2e%2e%2fconfig",
    "..%2F..%2F.env",
    "prompts",
    "vocabulary",
    "CLEANUP",  # the allowlist is exact, not case-folded
)


@pytest.mark.parametrize("name", NOT_PROMPTS)
def test_a_prompt_name_outside_the_allowlist_is_refused(client, name):
    for url in (f"/api/prompts/{name}", f"/prompts/{name}"):
        response = client.get(url)
        assert 400 <= response.status_code < 500, (
            f"{url} answered {response.status_code}"
        )
        # Belt and braces: whatever came back, it is not an editor.
        assert "editor-area" not in response.text

    response = client.post(f"/api/prompts/{name}", json={"text": "should not land"})
    assert 400 <= response.status_code < 500, response.status_code


def test_a_path_traversal_in_a_prompt_name_never_reaches_the_filesystem(client, tmp_path):
    """The classic attempt, spelled several ways, against a real target file.

    ``.env`` is written *outside* the editable directory. No spelling of the
    name may read it, and no save may create anything next to it.
    """
    outside = tmp_path / "secret.env"
    outside.write_text(f"OPENAI_API_KEY={SECRET_MARKER}\n", encoding="utf-8")
    before = sorted(p.name for p in tmp_path.iterdir())

    for name in ("../secret.env", "..%2Fsecret.env", "....//secret.env"):
        response = client.get(f"/api/prompts/{name}")
        assert 400 <= response.status_code < 500
        assert SECRET_MARKER not in response.text

        response = client.post(f"/api/prompts/{name}", json={"text": "overwritten"})
        assert 400 <= response.status_code < 500

    assert outside.read_text(encoding="utf-8") == f"OPENAI_API_KEY={SECRET_MARKER}\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_the_api_lists_only_the_allowlisted_prompts(client):
    payload = client.get("/api/prompts").json()
    assert payload["count"] == len(app_module.PROMPT_SPECS)
    assert {entry["name"] for entry in payload["prompts"]} == set(app_module.PROMPT_NAMES)


# ---------------------------------------------------------------------------
# Nothing leaks
# ---------------------------------------------------------------------------


EDITOR_URLS = (
    "/vocabulary",
    "/prompts",
    "/prompts/summary",
    "/prompts/blog_alt1",
    "/api/vocabulary",
    "/api/prompts",
    "/api/prompts/summary",
)


def test_no_editor_route_returns_an_environment_value(client):
    for url in EDITOR_URLS:
        response = client.get(url)
        assert response.status_code == 200, f"{url} -> {response.status_code}"
        assert SECRET_MARKER not in response.text, f"{url} leaked the API key"
        assert "OPENAI_API_KEY" not in response.text, f"{url} named the API key"
        assert "HUGGINGFACE" not in response.text.upper()


def test_no_editor_route_publishes_the_path_of_an_edited_file(client, inputs_dir):
    """An editable file is named relative to the repository, or by filename only.

    The input files live outside the repository in these tests, so
    :func:`webui.app._display_path` degrades them to a bare filename - and the
    directory holding them must appear in no response.

    Scoped to the *input* directory on purpose. ``podcast_dir`` and
    ``database_path`` do appear, in the page footer: they are two of
    :data:`webui.app.META_KEYS`, an allowlist that predates this work and
    discloses them deliberately. Asserting over the whole temp tree would fail
    on that pre-existing decision instead of on anything these editors do.
    """
    marker = str(inputs_dir).replace("\\", "/")
    for url in EDITOR_URLS:
        body = client.get(url).text.replace("\\", "/")
        assert marker not in body, f"{url} exposed the path of an edited file"
    # And the filenames themselves are still shown, so this is not passing by
    # virtue of showing nothing at all.
    assert "vocabulary.json" in client.get("/vocabulary").text
    assert "summary_prompt.txt" in client.get("/prompts/summary").text


# ---------------------------------------------------------------------------
# The browser path: a form post, and the offered re-run
# ---------------------------------------------------------------------------


def test_a_browser_form_save_redirects_instead_of_showing_json(client):
    response = client.post(
        "/api/vocabulary",
        data={"text": '{"WAIcast": "WHYcast"}'},
        headers={"Accept": "text/html"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/vocabulary?saved=1"


def test_a_browser_form_rejection_re_renders_the_page_with_the_text_kept(client):
    path = vocab_path()
    before = path.read_bytes()
    response = client.post(
        "/api/vocabulary",
        data={"text": '{"broken": '},
        headers={"Accept": "text/html"},
    )
    assert response.status_code == 400
    assert "Not saved" in response.text
    assert '{&#34;broken&#34;: ' in response.text or '{"broken": ' in response.text
    assert path.read_bytes() == before


def test_the_offered_re_run_is_a_real_job_type_and_is_marked_paid(client):
    """The page offers a job; the API must accept exactly that job.

    This enqueues a row and stops. No worker runs in the test suite, so nothing
    is executed and nothing is spent - which is the whole reason cost lives on
    the job *type* and not in the request.
    """
    catalogue = {entry["type"]: entry for entry in app_module.job_types_payload()}
    assert app_module.VOCABULARY_APPLY_JOB in catalogue
    assert catalogue[app_module.VOCABULARY_APPLY_JOB]["cost"] is True

    for spec in app_module.PROMPT_SPECS:
        assert spec["job_type"] in catalogue, spec["name"]
        assert catalogue[spec["job_type"]]["cost"] is True, spec["name"]

    created = client.post(
        "/api/jobs",
        data={"type": app_module.VOCABULARY_APPLY_JOB, "base_name": "episode_1"},
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["status"] == "queued"
    assert body["cost"] is True


# ---------------------------------------------------------------------------
# The safety net
# ---------------------------------------------------------------------------


def test_real_repository_files_are_never_touched(podcast_dir, inputs_dir, tmp_path, monkeypatch):
    """Run a full save cycle and prove the operator's own files did not move.

    Everything above patches ``whycast.config``; this asserts the patch is what
    makes that true. If a route ever resolved a path at import time instead of
    per request, the writes below would land on the real files and this fails.
    """
    real_vocab = REPO_ROOT / "vocabulary.json"
    real_prompt = REPO_ROOT / "prompts" / "summary_prompt.txt"
    watched = [p for p in (real_vocab, real_prompt) if p.exists()]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in watched}

    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcast_dir))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "index2.db"))
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(tmp_path / "jobs2"))
    app = app_module.create_app(
        podcast_dir=str(podcast_dir), db_path=str(tmp_path / "index2.db")
    )
    with TestClient(app) as test_client:
        assert test_client.post(
            "/api/vocabulary", json={"text": '{"only": "here"}'}
        ).status_code == 200
        assert test_client.post(
            "/api/prompts/summary", json={"text": "only here"}
        ).status_code == 200

    for path, (content, mtime) in before.items():
        assert path.read_bytes() == content, f"{path} was modified"
        assert path.stat().st_mtime_ns == mtime, f"{path} was rewritten"
    # A write would also have left a backup next to the original. Nothing here
    # may create one: these routes never touched the real files at all.
    for path in watched:
        assert not Path(str(path) + ".bak").exists(), f"{path}.bak was created"


def test_the_real_vocabulary_and_prompts_pass_their_own_validators():
    """The operator's actual files must be saveable through this UI.

    The validators are stricter than the pipeline's loader, which skips a bad
    entry with a warning rather than failing. If the real ``vocabulary.json``
    held a duplicate key or a non-string value, opening the editor, adding one
    term and pressing Save would give a 400 that cannot be cleared from the UI -
    a dead end on day one, on the main thing this page is for. So it is checked
    against the real files, not only synthetic ones.

    Deliberately does not use the fixtures: it reads the repository's own files
    and writes nothing.
    """
    from webui.app import _EditorRejected, _validated_prompt, _validated_vocabulary

    vocab = REPO_ROOT / "vocabulary.json"
    if vocab.is_file():
        mapping, _ = _validated_vocabulary(vocab.read_text(encoding="utf-8"))
        assert mapping, "the real vocabulary parsed to nothing"

    for spec in app_module.PROMPT_SPECS:
        path = Path(getattr(whycast_config, spec["config_attr"]))
        if not path.is_file():
            continue  # blog_alt1 genuinely does not exist; the step is skipped
        try:
            _validated_prompt(path.read_text(encoding="utf-8"), spec)
        except _EditorRejected as exc:  # pragma: no cover - would be a real defect
            pytest.fail(f"{path.name} cannot be saved through the editor: {exc}")
