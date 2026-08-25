"""
Phase 4 (TASK-005): the before-snapshot diff and the configuration viewer.

Two properties matter most here and both are about not lying to the reader:

* a job with no snapshot must say so, not render an empty diff that reads as
  "nothing changed";
* the configuration page must never carry a secret, and the guard has to be an
  allowlist, because a denylist fails open the day someone adds a new key.
"""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from webui import snapshots  # noqa: E402
from webui import app as webui_app  # noqa: E402


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A web UI pointed at throwaway directories, never the real podcasts/."""
    podcasts = tmp_path / "podcasts"
    podcasts.mkdir()
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setenv("WHYCAST_PODCAST_DIR", str(podcasts))
    monkeypatch.setenv("WHYCAST_WEBUI_DB", str(tmp_path / "webui.db"))
    monkeypatch.setenv("WHYCAST_JOBS_DIR", str(jobs_dir))
    return {"podcasts": podcasts, "jobs_dir": jobs_dir}


@pytest.fixture()
def client(env):
    application = webui_app.create_app()
    with TestClient(application) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_a_snapshot_copies_the_artifacts_it_is_given(env):
    artifact = env["podcasts"] / "episode_1_summary.txt"
    artifact.write_text("before", encoding="utf-8")

    manifest = snapshots.take_snapshot(
        "job1", "episode_1",
        [{"kind": "summary", "fmt": "txt", "path": str(artifact)}],
    )

    assert manifest is not None
    entry = manifest["artifacts"][0]
    assert entry["copied"] is True
    copied = snapshots.snapshot_file("job1", "episode_1_summary.txt")
    assert open(copied, encoding="utf-8").read() == "before"


def test_a_snapshot_records_but_does_not_copy_a_huge_file(env, monkeypatch):
    """A 40 MB diff helps nobody, and the job log is not the place to grow one."""
    monkeypatch.setattr(snapshots, "MAX_SNAPSHOT_BYTES", 4)
    artifact = env["podcasts"] / "episode_1_transcript.txt"
    artifact.write_text("far too long for the cap", encoding="utf-8")

    manifest = snapshots.take_snapshot(
        "job2", "episode_1",
        [{"kind": "transcript", "fmt": "txt", "path": str(artifact)}],
    )

    entry = manifest["artifacts"][0]
    assert entry["copied"] is False
    assert "larger than" in entry["reason"]
    assert not os.path.exists(snapshots.snapshot_file("job2", artifact.name))


def test_a_snapshot_of_a_missing_file_is_skipped_not_fatal(env):
    manifest = snapshots.take_snapshot(
        "job3", "episode_1",
        [{"kind": "summary", "fmt": "txt", "path": str(env["podcasts"] / "gone.txt")}],
    )
    assert manifest is not None
    assert manifest["artifacts"] == []


def test_load_manifest_returns_none_without_one(env):
    assert snapshots.load_manifest("never-ran") is None


# ---------------------------------------------------------------------------
# The diff view
# ---------------------------------------------------------------------------


def _job_with_snapshot(client, env, before, after):
    """Create a job row, snapshot `before`, then leave `after` on disk."""
    from webui import jobs as jobs_module

    conn = client.app.state.conn
    job = jobs_module.enqueue(conn, "postprocess", base_name="episode_1")
    artifact = env["podcasts"] / "episode_1_summary.txt"
    artifact.write_text(before, encoding="utf-8")
    snapshots.take_snapshot(
        job["id"], "episode_1",
        [{"kind": "summary", "fmt": "txt", "path": str(artifact)}],
    )
    artifact.write_text(after, encoding="utf-8")
    return job["id"]


def test_a_job_without_a_snapshot_says_so(client):
    from webui import jobs as jobs_module

    conn = client.app.state.conn
    job = jobs_module.enqueue(conn, "selftest")

    body = client.get(f"/api/jobs/{job['id']}/diff").json()

    assert body["available"] is False
    assert "nothing to compare" in body["reason"]
    assert body["artifacts"] == []


def test_a_changed_artifact_shows_its_diff(client, env):
    job_id = _job_with_snapshot(client, env, "een regel\n", "een andere regel\n")

    body = client.get(f"/api/jobs/{job_id}/diff").json()

    assert body["available"] is True
    assert body["changed_count"] == 1
    entry = body["artifacts"][0]
    assert entry["status"] == "changed"
    assert any(line.startswith("-een regel") for line in entry["lines"])
    assert any(line.startswith("+een andere regel") for line in entry["lines"])


def test_an_untouched_artifact_reads_as_unchanged(client, env):
    job_id = _job_with_snapshot(client, env, "zelfde\n", "zelfde\n")

    body = client.get(f"/api/jobs/{job_id}/diff").json()

    assert body["changed_count"] == 0
    assert body["artifacts"][0]["status"] == "unchanged"


def test_an_artifact_deleted_by_the_run_is_reported_as_removed(client, env):
    job_id = _job_with_snapshot(client, env, "was er\n", "was er\n")
    os.remove(env["podcasts"] / "episode_1_summary.txt")

    body = client.get(f"/api/jobs/{job_id}/diff").json()

    assert body["artifacts"][0]["status"] == "removed"


def test_a_huge_diff_is_truncated_rather_than_shipped_whole(client, env, monkeypatch):
    monkeypatch.setattr(webui_app, "_MAX_DIFF_LINES", 5)
    before = "\n".join(f"regel {i}" for i in range(200))
    after = "\n".join(f"anders {i}" for i in range(200))
    job_id = _job_with_snapshot(client, env, before, after)

    entry = client.get(f"/api/jobs/{job_id}/diff").json()["artifacts"][0]

    assert entry["truncated"] is True
    assert len(entry["lines"]) == 5


def test_the_diff_page_renders(client, env):
    job_id = _job_with_snapshot(client, env, "oud\n", "nieuw\n")
    response = client.get(f"/jobs/{job_id}/diff")
    assert response.status_code == 200
    assert "episode_1" in response.text


def test_an_unknown_job_has_no_diff(client):
    assert client.get("/api/jobs/nope/diff").status_code == 404


# ---------------------------------------------------------------------------
# The configuration viewer
# ---------------------------------------------------------------------------


def test_config_shows_allowlisted_settings(client):
    body = client.get("/api/config").json()
    keys = {item["key"] for item in body["settings"]}
    assert "MODEL_SIZE" in keys
    assert "DIARIZATION_MODEL" in keys


def test_config_never_carries_a_secret(client, monkeypatch):
    """The allowlist must hold even when the environment is full of secrets."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-SHOULD-NEVER-APPEAR")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_SHOULD-NEVER-APPEAR")

    raw = client.get("/api/config").text + client.get("/config").text

    assert "SHOULD-NEVER-APPEAR" not in raw
    assert "sk-" not in raw
    assert "hf_" not in raw
    for forbidden in ("OPENAI_API_KEY", "HUGGINGFACE_TOKEN", "HF_TOKEN"):
        assert forbidden not in raw


def test_config_reports_a_new_secret_nowhere(client, monkeypatch):
    """A denylist would fail open here; the allowlist is what makes this pass.

    Simulates the future: somebody adds a token to whycast.config. Nothing in
    this page iterates the module, so it cannot appear.
    """
    from whycast import config as whycast_config

    monkeypatch.setattr(
        whycast_config, "SOME_NEW_TOKEN", "tok-LEAKED", raising=False
    )
    assert "LEAKED" not in client.get("/api/config").text


def test_config_shows_prompt_files_by_name_only(client):
    body = client.get("/api/config").json()
    names = {item["filename"] for item in body["paths"]}
    assert "vocabulary.json" in names
    # Basenames, not full paths: the directory is noise, and the page is about
    # which file is in play and whether it is there.
    for item in body["paths"]:
        assert os.sep not in item["filename"]
        assert "exists" in item


def test_config_page_renders(client):
    response = client.get("/config")
    assert response.status_code == 200
    assert "Configuration" in response.text
