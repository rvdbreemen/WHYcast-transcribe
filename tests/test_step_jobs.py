"""
Single-step re-runs: one post-processing step at a time.

Editing one prompt and re-running the whole chain pays for four model calls to
see one change. These jobs run one step, read what the previous step left on
disk, and replace only their own output.

The property that matters most here is the refusal: a step whose input is not
on disk must say which job produces it, not fail with "not found". Half the
value of running one step is knowing why you cannot.

No test makes a paid call - every pipeline step is patched.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from webui import jobs as jobs_module  # noqa: E402
from webui import runner as runner_module  # noqa: E402
from webui.runner import STEP_JOB_TYPES, STEP_JOBS  # noqa: E402
from whycast.errors import WhycastError  # noqa: E402


# ---------------------------------------------------------------------------
# The catalogue, the handlers and the backup rules come from one table
# ---------------------------------------------------------------------------


def test_every_step_has_a_job_a_handler_and_a_backup_rule():
    """Three tables, one source. A step missing from one of them is a bug."""
    for job_type in STEP_JOB_TYPES:
        assert job_type in jobs_module.JOB_TYPES, f"{job_type} has no job type"
        assert job_type in runner_module._HANDLERS, f"{job_type} has no handler"
        assert job_type in runner_module._JOB_OUTPUT_KINDS, f"{job_type} has no backup rule"


def test_a_step_replaces_only_its_own_output():
    """Running one step must leave the other artifacts alone.

    This is the whole point: the summary you were happy with survives a blog
    re-run, so it is neither regenerated nor moved to the backup.
    """
    for name, spec in STEP_JOBS.items():
        moved = runner_module._JOB_OUTPUT_KINDS[f"step_{name}"]
        assert moved == frozenset(spec["writes"])
        assert len(moved) == 1, "a single step writes a single kind"


def test_every_step_costs_money_and_no_step_claims_the_gpu():
    """Each step is one model call; none of them transcribes."""
    for job_type in STEP_JOB_TYPES:
        spec = jobs_module.JOB_TYPES[job_type]
        assert spec["cost"] is True
        assert spec["gpu"] is False
        assert spec["requires_base_name"] is True


def test_the_steps_are_offered_in_pipeline_order():
    """Cleanup feeds summary feeds blog: the order is the dependency chain."""
    assert STEP_JOB_TYPES == (
        "step_cleanup",
        "step_summary",
        "step_blog",
        "step_blog_alt1",
        "step_history",
    )


def test_the_dependency_chain_is_declared_honestly():
    """What each step reads has to match what the step function needs."""
    assert STEP_JOBS["cleanup"]["needs"] == ()
    assert STEP_JOBS["summary"]["needs"] == ("cleaned",)
    assert STEP_JOBS["blog"]["needs"] == ("cleaned", "summary")
    assert STEP_JOBS["blog_alt1"]["needs"] == ("cleaned", "summary")
    assert STEP_JOBS["history"]["needs"] == ("cleaned",)


# ---------------------------------------------------------------------------
# Reading a step's input
# ---------------------------------------------------------------------------


class _Ctx:
    """The few attributes _read_artifact touches."""

    def __init__(self, output_dir):
        self.output_dir = str(output_dir)
        self.job_id = "job"
        self.job_type = "step_summary"
        self.base_name = "episode_42"
        self.params = {}


def _episode(tmp_path, **artifacts):
    entries = []
    for kind, text in artifacts.items():
        path = tmp_path / f"episode_42_{kind}.txt"
        path.write_text(text, encoding="utf-8")
        entries.append({"kind": kind, "fmt": "txt", "path": str(path)})
    return {"base_name": "episode_42", "artifacts": entries}


def test_reading_a_present_input_returns_its_text(tmp_path):
    ctx = _Ctx(tmp_path)
    episode = _episode(tmp_path, cleaned="de opgeschoonde tekst\n")

    text = runner_module._read_artifact(ctx, episode, "cleaned")

    assert text == "de opgeschoonde tekst\n"


def test_a_missing_input_names_the_job_that_makes_it(tmp_path):
    """"summary not found" leaves the reader to work out the next move."""
    ctx = _Ctx(tmp_path)
    episode = _episode(tmp_path, cleaned="er is wel een cleaned\n")

    with pytest.raises(WhycastError) as excinfo:
        runner_module._read_artifact(ctx, episode, "summary")

    message = str(excinfo.value)
    assert "summary" in message and "episode_42" in message
    assert "Summary only" in message, "it must name the job that produces it"


def test_a_missing_cleaned_points_at_cleanup(tmp_path):
    ctx = _Ctx(tmp_path)
    episode = _episode(tmp_path, summary="alleen een summary\n")

    with pytest.raises(WhycastError) as excinfo:
        runner_module._read_artifact(ctx, episode, "cleaned")

    assert "Cleanup only" in str(excinfo.value)


def test_an_empty_input_is_refused_rather_than_processed(tmp_path):
    """An empty file is not an input; sending it to a model wastes the call."""
    ctx = _Ctx(tmp_path)
    episode = _episode(tmp_path, cleaned="   \n\n")

    with pytest.raises(WhycastError) as excinfo:
        runner_module._read_artifact(ctx, episode, "cleaned")

    assert "empty" in str(excinfo.value)


def test_an_input_outside_the_podcast_directory_is_refused(tmp_path):
    """The path comes from the index, and is still re-checked against the root."""
    outside = tmp_path.parent / "elders.txt"
    outside.write_text("niet van hier\n", encoding="utf-8")
    ctx = _Ctx(tmp_path / "podcasts")
    (tmp_path / "podcasts").mkdir()
    episode = {
        "base_name": "episode_42",
        "artifacts": [{"kind": "cleaned", "fmt": "txt", "path": str(outside)}],
    }

    with pytest.raises(WhycastError):
        runner_module._read_artifact(ctx, episode, "cleaned")


# ---------------------------------------------------------------------------
# Running a step
# ---------------------------------------------------------------------------


def test_a_step_calls_only_its_own_pipeline_function(tmp_path, monkeypatch):
    """Running the blog step must not regenerate the summary as a side effect."""
    from whycast.pipeline import postprocess

    called = []
    for name in ("summary_step", "blog_step", "alt_blog_step", "history_step"):
        monkeypatch.setattr(
            postprocess, name,
            lambda *a, _n=name, **k: called.append(_n),
        )
    monkeypatch.setattr(
        postprocess, "cleanup_step",
        lambda *a, **k: called.append("cleanup_step") or "x",
    )

    ctx = _Ctx(tmp_path)
    ctx.job_type = "step_blog"
    episode = _episode(tmp_path, cleaned="schoon\n", summary="samenvatting\n")
    monkeypatch.setattr(runner_module, "_resolve_episode", lambda _ctx: episode)

    runner_module._HANDLERS["step_blog"](ctx)

    assert called == ["blog_step"]


def test_the_cleanup_step_writes_the_cleaned_transcript(tmp_path, monkeypatch):
    from whycast.pipeline import postprocess

    transcript = "[SPEAKER_00] ruwe tekst\n"
    episode = _episode(tmp_path, transcript=transcript)
    monkeypatch.setattr(
        runner_module, "_read_transcript",
        lambda _ctx: (episode, transcript, episode["artifacts"][0]["path"]),
    )
    monkeypatch.setattr(postprocess, "cleanup_step", lambda _t: "opgeschoond\n")

    ctx = _Ctx(tmp_path)
    ctx.job_type = "step_cleanup"
    runner_module._HANDLERS["step_cleanup"](ctx)

    written = tmp_path / "episode_42_cleaned.txt"
    assert written.read_text(encoding="utf-8") == "opgeschoond\n"


# ---------------------------------------------------------------------------
# Import order
# ---------------------------------------------------------------------------


def test_the_runner_imports_on_its_own(tmp_path):
    """``import webui.runner`` must work without webui.jobs being loaded first.

    Regression for a circular import: the step table lived in the runner and
    webui.jobs imported it back, so webui.jobs depended on webui.runner and
    webui.runner on webui.jobs. Importing either one first was fine; importing
    the runner *first* raised ImportError.

    ``python -m webui.runner`` - how the worker starts it - survived that by
    accident: ``-m`` loads the file as ``__main__`` and then imports the module
    a second time under its real name, which completes. So the broken import
    was invisible from the one entry point that mattered, and surfaced only
    when something did a plain import.

    Run in a subprocess with an empty module cache, because by the time this
    test runs the package is long since imported.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, r'%s');"
         "import webui.runner as r;"
         "print(len(r.STEP_JOB_TYPES))" % REPO_ROOT],
        capture_output=True, text=True, timeout=300, cwd=REPO_ROOT,
    )

    assert result.returncode == 0, (
        f"importing webui.runner first failed:\n{result.stderr[-1500:]}"
    )
    assert result.stdout.strip() == "5"


def test_the_step_table_lives_below_the_runner():
    """The table is in webui.jobs, which knows nothing about the pipeline.

    Direction matters: webui.jobs is imported by the web server, which must not
    pull in torch. Keeping the table there is what lets the runner depend on
    jobs and not the other way round.
    """
    from webui import jobs as jobs_mod

    assert hasattr(jobs_mod, "STEP_JOBS")
    assert runner_module.STEP_JOBS is jobs_mod.STEP_JOBS
