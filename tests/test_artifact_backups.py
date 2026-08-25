"""
ADR-009's backup decision, pinned at the call sites instead of only at the
primitive.

Two halves that must both stay true, and until now neither was tested where it
actually matters:

* the artifact writers leave **no** ``.bak`` - the decision, made because a
  re-run of the archive doubled the file count in ``podcasts/``;
* ``whycast.io_utils`` still backs up **when asked** - the capability the ADR
  deliberately keeps, so a future caller gets it by passing one word.

A ``.bak`` only appears on the *second* write of a file, so every test here
writes twice. The existing golden tests write each artifact once and can
therefore never catch a regression in this.
"""

import ast
import os
import types

import pytest

from whycast.io_utils import BACKUP_SUFFIX, TEMP_SUFFIX, atomic_write_text
from whycast.pipeline.outputs import write_all_format
from whycast.pipeline.speakers import write_merged_transcript
from whycast.pipeline.transcription import write_transcript_files

MARKDOWN = "# Episode 42\n\nThe answer, at last.\n\n- one\n- two\n"
TRANSCRIPT = (
    "[SPEAKER_00] Welcome to the WHYcast.\n"
    "[SPEAKER_00] Today we talk about the campsite network.\n"
    "[SPEAKER_01] Which is, famously, mostly harmless.\n"
)

WHYCAST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "whycast"
)
WRITERS = {"atomic_write_text", "atomic_write_bytes", "atomic_writer"}


def stray(directory):
    """Everything that is neither the artifact itself nor an artifact format."""
    return sorted(
        n for n in os.listdir(directory)
        if n.endswith(BACKUP_SUFFIX) or n.endswith(TEMP_SUFFIX)
    )


# ---------------------------------------------------------------------------
# The artifact writers: no .bak, ever
# ---------------------------------------------------------------------------

def test_write_all_format_leaves_no_backup(tmp_path):
    write_all_format(MARKDOWN, "episode_42_summary", str(tmp_path))
    write_all_format(MARKDOWN + "\nSecond run.\n", "episode_42_summary", str(tmp_path))

    assert sorted(os.listdir(tmp_path)) == [
        "episode_42_summary.html", "episode_42_summary.txt", "episode_42_summary.wiki",
    ]
    assert stray(tmp_path) == []


def test_write_merged_transcript_leaves_no_backup(tmp_path):
    base = str(tmp_path / "episode_42")
    write_merged_transcript(TRANSCRIPT, base)
    write_merged_transcript(TRANSCRIPT.replace("Welcome", "Welkom"), base)

    assert os.listdir(tmp_path) == ["episode_42_merged.txt"]
    assert stray(tmp_path) == []


def test_write_transcript_files_leaves_no_backup(tmp_path):
    segments = [
        types.SimpleNamespace(start=0.0, end=2.0, text="Welcome to the WHYcast."),
        types.SimpleNamespace(start=2.0, end=5.5, text="Episode forty-two."),
    ]
    clean = str(tmp_path / "episode_42.txt")
    stamped = str(tmp_path / "episode_42_ts.txt")

    write_transcript_files(segments, clean, stamped)
    write_transcript_files(segments, clean, stamped)

    assert sorted(os.listdir(tmp_path)) == ["episode_42.txt", "episode_42_ts.txt"]
    assert stray(tmp_path) == []


def test_a_rewritten_artifact_really_is_the_new_one(tmp_path):
    """No backup does not mean no write: the overwrite still has to happen."""
    write_all_format("# first\n", "episode_42_summary", str(tmp_path))
    write_all_format("# second\n", "episode_42_summary", str(tmp_path))

    assert (tmp_path / "episode_42_summary.txt").read_text(
        encoding="utf-8") == "# second\n"


# ---------------------------------------------------------------------------
# The primitive: still backs up when asked
# ---------------------------------------------------------------------------

def test_the_primitive_still_backs_up_when_asked(tmp_path):
    """The control for every test above. If this fails they prove nothing."""
    target = tmp_path / "kept.txt"
    atomic_write_text(target, "version one\n", backup=True)
    atomic_write_text(target, "version two\n", backup=True)

    backup = tmp_path / ("kept.txt" + BACKUP_SUFFIX)
    assert backup.read_text(encoding="utf-8") == "version one\n"
    assert target.read_text(encoding="utf-8") == "version two\n"


def test_backing_up_is_still_the_default(tmp_path):
    """ADR-009 Must Not: do not change the io_utils default to backup=False."""
    target = tmp_path / "kept.txt"
    atomic_write_text(target, "version one\n")
    atomic_write_text(target, "version two\n")

    assert (tmp_path / ("kept.txt" + BACKUP_SUFFIX)).read_text(
        encoding="utf-8") == "version one\n"


# ---------------------------------------------------------------------------
# The tripwire for the next writer
# ---------------------------------------------------------------------------

def _writer_calls():
    """Every call to an io_utils writer in whycast/, as (file, line, node)."""
    found = []
    for root, _dirs, names in os.walk(WHYCAST_DIR):
        if "__pycache__" in root:
            continue
        for name in sorted(names):
            if not name.endswith(".py") or name == "io_utils.py":
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called = getattr(func, "id", None) or getattr(func, "attr", None)
                if called in WRITERS:
                    found.append((os.path.relpath(path, WHYCAST_DIR), node.lineno, node))
    return found


def test_every_artifact_writer_says_backup_false_out_loud():
    """Option A's stated risk, guarded.

    ADR-009 chose per-call-site opt-out over a changed default, and named the
    cost itself: "eight call sites must each be found and changed, and a missed
    one silently keeps writing .bak". Nothing enforced that - the ADR's
    Enforcement block has no require_pattern - so this walks the AST instead of
    trusting a grep, and a new writer that forgets the keyword fails here.

    If a caller ever *wants* a backup, that is a decision: change this test and
    ADR-009 together, in that commit, not quietly.
    """
    calls = _writer_calls()
    assert calls, "the AST scan found no writers at all - it is broken, not clean"

    offenders = []
    for relpath, lineno, node in calls:
        keywords = {kw.arg: kw.value for kw in node.keywords}
        backup = keywords.get("backup")
        explicit_false = isinstance(backup, ast.Constant) and backup.value is False
        if not explicit_false:
            offenders.append(f"{relpath}:{lineno}")

    assert offenders == [], (
        "these call sites do not pass backup=False explicitly, so they inherit "
        f"the backup default and will write .bak files: {offenders}"
    )


def test_the_writers_this_suite_covers_are_all_of_them():
    """Keeps the three writer tests above honest as the package grows."""
    modules = sorted({relpath for relpath, _, _ in _writer_calls()})
    assert modules == [
        os.path.join("pipeline", "feed.py"),
        os.path.join("pipeline", "outputs.py"),
        os.path.join("pipeline", "speakers.py"),
        os.path.join("pipeline", "transcription.py"),
    ], (
        "a module started writing artifacts; give it a no-.bak test here "
        "(feed.py's writes are audio downloads, covered in test_feed_download.py)"
    )


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_the_primitive_keeps_its_backup_capability(writer):
    """The signature half of the same promise, checked without writing a file."""
    import inspect

    from whycast import io_utils

    default = inspect.signature(getattr(io_utils, writer)).parameters["backup"].default
    assert default is True, f"{writer} lost its backup=True default (ADR-009 Must Not)"
