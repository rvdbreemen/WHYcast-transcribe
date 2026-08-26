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
import json
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

#: The call sites that deliberately DO keep a .bak, as (file, enclosing
#: function). Not artifacts: human input. ADR-010 made
#: ``<base>_speakers.json`` a pipeline input a person may edit, so the run that
#: rewrites it may be rewriting somebody's correction - ADR-009's "the pipeline
#: can just make it again" does not hold for typing. Adding an entry here is a
#: decision that belongs in an ADR, same as removing one.
HUMAN_INPUT_WRITERS = {
    (os.path.join("pipeline", "speakers.py"), "save_speaker_mapping"),
}


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
    """Every call to an io_utils writer in whycast/, as (file, line, node, func).

    ``func`` is the name of the function the call sits in, so the tripwire below
    can tell the artifact writers apart from the one human-input writer without
    resorting to line numbers that move.
    """
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
            owner = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for inner in ast.walk(node):
                        if isinstance(inner, ast.Call):
                            # ast.walk is breadth-first, so an enclosing def is
                            # seen before the def nested in it and this
                            # overwrites outer with inner: the innermost
                            # function that contains the call wins.
                            owner[id(inner)] = node.name
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called = getattr(func, "id", None) or getattr(func, "attr", None)
                if called in WRITERS:
                    found.append((
                        os.path.relpath(path, WHYCAST_DIR),
                        node.lineno,
                        node,
                        owner.get(id(node), "<module>"),
                    ))
    return found


def test_every_artifact_writer_says_backup_false_out_loud():
    """Option A's stated risk, guarded.

    ADR-009 chose per-call-site opt-out over a changed default, and named the
    cost itself: "eight call sites must each be found and changed, and a missed
    one silently keeps writing .bak". Nothing enforced that - the ADR's
    Enforcement block has no require_pattern - so this walks the AST instead of
    trusting a grep, and a new writer that forgets the keyword fails here.

    If a caller ever *wants* a backup, that is a decision: change this test and
    ADR-009 together, in that commit, not quietly. That has happened exactly
    once, and it is in :data:`HUMAN_INPUT_WRITERS`.
    """
    calls = _writer_calls()
    assert calls, "the AST scan found no writers at all - it is broken, not clean"

    offenders = []
    for relpath, lineno, node, func in calls:
        if (relpath, func) in HUMAN_INPUT_WRITERS:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        backup = keywords.get("backup")
        explicit_false = isinstance(backup, ast.Constant) and backup.value is False
        if not explicit_false:
            offenders.append(f"{relpath}:{lineno}")

    assert offenders == [], (
        "these call sites do not pass backup=False explicitly, so they inherit "
        f"the backup default and will write .bak files: {offenders}"
    )


def test_the_human_input_writer_says_backup_true_out_loud():
    """The other half of the same rule, for the one file that is not an artifact.

    ADR-010 made ``<base>_speakers.json`` a pipeline *input*: the pipeline reads
    it, a person edits it, and a run that overwrites it may be overwriting
    somebody's typing. That is precisely the case ADR-009's Must Not was never
    about, so it keeps its ``.bak`` - and losing that keyword quietly would be
    as much of a regression as gaining one next to an artifact.
    """
    covered = {
        (relpath, func) for relpath, _lineno, _node, func in _writer_calls()
    }
    missing = sorted(HUMAN_INPUT_WRITERS - covered)
    assert missing == [], (
        f"these human-input writers no longer write anything: {missing}. "
        "If the write moved, move the entry; if it is gone, so is the "
        "exception, and ADR-010 needs updating."
    )

    offenders = []
    for relpath, lineno, node, func in _writer_calls():
        if (relpath, func) not in HUMAN_INPUT_WRITERS:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        backup = keywords.get("backup")
        if not (isinstance(backup, ast.Constant) and backup.value is True):
            offenders.append(f"{relpath}:{lineno}")

    assert offenders == [], (
        "these call sites write human input without an explicit backup=True, "
        f"so a hand-edited file would be overwritten with no way back: {offenders}"
    )


def test_saving_a_speaker_mapping_keeps_the_previous_one(tmp_path):
    """The behaviour the keyword buys, checked by writing twice (ADR-010)."""
    from whycast.pipeline.speakers import save_speaker_mapping

    save_speaker_mapping({"SPEAKER_00": "Nancy"}, "episode_42", str(tmp_path))
    save_speaker_mapping(
        {"SPEAKER_00": "Ad"}, "episode_42", str(tmp_path), source="human"
    )

    current = tmp_path / "episode_42_speakers.json"
    backup = tmp_path / ("episode_42_speakers.json" + BACKUP_SUFFIX)

    assert json.loads(current.read_text(encoding="utf-8"))["speakers"] == {
        "SPEAKER_00": "Ad"
    }
    assert json.loads(backup.read_text(encoding="utf-8"))["speakers"] == {
        "SPEAKER_00": "Nancy"
    }, "the previous names must survive one overwrite - that is the whole point"



def test_the_cleaned_transcript_leaves_no_backup(tmp_path):
    """``<base>_cleaned.txt`` is an artifact like any other: regenerable, no .bak.

    It is the text summary, blog and history are actually written from, so it is
    worth keeping on disk - but a re-run makes it again, which is exactly the
    case ADR-009 says gets no backup and ADR-011 says gets moved aside first.
    """
    from whycast.pipeline.postprocess import _save_cleaned

    _save_cleaned("eerste opschoning\n", "ruwe tekst\n", "episode_42", str(tmp_path))
    _save_cleaned("tweede opschoning\n", "ruwe tekst\n", "episode_42", str(tmp_path))

    assert os.listdir(tmp_path) == ["episode_42_cleaned.txt"]
    assert stray(tmp_path) == []
    assert (tmp_path / "episode_42_cleaned.txt").read_text(encoding="utf-8") == (
        "tweede opschoning\n"
    )


def test_no_cleaned_transcript_when_cleanup_changed_nothing(tmp_path):
    """A file called *cleaned* that copies its input claims work nobody did.

    ``cleanup_step`` hands back its input unchanged when the prompt is missing
    or the model's answer came back too short to trust. Writing that would put a
    plausible-looking artifact on disk with nothing behind it.
    """
    from whycast.pipeline.postprocess import _save_cleaned

    assert _save_cleaned("zelfde tekst", "zelfde tekst", "episode_42", str(tmp_path)) is None
    assert os.listdir(tmp_path) == []


def test_the_writers_this_suite_covers_are_all_of_them():
    """Keeps the three writer tests above honest as the package grows.

    Read this together with :data:`HUMAN_INPUT_WRITERS`: since ADR-010,
    ``speakers.py`` appears here for two opposite reasons - its artifact writes
    (``write_merged_transcript``) must leave no ``.bak``, and its one input
    write (``save_speaker_mapping``) must keep one. A module in this list is a
    prompt to write the test that fits what it writes, not proof that a no-.bak
    test exists for every write in it.
    """
    modules = sorted({relpath for relpath, _, _, _ in _writer_calls()})
    assert modules == [
        os.path.join("pipeline", "diarization.py"),
        os.path.join("pipeline", "feed.py"),
        os.path.join("pipeline", "outputs.py"),
        os.path.join("pipeline", "postprocess.py"),
        os.path.join("pipeline", "speakers.py"),
        os.path.join("pipeline", "transcription.py"),
    ], (
        "a module started writing artifacts; give it a no-.bak test here "
        "(feed.py's writes are audio downloads, covered in test_feed_download.py; "
        "diarization.py's write is .env, covered just below)"
    )


def test_the_env_file_is_written_atomically_and_without_a_backup(tmp_path, monkeypatch):
    """``.env`` is the one file that must be atomic *and* must have no ``.bak``.

    Both halves for the same reason, pulling in opposite directions. It holds
    OPENAI_API_KEY, HUGGINGFACE_TOKEN and PODCAST_FEED_URL, so ADR-009 is right
    that there must be no ``.env.bak`` - a backup of a secrets file is a second
    copy of the secrets, lying around unnoticed. That leaves atomicity as the
    only protection the file can have, and until now it had neither: a plain
    truncating ``open(env_file, 'w')`` meant any failure between the truncate
    and the write left an empty ``.env`` and took every secret with it, while
    the function returned False and logged a line about a token.
    """
    from whycast.pipeline import diarization

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-real-key\nHUGGINGFACE_TOKEN=hf_old\n"
        "PODCAST_FEED_URL=https://example.invalid/feed.xml\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(diarization, "base_dir", str(tmp_path))
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    assert diarization.set_huggingface_token("hf_new") is True

    written = env_file.read_text(encoding="utf-8")
    assert "HUGGINGFACE_TOKEN=hf_new" in written
    assert "OPENAI_API_KEY=sk-real-key" in written, "the other secrets must survive"
    assert "PODCAST_FEED_URL=https://example.invalid/feed.xml" in written
    assert stray(tmp_path) == [], "a .env.bak would be a second copy of the secrets"

    # Written twice, because a .bak only ever appears on the second write.
    assert diarization.set_huggingface_token("hf_newer") is True
    assert stray(tmp_path) == []
    assert "HUGGINGFACE_TOKEN=hf_newer" in env_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_the_primitive_keeps_its_backup_capability(writer):
    """The signature half of the same promise, checked without writing a file."""
    import inspect

    from whycast import io_utils

    default = inspect.signature(getattr(io_utils, writer)).parameters["backup"].default
    assert default is True, f"{writer} lost its backup=True default (ADR-009 Must Not)"
