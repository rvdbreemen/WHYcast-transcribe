"""
Golden extraction tests for the ADR-008 monolith -> whycast package split.

Two safety nets in one suite:

1. Old-vs-new: import BOTH the legacy ``transcribe`` monolith and the new
   ``whycast`` modules, call each pure function with identical deterministic
   inputs, and assert the results are EXACTLY equal.

2. Snapshots: the old module's output is written once to
   ``tests/golden/<func>.snapshot.txt`` and the whycast output is ALWAYS
   asserted against the stored snapshot. When ``transcribe.py`` later becomes
   a thin shim (old == new trivially), the snapshots keep pinning behavior.

Note: ``import transcribe`` performs heavy module-level imports (torch,
faster_whisper) and creates ``transcribe.log`` -- expected in this test env.
"""

import importlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_DIR = REPO_ROOT / "tests" / "golden"

# ---------------------------------------------------------------------------
# Deterministic inputs
# ---------------------------------------------------------------------------

DIARIZED_TRANSCRIPT = "\n".join([
    "[00:00:00.00 --> 00:00:04.20] [SPEAKER_00] Welcome to the WHYcast, episode forty-two.",
    "[00:00:04.20 --> 00:00:07.90] [SPEAKER_00] Today we talk about the campsite network.",
    "[00:00:07.90 --> 00:00:12.10] [SPEAKER_01] Thanks for having me, happy to be here.",
    "[00:00:12.10 --> 00:00:15.00] [SPEAKER_UNKNOWN] Absolutely, glad to join as well.",
    "[00:00:15.00 --> 00:00:19.30] [SPEAKER_01] So the fiber ring covers the whole field.",
    "[00:00:19.30 --> 00:00:23.80] [SPEAKER_00] And power comes from the generator village.",
])

# Simpler speaker-tagged text (no timestamps) exercising merge branches:
# consecutive same-speaker lines, bracketed and colon formats, continuation
# lines without a label, and empty lines.
MERGE_TRANSCRIPT = "\n".join([
    "[SPEAKER_00] Welcome to the WHYcast.",
    "[SPEAKER_00] This is episode forty-two.",
    "",
    "[SPEAKER_01] Thanks for having me.",
    "and this line has no label at all,",
    "[SPEAKER_01] It is great to be here.",
    "Robert: Let us switch to colon format.",
    "Robert: It should merge these two lines.",
    "Marvin: Life, do not talk to me about life.",
    "[SPEAKER_00] Back to brackets for the finale.",
])

UNKNOWN_TRANSCRIPT = "\n".join([
    "[SPEAKER_UNKNOWN] Who said this line?",
    "[SPEAKER_07] A numbered speaker that was never mapped.",
    "[SPEAKER_EXTRA] Some other bracketed speaker tag.",
    "Robert: An already-mapped line stays untouched.",
    "[SPEAKER_00] Another numbered one.",
])

SPEAKER_MAPPING = {
    "[SPEAKER_00]": "Robert",
    "SPEAKER_01": "Marvin",  # exercises the bracket-adding branch
}

ANALYSIS_TEXT_MAIN = "\n".join([
    "DETAILED SPEAKER ANALYSIS REPORT",
    "================================",
    "",
    "SPEAKER_00:",
    "- Speech patterns: welcomes listeners, steers the conversation",
    "- Role: host",
    "",
    "SPEAKER_01:",
    "- Speech patterns: answers technical questions in depth",
    "- Role: guest",
    "",
    "FINAL MAPPING FOR TRANSCRIPT:",
    "SPEAKER_00 → Robert",
    "SPEAKER_01 → Marvin",
    "SPEAKER_02 -> Guest (Trillian)",
    "NOT_A_SPEAKER → ShouldBeIgnored",
    "",
    "CONFIDENCE SUMMARY:",
    "SPEAKER_00: high",
    "SPEAKER_01: medium",
])

ANALYSIS_TEXT_SECTIONS = "\n".join([
    "Speaker-by-speaker review follows.",
    "",
    "SPEAKER_00:",
    "- Evidence: opens the show",
    "- Final Label: [Robert]",
    "",
    "SPEAKER_01:",
    "- Evidence: technical answers",
    "- Final Label: Marvin",
])

ANALYSIS_TEXT_LOOSE = "\n".join([
    "The analysis suggests SPEAKER_00 → Robert vd B",
    "and also SPEAKER_01 -> Ford Prefect",
    "with no formal mapping section anywhere.",
])

MARKDOWN_DOC = "\n".join([
    "# Episode 42 — The Answer",
    "",
    "**Bold statement** about the *emphasised* campsite network.",
    "",
    "## Topics",
    "",
    "- First point with [a link](https://example.com/ep42)",
    "- Second point that is **bold inside a list**",
    "- Third point",
    "",
    "1. Ordered item one",
    "2. Ordered item two",
    "",
    "### Details",
    "",
    "A paragraph with `inline code` and a bare URL: https://why2025.org",
    "",
    "Another paragraph. Mostly harmless.",
])

LONG_TEXT = "\n\n".join(
    "Paragraph {i}: ".format(i=i)
    + " ".join("word{i}x{j}".format(i=i, j=j) for j in range(40))
    + ". End of paragraph {i}.".format(i=i)
    for i in range(60)
)

VOCAB_MAPPINGS = {
    "wycast": "WHYcast",
    "why 2025": "WHY2025",
    "galactic hitchhiker": "Galactic Hitchhiker",
    "sid chip": "SID chip",
}

VOCAB_TEXT = "\n".join([
    "Welcome to the wycast, recorded at why 2025.",
    "The Wycast loves a good sid chip tune.",
    "A true galactic hitchhiker never panics.",
    "But wycasting is not a word we correct (word boundary check).",
    "WYCAST in caps is corrected case-insensitively.",
])

# Vocabulary JSON written to a temp file: includes entries the loader must
# reject (blank key, non-string value) plus valid mappings.
VOCAB_JSON_CONTENT = (
    '{\n'
    '  "teh": "the",\n'
    '  "wy2025": "WHY2025",\n'
    '  "   ": "blank-key-should-be-skipped",\n'
    '  "badvalue": 42,\n'
    '  "improbability drive": "Infinite Improbability Drive"\n'
    '}\n'
)

WORKFLOW_TRANSCRIPT = "\n".join([
    "[SPEAKER_00] Welcome to the WHYcast, this is episode forty-two.",
    "[SPEAKER_00] Today we dig into the field network and the DECT setup.",
    "[SPEAKER_01] Thanks Robert, happy to explain how the fiber ring works.",
    "[SPEAKER_00] Let us start with the backbone topology.",
    "[SPEAKER_UNKNOWN] It is a redundant ring with two uplinks.",
    "[SPEAKER_00] That sounds mostly harmless. What about power?",
    "[SPEAKER_01] Power comes from the generator village on the east side.",
    "[SPEAKER_UNKNOWN] And every datenklo has its own switch.",
    "[SPEAKER_01] Exactly, forty-two of them across the terrain.",
])

# ---------------------------------------------------------------------------
# Helpers: module access, serialization, snapshot handling
# ---------------------------------------------------------------------------

NEW_MODULE_FOR = {
    "merge_speaker_lines": "whycast.pipeline.speakers",
    "apply_speaker_mapping_programmatically": "whycast.pipeline.speakers",
    "handle_unknown_speakers": "whycast.pipeline.speakers",
    "parse_speaker_mapping_from_analysis": "whycast.pipeline.speakers",
    "split_into_chunks": "whycast.pipeline.llm",
    "estimate_token_count": "whycast.pipeline.llm",
    "truncate_transcript": "whycast.pipeline.llm",
    "format_timestamp": "whycast.pipeline.transcription",
    "convert_markdown_to_html": "whycast.pipeline.outputs",
    "convert_markdown_to_wiki": "whycast.pipeline.outputs",
    "apply_vocabulary_corrections": "whycast.pipeline.vocabulary",
    "load_vocabulary_mappings": "whycast.pipeline.vocabulary",
}


@pytest.fixture(scope="session")
def old_mod():
    """The legacy transcribe monolith, or None if it can no longer be imported."""
    try:
        return importlib.import_module("transcribe")
    except Exception:
        return None


@pytest.fixture(scope="session")
def vocab_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("vocab") / "vocabulary.json"
    path.write_text(VOCAB_JSON_CONTENT, encoding="utf-8")
    return str(path)


def new_func(name):
    module = importlib.import_module(NEW_MODULE_FOR[name])
    func = getattr(module, name, None)
    if func is None:
        pytest.fail(f"{NEW_MODULE_FOR[name]} is missing {name}")
    return func


def serialize(value):
    """Stable text form for snapshot files."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def check_golden(func_name, old_mod, invoke):
    """
    Run ``invoke(callable)`` for both implementations.

    - If the old function is available: assert old == new exactly, and create
      the snapshot from the OLD output if it does not exist yet.
    - Always assert the whycast output equals the stored snapshot.
    - Skip only when neither the old function nor a snapshot exists.
    """
    new_result = invoke(new_func(func_name))

    old_func = getattr(old_mod, func_name, None) if old_mod is not None else None
    snapshot_path = GOLDEN_DIR / f"{func_name}.snapshot.txt"

    if old_func is not None:
        old_result = invoke(old_func)
        assert old_result == new_result, (
            f"{func_name}: whycast output diverges from legacy transcribe.py"
        )
        if not snapshot_path.exists():
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(serialize(old_result), encoding="utf-8", newline="\n")

    if not snapshot_path.exists():
        pytest.skip(
            f"{func_name}: legacy function unavailable and no snapshot stored yet"
        )

    stored = snapshot_path.read_text(encoding="utf-8")
    assert serialize(new_result) == stored, (
        f"{func_name}: whycast output diverges from pinned snapshot "
        f"{snapshot_path.name}"
    )


# ---------------------------------------------------------------------------
# Level 1: pure function golden tests
# ---------------------------------------------------------------------------

def test_merge_speaker_lines(old_mod):
    check_golden(
        "merge_speaker_lines", old_mod,
        lambda f: {
            "merge_transcript": f(MERGE_TRANSCRIPT),
            "diarized_with_timestamps": f(DIARIZED_TRANSCRIPT),
            "empty": f(""),
            "whitespace_only": f("   \n  "),
        },
    )


def test_apply_speaker_mapping_programmatically(old_mod):
    check_golden(
        "apply_speaker_mapping_programmatically", old_mod,
        lambda f: {
            "diarized": f(DIARIZED_TRANSCRIPT, SPEAKER_MAPPING),
            "empty_mapping": f(DIARIZED_TRANSCRIPT, {}),
            "empty_transcript": f("", SPEAKER_MAPPING),
        },
    )


def test_handle_unknown_speakers(old_mod):
    check_golden(
        "handle_unknown_speakers", old_mod,
        lambda f: {
            "unknowns": f(UNKNOWN_TRANSCRIPT),
            "diarized": f(DIARIZED_TRANSCRIPT),
            "no_tags": f("Robert: nothing to do here."),
        },
    )


def test_parse_speaker_mapping_from_analysis(old_mod):
    check_golden(
        "parse_speaker_mapping_from_analysis", old_mod,
        lambda f: {
            "final_mapping_section": f(ANALYSIS_TEXT_MAIN),
            "per_speaker_sections": f(ANALYSIS_TEXT_SECTIONS),
            "loose_patterns": f(ANALYSIS_TEXT_LOOSE),
            "no_mapping_at_all": f("Nothing useful in this text."),
        },
    )


def test_split_into_chunks(old_mod):
    check_golden(
        "split_into_chunks", old_mod,
        lambda f: {
            "long_800_80": f(LONG_TEXT, 800, 80),
            "long_2000_200": f(LONG_TEXT, 2000, 200),
            "short_default": f("A short text."),
        },
    )


def test_estimate_token_count(old_mod):
    check_golden(
        "estimate_token_count", old_mod,
        lambda f: {
            "empty": f(""),
            "short": f("Don't panic."),
            "long": f(LONG_TEXT),
            "markdown": f(MARKDOWN_DOC),
        },
    )


def test_truncate_transcript(old_mod):
    check_golden(
        "truncate_transcript", old_mod,
        lambda f: {
            "truncated_100": f(LONG_TEXT, 100),
            "truncated_500": f(LONG_TEXT, 500),
            "untouched": f("Short enough already.", 1000),
        },
    )


def test_format_timestamp(old_mod):
    check_golden(
        "format_timestamp", old_mod,
        lambda f: {
            "zero": f(0.0),
            "minute": f(61.5),
            "hour": f(3661.257),
            "range_short": f(12.0, 34.56),
            "range_hour": f(3600.0, 3725.999),
        },
    )


def test_convert_markdown_to_html(old_mod):
    check_golden(
        "convert_markdown_to_html", old_mod,
        lambda f: {"markdown_doc": f(MARKDOWN_DOC), "empty": f("")},
    )


def test_convert_markdown_to_wiki(old_mod):
    check_golden(
        "convert_markdown_to_wiki", old_mod,
        lambda f: {"markdown_doc": f(MARKDOWN_DOC), "empty": f("")},
    )


def test_apply_vocabulary_corrections(old_mod):
    check_golden(
        "apply_vocabulary_corrections", old_mod,
        lambda f: {
            "corrections": f(VOCAB_TEXT, VOCAB_MAPPINGS),
            "empty_mappings": f(VOCAB_TEXT, {}),
        },
    )


def test_load_vocabulary_mappings(old_mod, vocab_file):
    check_golden(
        "load_vocabulary_mappings", old_mod,
        lambda f: {
            "valid_file": f(vocab_file),
            "missing_file": f(str(REPO_ROOT / "tests" / "no_such_vocab.json")),
        },
    )


# ---------------------------------------------------------------------------
# Level 2: process_transcript_workflow with a monkeypatched OpenAI layer
# ---------------------------------------------------------------------------

def fake_process_with_openai(text, prompt, model_name, max_tokens=None, **kwargs):
    """Deterministic stand-in for the OpenAI call: a tagged echo."""
    return "\n".join([
        "[FAKE-OPENAI]",
        f"MODEL: {model_name}",
        f"MAX_TOKENS: {max_tokens}",
        "PROMPT:",
        prompt,
        "TEXT:",
        text,
        "[/FAKE-OPENAI]",
    ])


class _NoNetworkOpenAI:
    """Constructor always raises, so the direct client path in
    attribute_unknown_speakers_with_ai deterministically takes its
    pattern-based except-branch — and no network call is ever possible."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("golden test: OpenAI client disabled")


_TIMESTAMP_RE = re.compile(r"Generated: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def _normalize(text):
    """Blank out the one non-deterministic bit (datetime.now in the analysis report)."""
    return _TIMESTAMP_RE.sub("Generated: <TIMESTAMP>", text)


def _read_output_dir(directory):
    out = {}
    for path in sorted(Path(directory).iterdir()):
        out[path.name] = _normalize(path.read_text(encoding="utf-8"))
    return out


def test_process_transcript_workflow_flow(old_mod, monkeypatch, tmp_path):
    if old_mod is None:
        pytest.skip("legacy transcribe module unavailable")
    if getattr(old_mod, "process_transcript_workflow", None) is None:
        pytest.skip("legacy process_transcript_workflow missing (post-shim)")

    # Prompt paths inside attribute_unknown_speakers_with_ai are cwd-relative.
    monkeypatch.chdir(REPO_ROOT)
    # Make the direct OpenAI() client in attribute_unknown_speakers_with_ai
    # fail identically in both implementations (its except-branch is the
    # deterministic pattern-based fallback). whycast.config runs load_dotenv()
    # at import time, so the key may be in os.environ even on CI-like runs;
    # delenv alone is not enough of a guarantee, hence the class patch below.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Patch the OpenAI layer in the OLD module...
    monkeypatch.setattr(old_mod, "process_with_openai", fake_process_with_openai)

    # ...and in every NEW module that bound the name at import time.
    llm = importlib.import_module("whycast.pipeline.llm")
    postprocess = importlib.import_module("whycast.pipeline.postprocess")
    speakers = importlib.import_module("whycast.pipeline.speakers")
    monkeypatch.setattr(llm, "process_with_openai", fake_process_with_openai)
    monkeypatch.setattr(postprocess, "process_with_openai", fake_process_with_openai)
    monkeypatch.setattr(speakers, "process_with_openai", fake_process_with_openai)

    # No real OpenAI client, ever: the old module does a fresh
    # ``from openai import OpenAI`` inside attribute_unknown_speakers_with_ai,
    # the new module bound whycast._deps.OpenAI at import time.
    openai_pkg = importlib.import_module("openai")
    monkeypatch.setattr(openai_pkg, "OpenAI", _NoNetworkOpenAI)
    monkeypatch.setattr(speakers, "OpenAI", _NoNetworkOpenAI)

    old_dir = tmp_path / "old_out"
    new_dir = tmp_path / "new_out"
    old_dir.mkdir()
    new_dir.mkdir()

    old_results = old_mod.process_transcript_workflow(
        WORKFLOW_TRANSCRIPT, "ep042", str(old_dir))
    new_results = postprocess.process_transcript_workflow(
        WORKFLOW_TRANSCRIPT, "ep042", str(new_dir))

    # Same result keys, same values.
    assert sorted(old_results.keys()) == sorted(new_results.keys())
    for key in old_results:
        old_val = old_results[key]
        new_val = new_results[key]
        if isinstance(old_val, str) and isinstance(new_val, str):
            assert _normalize(old_val) == _normalize(new_val), (
                f"workflow result '{key}' diverges between old and new"
            )
        else:
            assert old_val == new_val, (
                f"workflow result '{key}' diverges between old and new"
            )

    # Same produced file sets and contents.
    old_files = _read_output_dir(old_dir)
    new_files = _read_output_dir(new_dir)
    assert old_files, "workflow produced no output files at all"
    assert sorted(old_files.keys()) == sorted(new_files.keys()), (
        "workflow produced different file sets"
    )
    for name in old_files:
        assert old_files[name] == new_files[name], (
            f"workflow output file '{name}' diverges between old and new"
        )
