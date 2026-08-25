"""
The speaker mapping as a pipeline input (ADR-010, TASK-004 phase 3).

``tests/test_speaker_mapping_precedence.py`` covers the four precedence
branches as a *cost* question - was the model asked. This module covers the
same four branches as a *correctness and safety* question, and then the things
that turned out to be true of them but were nobody's test:

* a mapping survives a force-reprocess, which used to delete it and its ``.bak``
  in one sweep (ADR-010's Must Not, and unrecoverable: ``podcasts/`` is
  gitignored);
* the saved-mapping path makes no OpenAI call *at all* - not merely no
  ``analyze_speakers_with_o4`` call. A second, unrelated paid call sat behind
  ``[SPEAKER_UNKNOWN]`` handling and fired on every transcript in this corpus,
  on the very path the run announces as "no model call needed";
* a mapping that names none of the transcript's labels is refused rather than
  applied to nothing;
* the unmapped-label warning ADR-010 names as its stale-mapping signal can
  actually fire.

NO PAID CALLS. Every OpenAI entry point in :mod:`whycast.pipeline.speakers` is
replaced by a tripwire that raises, including the ``OpenAI`` client class
itself: it was a direct ``OpenAI()`` construction, not a patched helper, that
made the twelve gpt-4o calls this module now pins shut.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whycast import episodes as episodes_module  # noqa: E402
from whycast.errors import SpeakerMappingError  # noqa: E402
from whycast.events import CollectSink, use_sink  # noqa: E402
from whycast.io_utils import BACKUP_SUFFIX  # noqa: E402
from whycast.pipeline import speakers as speakers_module  # noqa: E402
from whycast.pipeline.feed import delete_episode_files  # noqa: E402
from whycast.pipeline.speakers import (  # noqa: E402
    SPEAKER_MAP_SUFFIX,
    fingerprint_transcript,
    load_speaker_mapping,
    save_speaker_mapping,
    speaker_map_path,
)

BASE = "episode_42"

#: Two named speakers and one gap the diarizer could not place. The gap is
#: bracketed by the SAME speaker, so the deterministic attribution rule claims
#: it - which is what the removed gpt-4o call was pretending to decide.
TRANSCRIPT = (
    "[SPEAKER_00] Welcome to the WHYcast, episode forty-two.\n"
    "[SPEAKER_00] Today we talk about the campsite network.\n"
    "[SPEAKER_UNKNOWN] Mostly harmless, I would say.\n"
    "[SPEAKER_00] Which is exactly the point.\n"
    "[SPEAKER_01] Thanks for having me.\n"
)


class _Tripwire:
    """Stands in for a paid entry point. Being called at all is the failure."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise AssertionError(
            f"{self.name} was reached: this run would have cost money. The "
            f"saved-mapping path is advertised as free and must stay free."
        )


@pytest.fixture(autouse=True)
def no_paid_calls(monkeypatch):
    """Every route to OpenAI in the speakers module, closed.

    ``OpenAI`` is in the list on purpose and is the one that matters. Patching
    only ``analyze_speakers_with_o4`` proves the *analysis* was skipped and
    nothing else: ``attribute_unknown_speakers_with_ai`` used to construct its
    own ``OpenAI()`` client and call gpt-4o once per five ``[SPEAKER_UNKNOWN]``
    segments, on the saved-mapping path, in the same run that printed "Speaker
    analysis skipped - no model call needed".
    """
    for name in (
        "OpenAI",
        "analyze_speakers_with_o4",
        "process_with_openai",
        "speaker_assignment_fallback",
    ):
        monkeypatch.setattr(speakers_module, name, _Tripwire(name))
    monkeypatch.setattr(speakers_module, "openai_available", True)


def run_step(output_dir, transcript=TRANSCRIPT, base=BASE):
    """Run ``speaker_assignment_step`` and collect what it told the operator."""
    sink = CollectSink()
    with use_sink(sink):
        try:
            result, error = (
                speakers_module.speaker_assignment_step(
                    transcript, base, str(output_dir)
                ),
                None,
            )
        except Exception as exc:
            result, error = None, exc
    text = "\n".join(event.message for event in sink.events)
    return result, error, text, sink.events


def write_raw(path, payload):
    """Write a mapping file exactly as given - a hand edit, not a save."""
    Path(path).write_bytes(payload.encode("utf-8"))


# ---------------------------------------------------------------------------
# The four precedence branches (ADR-010's Decision Outcome)
# ---------------------------------------------------------------------------


def test_branch_1_a_saved_mapping_is_applied_and_costs_nothing(tmp_path):
    """The file is there and current: those are the names, no model, no charge."""
    save_speaker_mapping(
        {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript(TRANSCRIPT),
        source="human",
    )

    result, error, text, _ = run_step(tmp_path)

    assert error is None, error
    assert "Nancy:" in result and "Ad:" in result
    assert "[SPEAKER_00]" not in result
    assert "no model call needed" in text


def test_branch_1_makes_no_openai_call_of_any_kind(tmp_path):
    """ADR-010's Confirmation #1, taken literally: *no* OpenAI call.

    The autouse tripwire covers ``OpenAI`` itself, so the twelve gpt-4o calls
    that used to hide behind ``[SPEAKER_UNKNOWN]`` handling would fail this.
    The transcript deliberately contains such a segment; without it the test
    would pass while proving nothing.
    """
    assert "[SPEAKER_UNKNOWN]" in TRANSCRIPT, "the fixture must exercise that path"
    save_speaker_mapping(
        {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}, BASE, str(tmp_path)
    )

    result, error, _, _ = run_step(tmp_path)

    assert error is None, error
    assert "Mostly harmless" in result, "no content may be lost"
    assert "Nancy:" in result and "Ad:" in result


def test_removing_the_paid_attribution_call_changed_no_output(tmp_path):
    """Pins the outcome the removed gpt-4o call did not affect.

    Inside :func:`speaker_assignment_programmatic` the mapping is applied
    *first*, so by the time the unknown-segment pass runs there are no
    ``[SPEAKER_xx]`` tags left for it to read a neighbour from - the gap
    collapses to a generic ``Speaker:``. That is unchanged from before, and it
    is precisely why the twelve paid calls were measurable as useless: the
    attributions came from ``previous_speaker == next_speaker`` in both the
    success and the failure branch, and here neither one can match.
    """
    write_raw(
        speaker_map_path(BASE, str(tmp_path)),
        '{"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}\n',
    )

    result, error, _, _ = run_step(tmp_path)

    assert error is None, error
    assert "Speaker: Mostly harmless, I would say." in result
    assert "[SPEAKER_UNKNOWN]" not in result


def test_the_neighbour_rule_still_works_where_it_can_read_neighbours():
    """The decision procedure itself, on a transcript that still has labels.

    Same speaker either side of the gap means nobody took the floor, so the gap
    is theirs. This is the whole of what the model was being paid to confirm.
    """
    out = speakers_module.attribute_unknown_speakers(TRANSCRIPT)

    assert out.count("[SPEAKER_00]") == 4, "the gap belongs to the speaker around it"
    assert "[SPEAKER_UNKNOWN]" not in out


def test_the_neighbour_rule_declines_when_the_floor_changed_hands():
    """Different speakers either side is a real handover; guessing would be wrong.

    The gap keeps a generic ``Speaker:``. Note that a surviving unknown sends
    the whole transcript through :func:`handle_unknown_speakers`, which also
    de-brackets the labels around it - pre-existing, and the reason the
    unmapped-label check has to run before this point rather than after it.
    """
    handover = (
        "[SPEAKER_00] I will hand over now.\n"
        "[SPEAKER_UNKNOWN] Thank you very much.\n"
        "[SPEAKER_01] Right, my turn.\n"
    )

    out = speakers_module.attribute_unknown_speakers(handover)

    assert "Speaker: Thank you very much." in out, "an unclaimed gap stays generic"
    assert "Speaker 00: I will hand over now." in out
    assert "Speaker 01: Right, my turn." in out


def test_branch_2_a_stale_mapping_is_refused_never_applied(tmp_path):
    """A re-transcription renumbers the clusters; applying anyway misattributes."""
    save_speaker_mapping(
        {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript("an entirely different transcript"),
        source="human",
    )

    result, error, _, _ = run_step(tmp_path)

    assert isinstance(error, SpeakerMappingError)
    assert "made for a different transcript" in str(error)
    assert "Nothing has been deleted" in str(error)
    assert result is None
    assert os.path.isfile(speaker_map_path(BASE, str(tmp_path))), (
        "a stale mapping is flagged, never deleted (ADR-010 Open Question)"
    )


def test_branch_3_no_file_asks_the_model_and_saves_the_answer(tmp_path, monkeypatch):
    """The one branch that may pay, and it must leave something to correct."""
    answer = {"[SPEAKER_00]": "Nancy", "[SPEAKER_01]": "Ad"}
    calls = []

    def analyse(transcript, output_basename=None, output_dir=None):
        calls.append(output_basename)
        return dict(answer)

    monkeypatch.setattr(speakers_module, "analyze_speakers_with_o4", analyse)

    result, error, text, _ = run_step(tmp_path)

    assert error is None, error
    assert calls == [BASE], "branch 3 is the branch that asks the model"
    saved = json.loads(
        Path(speaker_map_path(BASE, str(tmp_path))).read_text(encoding="utf-8")
    )
    assert saved["speakers"] == {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}, (
        "keys are stored bare, so the next run and the editor can read them"
    )
    assert saved["source"] == "llm"
    assert saved["transcript_fingerprint"] == fingerprint_transcript(TRANSCRIPT)
    assert "Nancy:" in result


def test_branch_4_a_malformed_mapping_raises_and_is_not_papered_over(tmp_path):
    """A typo must be visible, not replaced by a paid call (ADR-010 Must Not)."""
    write_raw(speaker_map_path(BASE, str(tmp_path)), '{"SPEAKER_00": ')

    result, error, _, _ = run_step(tmp_path)

    assert isinstance(error, SpeakerMappingError)
    assert "not valid JSON" in str(error)
    assert result is None


# ---------------------------------------------------------------------------
# The bare hand-edited form
# ---------------------------------------------------------------------------


def test_the_bare_object_form_loads_and_counts_as_human(tmp_path):
    """``{"SPEAKER_00": "Nancy"}`` is the form the format exists to support."""
    write_raw(
        speaker_map_path(BASE, str(tmp_path)),
        '{"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}\n',
    )

    record = speakers_module.load_speaker_map_record(BASE, str(tmp_path))

    assert record["speakers"] == {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}
    assert record["source"] == "human", "if a person typed it, a person is the source"
    assert record["transcript_fingerprint"] is None


def test_a_bracketed_hand_edit_loads_with_bare_keys(tmp_path):
    """What somebody pastes out of the analysis report still has to work."""
    write_raw(speaker_map_path(BASE, str(tmp_path)), '{"[SPEAKER_00]": "Nancy"}\n')

    assert load_speaker_mapping(BASE, str(tmp_path)) == {"SPEAKER_00": "Nancy"}


def test_a_fingerprintless_mapping_applies_but_says_it_was_not_checked(tmp_path):
    """The documented trade-off, made visible instead of silent.

    ADR-010's Must says a re-transcription marks a mapping stale because "the
    file records which transcript it was made for" - and the bare hand-written
    form records nothing, so it can never be flagged. Refusing it would break
    the case the form exists for, so the run says the check was not made.
    """
    write_raw(speaker_map_path(BASE, str(tmp_path)), '{"SPEAKER_00": "Nancy"}\n')

    result, error, text, events = run_step(tmp_path)

    assert error is None, error
    assert "Nancy:" in result
    assert "records no transcript fingerprint" in text
    assert any(event.level == "warning" for event in events)


# ---------------------------------------------------------------------------
# Malformed input, in the shapes a person actually produces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, expected",
    [
        ('["SPEAKER_00", "Nancy"]', "not a JSON object"),
        ('{"speakers": "Nancy"}', '"speakers" holds a str'),
        ('{"SPEAKER_00": 42}', "must be text"),
        ('{"SPEAKER_00": "Nancy", "": "Ad"}', "empty speaker label"),
        ("{}", "maps no speakers at all"),
        ('{"SPEAKER_00": "   "}', "is blank"),
        ('{"speakers": {"SPEAKER_00": "Nancy"}, "transcript_fingerprint": 7}',
         "it must be the sha256"),
    ],
)
def test_malformed_mappings_raise_speaker_mapping_error(tmp_path, payload, expected):
    """Every unusable shape reports itself; none falls back to a paid call."""
    write_raw(speaker_map_path(BASE, str(tmp_path)), payload)

    with pytest.raises(SpeakerMappingError) as caught:
        speakers_module.load_speaker_map_record(BASE, str(tmp_path))

    assert expected in str(caught.value)


def test_a_blank_name_is_refused_on_the_way_in_as_well(tmp_path):
    """Refusing to write a file we would refuse to read back.

    A whitespace-only name used to be stored as ``""`` - a non-empty dict, so
    it passed the "it is empty" guard - and then produced a transcript line
    beginning with a bare ``":"``.
    """
    with pytest.raises(SpeakerMappingError, match="is blank"):
        save_speaker_mapping(
            {"SPEAKER_00": "   ", "SPEAKER_01": "Ad"}, BASE, str(tmp_path)
        )

    assert not os.path.exists(speaker_map_path(BASE, str(tmp_path)))


def test_a_mapping_naming_nobody_in_this_transcript_is_refused(tmp_path):
    """The dropped zero: ``SPEAKER_1`` where the transcript says ``SPEAKER_01``.

    This used to succeed. The mapping took precedence, the paid analysis was
    skipped, every replacement missed, and ``handle_unknown_speakers`` turned
    the real labels into ``Speaker 00:`` - a run reporting success while
    producing a transcript strictly worse than the model would have made.
    """
    write_raw(
        speaker_map_path(BASE, str(tmp_path)),
        '{"SPEAKER_1": "Nancy", "SPEAKER_2": "Ad"}\n',
    )

    result, error, _, _ = run_step(tmp_path)

    assert isinstance(error, SpeakerMappingError)
    assert "names none of the speakers" in str(error)
    assert "SPEAKER_1 and SPEAKER_01 are different labels" in str(error)
    assert result is None


def test_a_partial_mapping_is_allowed_and_warns_about_the_rest(tmp_path):
    """Naming one of two speakers is a supported edit, not an error.

    But the labels it did not name must be *reported*. That warning is what
    ADR-010 nominates as its stale-mapping signal, and it was unreachable: the
    check ran after ``handle_unknown_speakers`` had already rewritten every
    remaining tag, so the run said "All speaker tags successfully mapped"
    unconditionally, however wrong the mapping was.
    """
    write_raw(speaker_map_path(BASE, str(tmp_path)), '{"SPEAKER_00": "Nancy"}\n')

    result, error, text, events = run_step(tmp_path)

    assert error is None, error
    assert "Nancy:" in result
    assert "Unmapped speakers: [SPEAKER_01]" in text
    assert "All speaker tags successfully mapped" not in text
    assert any(event.level == "warning" for event in events)


def test_the_all_clear_is_only_said_when_it_is_true(tmp_path):
    """The control for the test above: a complete mapping still reports success."""
    write_raw(
        speaker_map_path(BASE, str(tmp_path)),
        '{"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}\n',
    )

    _, error, text, _ = run_step(tmp_path)

    assert error is None, error
    assert "All speaker tags successfully mapped" in text
    assert "Unmapped speakers" not in text


def test_speaker_unknown_alone_never_counts_as_an_unmapped_label(tmp_path):
    """Every transcript in this corpus has SPEAKER_UNKNOWN segments.

    Warning on them would fire the stale-mapping signal on 100% of real runs,
    which is the same as having no signal - just noisier.
    """
    write_raw(
        speaker_map_path(BASE, str(tmp_path)),
        '{"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}\n',
    )

    _, error, text, _ = run_step(tmp_path)

    assert error is None, error
    assert "Unmapped speakers" not in text


# ---------------------------------------------------------------------------
# A name is data, not a regular-expression template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [r"Nancy \1", r"C:\temp", r"Ad \g<x>", "B$0rn"])
def test_a_name_is_substituted_literally(name):
    """``re.sub`` reads its replacement as a template; these names are text.

    A hand-edited mapping is arbitrary text. ``\\1`` raises "invalid group
    reference" mid-run, after the GPU has been paid for; ``C:\\temp`` silently
    expands ``\\t`` to a tab.
    """
    out = speakers_module.apply_speaker_mapping_programmatically(
        "[SPEAKER_00] hello\n", {"SPEAKER_00": name}
    )

    assert out == f"{name}: hello\n"
    assert "\t" not in out


# ---------------------------------------------------------------------------
# A correction survives (ADR-010's Confirmation #3)
# ---------------------------------------------------------------------------


def test_a_hand_edit_survives_a_re_run_and_replaces_the_models_guess(tmp_path):
    """Run once with the model, correct the file, run again: the edit wins."""
    calls = []

    def analyse(transcript, output_basename=None, output_dir=None):
        calls.append(output_basename)
        return {"[SPEAKER_00]": "Nancy", "[SPEAKER_01]": "Ad"}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(speakers_module, "analyze_speakers_with_o4", analyse)
        first, error, _, _ = run_step(tmp_path)
    assert error is None, error
    assert "Nancy:" in first and calls == [BASE]

    # The person corrects the file: SPEAKER_00 was never Nancy, it was Marvin.
    save_speaker_mapping(
        {"SPEAKER_00": "Marvin", "SPEAKER_01": "Ad"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript(TRANSCRIPT),
        source="human",
    )

    # The tripwire is back in force: a second run must not ask the model again.
    second, error, text, _ = run_step(tmp_path)

    assert error is None, error
    assert "Marvin:" in second
    assert "Nancy" not in second, "the correction replaces the guess permanently"
    assert calls == [BASE], "the re-run must not pay for the answer again"
    assert "edited by hand" in text


def test_an_in_flight_run_does_not_overwrite_a_mapping_saved_meanwhile(tmp_path):
    """The window is the whole duration of the paid analysis.

    ``speaker_assignment_step`` checks for the file at the start of phase 1 and
    writes at the end of it. A person who saves in between would have had their
    names replaced by the model's, ``source`` flipped from "human" back to
    "llm", with no signal at all - which ADR-010's Must Not covers as squarely
    as deletion does.
    """
    def analyse(transcript, output_basename=None, output_dir=None):
        # The editor saves while the analysis is running.
        save_speaker_mapping(
            {"SPEAKER_00": "Typed by hand", "SPEAKER_01": "Ad"},
            BASE,
            str(tmp_path),
            source="human",
        )
        return {"[SPEAKER_00]": "Nancy", "[SPEAKER_01]": "Ad"}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(speakers_module, "analyze_speakers_with_o4", analyse)
        _, error, text, _ = run_step(tmp_path)

    assert error is None, error
    saved = json.loads(
        Path(speaker_map_path(BASE, str(tmp_path))).read_text(encoding="utf-8")
    )
    assert saved["speakers"]["SPEAKER_00"] == "Typed by hand"
    assert saved["source"] == "human", "the model's guess must not reclaim the file"
    assert "discarded rather than overwriting" in text


def test_saving_a_mapping_keeps_the_previous_one(tmp_path):
    """ADR-009: human input is written with ``backup=True``. One undo."""
    path = Path(speaker_map_path(BASE, str(tmp_path)))
    backup = Path(str(path) + BACKUP_SUFFIX)

    save_speaker_mapping({"SPEAKER_00": "Nancy"}, BASE, str(tmp_path), source="human")
    assert not backup.exists(), "nothing to back up on a first write"

    save_speaker_mapping({"SPEAKER_00": "Marvin"}, BASE, str(tmp_path), source="human")

    assert json.loads(path.read_text(encoding="utf-8"))["speakers"] == {
        "SPEAKER_00": "Marvin"
    }
    assert json.loads(backup.read_text(encoding="utf-8"))["speakers"] == {
        "SPEAKER_00": "Nancy"
    }


# ---------------------------------------------------------------------------
# A force-reprocess must not destroy human work (ADR-010's Must Not)
# ---------------------------------------------------------------------------


def test_a_force_reprocess_keeps_the_mapping_and_its_backup(tmp_path):
    """The blocker: ``--force`` used to take the mapping *and* its ``.bak``.

    Both copies of somebody's typing, gone on one click, in a directory that is
    gitignored. The boundary controls are here too: ``episode_43`` and
    ``episode_4`` prove the ADR-009 neighbour fix still holds, so this test
    fails if either protection regresses.
    """
    audio = tmp_path / f"{BASE}.mp3"
    files = {
        f"{BASE}.mp3": "audio",
        f"{BASE}_transcript.txt": "generated",
        f"{BASE}_summary.txt": "generated",
        f"{BASE}_speaker_assignment.txt": "generated",
        f"{BASE}{SPEAKER_MAP_SUFFIX}": '{"SPEAKER_00": "Nancy"}',
        f"{BASE}{SPEAKER_MAP_SUFFIX}{BACKUP_SUFFIX}": '{"SPEAKER_00": "Marvin"}',
        f"episode_43{SPEAKER_MAP_SUFFIX}": '{"SPEAKER_00": "Neighbour"}',
        f"episode_4{SPEAKER_MAP_SUFFIX}": '{"SPEAKER_00": "Neighbour"}',
        "episode_43_summary.txt": "a neighbour's artifact",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    delete_episode_files(BASE, str(tmp_path), exclude_files=[str(audio)])

    survivors = sorted(p.name for p in tmp_path.iterdir())
    assert survivors == sorted([
        f"{BASE}.mp3",
        f"{BASE}{SPEAKER_MAP_SUFFIX}",
        f"{BASE}{SPEAKER_MAP_SUFFIX}{BACKUP_SUFFIX}",
        "episode_43_summary.txt",
        f"episode_43{SPEAKER_MAP_SUFFIX}",
        f"episode_4{SPEAKER_MAP_SUFFIX}",
    ])
    assert (tmp_path / f"{BASE}{SPEAKER_MAP_SUFFIX}").read_text(
        encoding="utf-8"
    ) == '{"SPEAKER_00": "Nancy"}', "the surviving mapping must be the real one"


def test_a_force_reprocess_still_deletes_the_artifacts_it_is_for(tmp_path):
    """The control. Sparing human input must not spare everything."""
    (tmp_path / f"{BASE}_summary.txt").write_text("generated", encoding="utf-8")
    (tmp_path / f"{BASE}_transcript.txt").write_text("generated", encoding="utf-8")

    delete_episode_files(BASE, str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_the_input_predicate_and_the_backup_suffix_agree(tmp_path):
    """``episodes.py`` repeats ``.bak`` rather than importing it. Pin them equal."""
    assert episodes_module._BACKUP_SUFFIX == BACKUP_SUFFIX

    assert episodes_module.is_input_filename(f"{BASE}{SPEAKER_MAP_SUFFIX}")
    assert episodes_module.is_input_filename(
        f"{BASE}{SPEAKER_MAP_SUFFIX}{BACKUP_SUFFIX}"
    )
    assert episodes_module.is_input_filename(f"{BASE}_SPEAKERS.JSON"), "case-insensitive"
    assert not episodes_module.is_input_filename(f"{BASE}_summary.txt")
    assert not episodes_module.is_input_filename(f"{BASE}_transcript.txt.bak")


def test_the_scanner_calls_the_mapping_an_input_not_an_artifact(tmp_path):
    """The taxonomy the delete rule leans on, checked at the source."""
    assert "speakers_map" in episodes_module.INPUT_KINDS
    assert "speakers_map" in episodes_module.ARTIFACT_KINDS

    (tmp_path / f"{BASE}.mp3").write_bytes(b"audio")
    (tmp_path / f"{BASE}{SPEAKER_MAP_SUFFIX}").write_text(
        '{"SPEAKER_00": "Nancy"}', encoding="utf-8"
    )

    scan = episodes_module.scan_podcasts(str(tmp_path))
    artifacts = scan.episodes[0].artifacts
    mapping = [a for a in artifacts if a.kind == "speakers_map"]

    assert len(mapping) == 1
    assert mapping[0].is_input is True
    assert not scan.unmatched, "the mapping must not be reported as a stray"
