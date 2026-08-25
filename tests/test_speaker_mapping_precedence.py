"""ADR-010's precedence rule, checked where it actually costs money.

``<base>_speakers.json`` exists so a correction survives a re-run and so a
re-run of an already-correct episode stops paying a reasoning model to
rediscover names it got right the first time. Both promises live in one branch
in :func:`whycast.pipeline.speakers.speaker_assignment_step`, and neither is
visible from the outside: the step returns an assigned transcript either way.
The only way to tell "used the saved mapping" from "asked the model again" is
to watch whether the model was asked.

So every test here patches :func:`analyze_speakers_with_o4` and asserts on its
call count, not just on the transcript that comes out.

Four branches, in the order ADR-010 states them:

1. file present, parses, fingerprint matches (or it has none) - use it, no call;
2. file present but stale - refuse, report, fail; never apply it blindly;
3. file absent - call the model as before, then save the answer;
4. file present but malformed - raise; never fall back to the model in silence.

The tripwires below are the other half of the point. If the programmatic
application fails for any incidental reason, ``speaker_assignment_step`` falls
back to :func:`speaker_assignment_fallback`, which calls OpenAI for real. A test
that only patched ``analyze_speakers_with_o4`` could therefore still spend
money while reporting success. Every run in this module asserts that no paid
entry point was reached at all.
"""

import json
import os

import pytest

from whycast.errors import SpeakerMappingError
from whycast.events import CollectSink, use_sink
from whycast.pipeline import speakers as speakers_module
from whycast.pipeline.speakers import (
    SPEAKER_MAP_SUFFIX,
    fingerprint_transcript,
    load_speaker_map_record,
    load_speaker_mapping,
    mapping_is_stale,
    save_speaker_mapping,
    speaker_map_path,
)

BASE = "episode_42"

TRANSCRIPT = (
    "[SPEAKER_00] Welcome to the WHYcast.\n"
    "[SPEAKER_01] Glad to be here.\n"
    "[SPEAKER_00] Shall we start at the beginning?\n"
    "[SPEAKER_01] Let's.\n"
)

#: What the model would answer, in the bracketed form
#: :func:`parse_speaker_mapping_from_analysis` produces.
MODEL_ANSWER = {"[SPEAKER_00]": "Nancy", "[SPEAKER_01]": "Ad"}


class _Tripwire:
    """Stands in for a paid call. Being called at all is the failure."""

    def __init__(self, name):
        self.name = name
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError(
            f"{self.name} was called: this run would have cost money"
        )


class _Run:
    """The outcome of one patched ``speaker_assignment_step``."""

    def __init__(self, result, error, events, analyze):
        self.result = result
        self.error = error
        self.events = events
        self.analyze = analyze

    @property
    def model_calls(self):
        return self.analyze.call_count

    @property
    def text(self):
        """Every emitted message, joined - for asserting on what a user is told."""
        return "\n".join(event.message for event in self.events)


@pytest.fixture
def run_step(monkeypatch):
    """Run the step with every paid path replaced by a tripwire.

    ``openai_available`` is forced True so the branch-3 test exercises the model
    path rather than the "no API key configured" early return, which would make
    the assertion pass for entirely the wrong reason.
    """

    def _run(output_dir, model_answer=None, transcript=TRANSCRIPT):
        analyze = _Recorder(model_answer)
        for name in ("process_with_openai", "speaker_assignment_fallback", "OpenAI"):
            monkeypatch.setattr(speakers_module, name, _Tripwire(name))
        monkeypatch.setattr(speakers_module, "analyze_speakers_with_o4", analyze)
        monkeypatch.setattr(speakers_module, "openai_available", True)

        sink = CollectSink()
        with use_sink(sink):
            try:
                result, error = (
                    speakers_module.speaker_assignment_step(
                        transcript, BASE, str(output_dir)
                    ),
                    None,
                )
            except Exception as exc:  # asserted on per branch
                result, error = None, exc
        return _Run(result, error, sink.events, analyze)

    return _run


class _Recorder:
    """A stand-in for the paid analysis that records whether it was wanted."""

    def __init__(self, answer):
        self.answer = answer
        self.call_count = 0

    def __call__(self, transcript, output_basename=None, output_dir=None):
        self.call_count += 1
        if self.answer is None:
            raise AssertionError(
                "analyze_speakers_with_o4 was called, but this test expected the "
                "saved mapping to be used instead - that is the paid call ADR-010 "
                "exists to avoid"
            )
        return dict(self.answer)


# ---------------------------------------------------------------------------
# Branch 1: the file is there and current
# ---------------------------------------------------------------------------


def test_a_saved_mapping_is_used_and_the_model_is_never_asked(tmp_path, run_step):
    """ADR-010's first Confirmation: mapped transcript, no OpenAI call."""
    save_speaker_mapping(
        {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript(TRANSCRIPT),
        source="human",
    )

    run = run_step(tmp_path)

    assert run.error is None
    assert run.model_calls == 0, "the saved mapping must replace the paid call"
    assert "Nancy" in run.result and "Ad" in run.result


def test_the_run_says_which_mapping_it_used_and_where_it_came_from(tmp_path, run_step):
    """Somebody who cannot see their correction being used will not trust it.

    The negative case matters as much: after a hand edit, re-running speaker
    assignment does *not* consult the model, which is the point but is not
    obvious from a button labelled "re-run speaker assignment". So the file is
    named and the skipped analysis is said out loud.
    """
    save_speaker_mapping(
        {"SPEAKER_00": "Nancy"}, BASE, str(tmp_path), source="human"
    )

    run = run_step(tmp_path)

    assert f"{BASE}{SPEAKER_MAP_SUFFIX}" in run.text
    assert "edited by hand" in run.text
    assert "skipped" in run.text
    assert "SPEAKER_00" in run.text and "Nancy" in run.text


def test_a_saved_mapping_still_applies_with_no_openai_configured(tmp_path, monkeypatch):
    """Branch 1 needs no model, so no API key should be required to honour it.

    Refusing to apply names a person already typed because the machine has no
    OpenAI key would be an odd way to respect their correction.
    """
    save_speaker_mapping(
        {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}, BASE, str(tmp_path)
    )
    monkeypatch.setattr(speakers_module, "openai_available", False)
    monkeypatch.setattr(
        speakers_module, "analyze_speakers_with_o4", _Tripwire("analyze_speakers_with_o4")
    )

    result = speakers_module.speaker_assignment_step(TRANSCRIPT, BASE, str(tmp_path))

    assert result is not None and "Nancy" in result


# ---------------------------------------------------------------------------
# Branch 3: no file yet
# ---------------------------------------------------------------------------


def test_with_no_mapping_the_model_runs_and_its_answer_is_saved(tmp_path, run_step):
    """ADR-010's second Confirmation: the LLM path is unchanged, plus a write."""
    path = speaker_map_path(BASE, str(tmp_path))
    assert not os.path.exists(path)

    run = run_step(tmp_path, model_answer=MODEL_ANSWER)

    assert run.error is None
    assert run.model_calls == 1, "with no saved mapping the model must be asked"
    assert os.path.exists(path), "the answer must be kept, or the next run pays again"

    saved = json.loads(open(path, encoding="utf-8").read())
    assert saved["speakers"] == {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}, (
        "bracketed keys from the analysis parser must be stored bare, the way "
        "the documented file format shows them"
    )
    assert saved["source"] == "llm"
    assert saved["transcript_fingerprint"] == fingerprint_transcript(TRANSCRIPT)
    assert saved["updated_at"]


def test_the_run_says_it_is_asking_the_model_and_then_that_it_saved(tmp_path, run_step):
    run = run_step(tmp_path, model_answer=MODEL_ANSWER)

    assert "asking the model" in run.text
    assert f"{BASE}{SPEAKER_MAP_SUFFIX}" in run.text
    assert "saved" in run.text


def test_a_second_run_of_the_same_transcript_costs_nothing(tmp_path, run_step):
    """The round trip is the promise: derive once, re-run free forever after.

    This is the test that would catch a fingerprint that is not stable across
    the save/load cycle - a mapping the pipeline wrote itself but then rejects
    as stale would turn every re-run into a hard failure.
    """
    first = run_step(tmp_path, model_answer=MODEL_ANSWER)
    assert first.model_calls == 1

    second = run_step(tmp_path)

    assert second.error is None
    assert second.model_calls == 0, (
        "the mapping written by the previous run must be accepted by the next "
        "one, not re-derived and not flagged stale"
    )
    assert "Nancy" in second.result


def test_a_hand_edit_survives_the_next_run(tmp_path, run_step):
    """ADR-010's third Confirmation, and the reason the file exists at all."""
    run_step(tmp_path, model_answer=MODEL_ANSWER)

    path = speaker_map_path(BASE, str(tmp_path))
    record = json.loads(open(path, encoding="utf-8").read())
    record["speakers"]["SPEAKER_00"] = "Ad"
    record["speakers"]["SPEAKER_01"] = "Nancy"
    record["source"] = "human"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    run = run_step(tmp_path)

    assert run.model_calls == 0
    assert "[SPEAKER_00] Welcome" not in run.result
    corrected = run.result.splitlines()
    assert any("Ad" in line and "Welcome to the WHYcast" in line for line in corrected), (
        "the corrected name must appear where the model's original one did"
    )


# ---------------------------------------------------------------------------
# Branch 2: stale
# ---------------------------------------------------------------------------


def test_a_stale_mapping_fails_the_step_instead_of_being_applied(tmp_path, run_step):
    """Diarization labels are per-run cluster ids, not identities.

    After a re-transcription ``SPEAKER_00`` may well be a different person, so
    applying the old names unchecked would confidently put one person's words in
    another's mouth. That is worse than failing, so this fails.
    """
    save_speaker_mapping(
        {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript("a different transcript"),
        source="human",
    )

    run = run_step(tmp_path)

    assert isinstance(run.error, SpeakerMappingError)
    assert run.model_calls == 0, (
        "a stale mapping must not silently become a paid re-analysis either - "
        "the person decides, not the pipeline"
    )


def test_a_stale_mapping_is_never_deleted(tmp_path, run_step):
    """It is human work. Discarding it is a decision only a person makes."""
    path = save_speaker_mapping(
        {"SPEAKER_00": "Nancy"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript("a different transcript"),
        source="human",
    )
    before = open(path, encoding="utf-8").read()

    run_step(tmp_path)

    assert os.path.exists(path)
    assert open(path, encoding="utf-8").read() == before


def test_the_stale_message_names_the_file_and_both_ways_out(tmp_path, run_step):
    """A failure a person cannot act on is just an outage.

    The message has to carry the filename, say that nothing was thrown away,
    and name both routes the UI offers: keep these names, or let the model
    decide again.
    """
    path = save_speaker_mapping(
        {"SPEAKER_00": "Nancy"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript("a different transcript"),
        source="human",
    )

    run = run_step(tmp_path)
    message = str(run.error)

    assert os.path.basename(path) in message
    assert "Nothing has been deleted" in message
    assert "transcript_fingerprint" in message, "the 'apply anyway' route"
    assert "delete the file" in message, "the 'let the model decide again' route"

    errors = [event for event in run.events if event.level == "error"]
    assert len(errors) == 1, "the sink is how the web UI shows this to a person"
    assert os.path.basename(path) in errors[0].message


@pytest.mark.parametrize(
    "make_bad",
    [
        pytest.param("stale", id="stale"),
        pytest.param("malformed", id="malformed"),
    ],
)
def test_the_remedy_survives_the_dashboards_error_column(tmp_path, make_bad):
    """The actionable half must outlive truncation, because that is what is read.

    ``webui.runner._short_error`` caps the job's error column at 500 characters.
    Both messages here are longer than that, so *something* is always cut - and
    the thing cut must be the reasoning, never the way out. This once failed:
    the stale message explained diarization clustering at length and then lost
    both remedies to the cap, leaving a person told only that their file was
    wrong and not what to do about it.

    The long base name and the real repository path are the point: a short
    tmp_path would hide the regression this guards.
    """
    from webui.runner import _MAX_DB_ERROR_CHARS, _short_error

    long_base = "episode_13_the_one_about_the_campsite_network"
    path = speaker_map_path(long_base, str(tmp_path))
    if make_bad == "stale":
        save_speaker_mapping(
            {"SPEAKER_00": "Nancy"},
            long_base,
            str(tmp_path),
            transcript_fingerprint=fingerprint_transcript("a different transcript"),
        )
        expected = ["Nothing has been deleted", "transcript_fingerprint", "delete the file"]
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"SPEAKER_00": "Nancy",}')
        expected = ["Fix it", "delete it"]

    with pytest.raises(SpeakerMappingError) as excinfo:
        load_speaker_map_record(
            long_base, str(tmp_path), transcript_fingerprint=fingerprint_transcript(TRANSCRIPT)
        )

    summary = _short_error(excinfo.value)
    assert len(summary) <= _MAX_DB_ERROR_CHARS + 40, "the cap plus its marker"
    assert os.path.basename(path) in summary, "a person must know which file"
    for phrase in expected:
        assert phrase in summary, (
            f"{phrase!r} was truncated away: the message explains before it "
            f"instructs, so the reader is left with a complaint and no remedy"
        )


def test_clearing_the_fingerprint_is_the_apply_anyway_route(tmp_path, run_step):
    """What the message tells the user to do must actually work."""
    path = save_speaker_mapping(
        {"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript("a different transcript"),
        source="human",
    )
    record = json.loads(open(path, encoding="utf-8").read())
    del record["transcript_fingerprint"]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    run = run_step(tmp_path)

    assert run.error is None
    assert run.model_calls == 0
    assert "Nancy" in run.result


def test_deleting_the_file_is_the_let_the_model_decide_again_route(tmp_path, run_step):
    save_speaker_mapping(
        {"SPEAKER_00": "Nancy"},
        BASE,
        str(tmp_path),
        transcript_fingerprint=fingerprint_transcript("a different transcript"),
    )
    os.remove(speaker_map_path(BASE, str(tmp_path)))

    run = run_step(tmp_path, model_answer=MODEL_ANSWER)

    assert run.error is None
    assert run.model_calls == 1


# ---------------------------------------------------------------------------
# Branch 4: malformed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason, content",
    [
        ("trailing comma", '{"SPEAKER_00": "Nancy",}'),
        ("not an object", '["Nancy", "Ad"]'),
        ("a name is not text", '{"SPEAKER_00": 42}'),
        ("maps nobody", "{}"),
        ("speakers is a string", '{"speakers": "Nancy"}'),
        ("speakers maps nobody", '{"speakers": {}}'),
        ("empty label", '{"   ": "Nancy"}'),
        (
            "fingerprint is a number",
            '{"speakers": {"SPEAKER_00": "N"}, "transcript_fingerprint": 42}',
        ),
    ],
)
def test_a_malformed_mapping_is_reported_not_papered_over(
    tmp_path, run_step, reason, content
):
    """A typo in a hand-edited file must be visible.

    Falling back to the model here would hide the mistake, charge for the
    privilege, and quietly discard whatever the person meant to type.
    """
    with open(speaker_map_path(BASE, str(tmp_path)), "w", encoding="utf-8") as handle:
        handle.write(content)

    run = run_step(tmp_path)

    assert isinstance(run.error, SpeakerMappingError), reason
    assert run.model_calls == 0, f"{reason}: fell back to the paid analysis"


def test_the_malformed_message_shows_the_shape_that_works(tmp_path):
    with open(speaker_map_path(BASE, str(tmp_path)), "w", encoding="utf-8") as handle:
        handle.write('{"SPEAKER_00": "Nancy",}')

    with pytest.raises(SpeakerMappingError) as excinfo:
        load_speaker_mapping(BASE, str(tmp_path))

    message = str(excinfo.value)
    assert os.path.basename(speaker_map_path(BASE, str(tmp_path))) in message
    assert '"SPEAKER_00": "Nancy"' in message, "show, do not merely describe"


# ---------------------------------------------------------------------------
# The bare hand-written form
# ---------------------------------------------------------------------------


def test_the_obvious_thing_written_by_hand_just_works(tmp_path, run_step):
    """``{"SPEAKER_00": "Nancy"}`` is what a person types. It must load.

    Requiring the envelope would mean the format only works for people who read
    the source first, which is not a format for humans.
    """
    with open(speaker_map_path(BASE, str(tmp_path)), "w", encoding="utf-8") as handle:
        json.dump({"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}, handle)

    assert load_speaker_mapping(BASE, str(tmp_path)) == {
        "SPEAKER_00": "Nancy",
        "SPEAKER_01": "Ad",
    }

    run = run_step(tmp_path)
    assert run.model_calls == 0
    assert "Nancy" in run.result


def test_a_hand_written_file_counts_as_human_and_is_never_stale(tmp_path):
    """It carries no fingerprint, and that must mean "applies", not "suspect".

    Treating a missing fingerprint as stale would break the one case the bare
    format exists to support.
    """
    with open(speaker_map_path(BASE, str(tmp_path)), "w", encoding="utf-8") as handle:
        json.dump({"SPEAKER_00": "Nancy"}, handle)

    record = load_speaker_map_record(BASE, str(tmp_path))

    assert record["source"] == "human"
    assert record["transcript_fingerprint"] is None
    assert not mapping_is_stale(record, fingerprint_transcript("anything at all"))


def test_bracketed_labels_load_as_bare_ones(tmp_path):
    """What a person copying from the analysis report would paste."""
    with open(speaker_map_path(BASE, str(tmp_path)), "w", encoding="utf-8") as handle:
        json.dump({"[SPEAKER_00]": "Nancy", "[SPEAKER_01]": "Ad"}, handle)

    assert load_speaker_mapping(BASE, str(tmp_path)) == {
        "SPEAKER_00": "Nancy",
        "SPEAKER_01": "Ad",
    }


def test_no_mapping_file_reads_as_none_not_as_an_error(tmp_path):
    """"Absent" is an ordinary state - it is branch 3, not a problem."""
    assert load_speaker_mapping(BASE, str(tmp_path)) is None
    assert load_speaker_map_record(BASE, str(tmp_path)) is None


# ---------------------------------------------------------------------------
# The fingerprint itself
# ---------------------------------------------------------------------------


def test_the_fingerprint_is_deterministic_and_content_sensitive():
    assert fingerprint_transcript(TRANSCRIPT) == fingerprint_transcript(TRANSCRIPT)
    assert fingerprint_transcript(TRANSCRIPT) != fingerprint_transcript(TRANSCRIPT + "!")
    assert len(fingerprint_transcript(TRANSCRIPT)) == 64


def test_the_fingerprint_survives_a_windows_line_ending_round_trip():
    """``atomic_write_text`` writes CRLF on Windows; the same text must hash alike.

    Without this, a mapping derived from a transcript held in memory would read
    back stale the moment that transcript had been through a file.
    """
    assert fingerprint_transcript(TRANSCRIPT) == fingerprint_transcript(
        TRANSCRIPT.replace("\n", "\r\n")
    )


def test_mapping_is_stale_needs_both_fingerprints_to_have_an_opinion():
    assert not mapping_is_stale({"transcript_fingerprint": "abc"}, None)
    assert not mapping_is_stale({"transcript_fingerprint": None}, "abc")
    assert not mapping_is_stale({"transcript_fingerprint": "abc"}, "abc")
    assert mapping_is_stale({"transcript_fingerprint": "abc"}, "def")


def test_saving_refuses_what_loading_would_reject(tmp_path):
    """Writing a file we would then refuse to read back helps nobody."""
    with pytest.raises(SpeakerMappingError):
        save_speaker_mapping({}, BASE, str(tmp_path))
    with pytest.raises(SpeakerMappingError):
        save_speaker_mapping({"SPEAKER_00": 42}, BASE, str(tmp_path))
    with pytest.raises(SpeakerMappingError):
        save_speaker_mapping(
            {"SPEAKER_00": "Nancy"}, BASE, str(tmp_path), source="astrology"
        )


def test_the_path_is_the_one_the_scanner_and_the_editor_agree_on(tmp_path):
    assert speaker_map_path(BASE, str(tmp_path)) == os.path.join(
        str(tmp_path), f"{BASE}{SPEAKER_MAP_SUFFIX}"
    )
    assert SPEAKER_MAP_SUFFIX == "_speakers.json"
