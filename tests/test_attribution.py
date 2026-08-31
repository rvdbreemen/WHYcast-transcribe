"""Joining Whisper words to pyannote turns (ADR-002).

The numbers in these tests are the ones measured on podcasts/episode_0.mp3 with
pyannote/speaker-diarization-3.1 on 2026-08-30; the shapes are miniatures of the
real failure modes found there.
"""

import pytest

from whycast.pipeline.attribution import (
    Span,
    Turn,
    attribute_words_to_speakers,
    speaker_for_interval,
    turns_from_diarization,
)


class FakeWord:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class FakeSegment:
    def __init__(self, start, end, text, words=None):
        self.start, self.end, self.text, self.words = start, end, text, words


def words(*triples):
    return [FakeWord(s, e, w) for s, e, w in triples]


class TestSpeakerForInterval:
    """Overlap, not midpoint containment."""

    TURNS = [Turn(0.0, 10.0, "SPEAKER_00"), Turn(12.0, 20.0, "SPEAKER_01")]

    def test_an_interval_inside_one_turn_gets_that_speaker(self):
        assert speaker_for_interval(2.0, 4.0, self.TURNS) == "SPEAKER_00"

    def test_the_midpoint_landing_in_a_gap_no_longer_loses_the_speaker(self):
        """The regression: midpoint 11.0 is in the 10-12 gap, so the old code
        returned None and the segment became SPEAKER_UNKNOWN."""
        assert speaker_for_interval(9.0, 13.0, self.TURNS) == "SPEAKER_00"

    def test_the_speaker_with_the_most_overlap_wins(self):
        # 12.0-20.0 overlaps by 6.0; 0.0-10.0 overlaps by 1.0.
        assert speaker_for_interval(9.0, 18.0, self.TURNS) == "SPEAKER_01"

    def test_no_overlap_at_all_gives_none(self):
        assert speaker_for_interval(10.5, 11.5, self.TURNS) is None

    def test_no_turns_gives_none(self):
        assert speaker_for_interval(1.0, 2.0, []) is None


class TestTurnsFromDiarization:
    """The pipeline passes a pyannote Annotation; the type hints claim dicts."""

    def test_reads_a_list_of_dicts(self):
        turns = turns_from_diarization([{"start": 0, "end": 5, "speaker": "SPEAKER_00"}])
        assert turns == [Turn(0.0, 5.0, "SPEAKER_00")]

    def test_reads_an_annotation_like_object(self):
        class FakeAnnotation:
            def itertracks(self, yield_label=False):
                seg = type("S", (), {"start": 1.0, "end": 4.0})()
                yield seg, None, "SPEAKER_02"

        assert turns_from_diarization(FakeAnnotation()) == [Turn(1.0, 4.0, "SPEAKER_02")]

    def test_nothing_gives_nothing(self):
        assert turns_from_diarization(None) == []
        assert turns_from_diarization([]) == []


class TestSegmentSpanningASpeakerChange:
    """The 23% case: one Whisper segment, two people."""

    TURNS = [Turn(0.0, 2.0, "SPEAKER_00"), Turn(2.0, 6.0, "SPEAKER_01")]

    @pytest.fixture
    def spans(self):
        segment = FakeSegment(0.0, 6.0, "Hi there I am Ad", words(
            (0.0, 0.5, "Hi"), (0.5, 1.9, " there"),
            (2.1, 3.0, " I"), (3.0, 4.0, " am"), (4.0, 5.0, " Ad"),
        ))
        return attribute_words_to_speakers([segment], self.TURNS)

    def test_the_segment_is_cut_at_the_speaker_change(self, spans):
        assert len(spans) == 2

    def test_each_half_gets_the_right_speaker(self, spans):
        assert spans[0].speaker == "SPEAKER_00"
        assert spans[1].speaker == "SPEAKER_01"

    def test_the_words_go_with_the_right_speaker(self, spans):
        assert spans[0].text == "Hi there"
        assert spans[1].text == "I am Ad"

    def test_no_text_is_lost(self, spans):
        assert " ".join(s.text for s in spans) == "Hi there I am Ad"


class TestSegmentBoundariesAreKept:
    """Splitting is the job; merging is not.

    Grouping words purely by speaker would fuse neighbouring segments by the
    same person into one long line, and the timestamped transcript would lose
    most of its resolution. A segment covering one speaker must come out as
    exactly the line it was, with the timestamp it had.
    """

    TURNS = [Turn(0.0, 20.0, "SPEAKER_00")]

    def test_two_segments_by_the_same_speaker_stay_two_lines(self):
        segments = [
            FakeSegment(0.0, 2.0, "First bit", words((0.0, 1.0, "First"), (1.0, 2.0, " bit"))),
            FakeSegment(2.0, 4.0, "Second bit", words((2.0, 3.0, "Second"), (3.0, 4.0, " bit"))),
        ]
        spans = attribute_words_to_speakers(segments, self.TURNS)
        assert [s.text for s in spans] == ["First bit", "Second bit"]

    def test_each_line_keeps_its_own_start_time(self):
        segments = [
            FakeSegment(0.0, 2.0, "First", words((0.0, 2.0, "First"))),
            FakeSegment(2.0, 4.0, "Second", words((2.0, 4.0, "Second"))),
        ]
        spans = attribute_words_to_speakers(segments, self.TURNS)
        assert [s.start for s in spans] == [0.0, 2.0]


class TestWordsFallingInSilence:
    """A word inside a gap keeps the speaker around it instead of becoming unknown."""

    TURNS = [Turn(0.0, 2.0, "SPEAKER_00"), Turn(5.0, 8.0, "SPEAKER_00")]

    def test_a_word_in_the_gap_is_attributed_to_the_surrounding_speaker(self):
        segment = FakeSegment(0.0, 8.0, "yes well yeah", words(
            (0.5, 1.5, "yes"), (3.0, 3.4, " well"), (5.5, 6.0, " yeah"),
        ))
        spans = attribute_words_to_speakers([segment], self.TURNS)
        assert len(spans) == 1
        assert spans[0].speaker == "SPEAKER_00"
        assert spans[0].text == "yes well yeah"

    def test_a_gap_at_the_very_start_borrows_from_what_follows(self):
        segment = FakeSegment(0.0, 8.0, "um hello", words(
            (0.1, 0.3, "um"), (5.5, 6.0, " hello"),
        ))
        spans = attribute_words_to_speakers(
            [segment], [Turn(5.0, 8.0, "SPEAKER_01")]
        )
        assert [s.speaker for s in spans] == ["SPEAKER_01"]


class TestWithoutWordTimings:
    """word_timestamps can be off; the text must still come through."""

    def test_a_segment_without_words_is_attributed_whole(self):
        segment = FakeSegment(0.0, 4.0, "No word timings here", words=None)
        spans = attribute_words_to_speakers([segment], [Turn(0.0, 4.0, "SPEAKER_00")])
        assert spans == [Span(0.0, 4.0, "SPEAKER_00", "No word timings here")]

    def test_empty_words_list_falls_back_to_the_segment(self):
        segment = FakeSegment(0.0, 4.0, "Still here", words=[])
        spans = attribute_words_to_speakers([segment], [Turn(0.0, 4.0, "SPEAKER_00")])
        assert spans[0].text == "Still here"


class TestWithoutDiarization:
    """No diarization means no labels, exactly as before."""

    def test_every_span_has_no_speaker(self):
        segment = FakeSegment(0.0, 2.0, "Hello there", words((0.0, 1.0, "Hello"), (1.0, 2.0, " there")))
        spans = attribute_words_to_speakers([segment], None)
        assert len(spans) == 1
        assert spans[0].speaker is None
        assert spans[0].text == "Hello there"


class TestNothingToAttribute:
    def test_no_segments_gives_no_spans(self):
        assert attribute_words_to_speakers([], [Turn(0, 1, "SPEAKER_00")]) == []

    def test_blank_segments_are_dropped(self):
        assert attribute_words_to_speakers([FakeSegment(0, 1, "   ", words=None)], []) == []
