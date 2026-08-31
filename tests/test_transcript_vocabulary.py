"""Vocabulary corrections must survive word-level speaker attribution (ADR-006).

This is a regression test for a defect introduced when the transcript started
being assembled from ``segment.words`` instead of ``segment.text``.

``process_segment`` applies the vocabulary map and writes the result back to
``segment.text``. It never touches ``segment.words``, which carry exactly what
Whisper heard. So the moment the writer switched to words, ADR-006 stopped
applying: measured on episode 0, the published transcript went from 3
occurrences of "WHYcast" to 0, saying "YCast" instead.
"""

import json

import pytest

from whycast.pipeline import transcription, vocabulary


class FakeWord:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class FakeSegment:
    def __init__(self, start, end, text, words):
        self.start, self.end, self.text, self.words = start, end, text, words


@pytest.fixture
def vocab(tmp_path, monkeypatch):
    """A live vocabulary.json, wired the way config would wire it."""
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps({"YCast": "WHYcast", "Rai": "WHY"}), encoding="utf-8")
    monkeypatch.setattr(vocabulary, "USE_CUSTOM_VOCABULARY", True)
    monkeypatch.setattr(vocabulary, "VOCABULARY_FILE", str(path))
    return path


def write(tmp_path, segments, speaker_segments=None):
    clean = tmp_path / "out.txt"
    stamped = tmp_path / "out_ts.txt"
    transcription.write_transcript_files(segments, str(clean), str(stamped), speaker_segments)
    return clean.read_text(encoding="utf-8"), stamped.read_text(encoding="utf-8")


class TestCorrectionsReachTheWrittenTranscript:
    def test_a_term_only_present_in_the_words_is_still_corrected(self, tmp_path, vocab):
        """The regression: the fix lives in segment.text, the output comes from words."""
        segment = FakeSegment(0.0, 2.0, "Welcome to the WHYcast.", [
            FakeWord(0.0, 0.5, "Welcome"), FakeWord(0.5, 1.0, " to"),
            FakeWord(1.0, 1.4, " the"), FakeWord(1.4, 2.0, " YCast."),
        ])
        clean, stamped = write(tmp_path, [segment])
        assert "WHYcast" in clean
        assert "YCast" not in clean
        assert "WHYcast" in stamped

    def test_correction_applies_to_every_span_of_a_split_segment(self, tmp_path, vocab):
        """A segment cut at a speaker change must be corrected in both halves."""
        from whycast.pipeline.attribution import Turn

        segment = FakeSegment(0.0, 4.0, "", [
            FakeWord(0.0, 1.0, "YCast"), FakeWord(1.0, 1.9, " rocks"),
            FakeWord(2.1, 3.0, " YCast"), FakeWord(3.0, 4.0, " again"),
        ])
        turns = [Turn(0.0, 2.0, "SPEAKER_00"), Turn(2.0, 4.0, "SPEAKER_01")]
        clean, _ = write(tmp_path, [segment], turns)
        assert clean.count("WHYcast") == 2, clean
        assert "YCast" not in clean

    def test_speaker_labels_are_not_touched_by_the_replacements(self, tmp_path, monkeypatch):
        """A vocabulary entry must never rewrite the label itself."""
        path = tmp_path / "vocabulary.json"
        path.write_text(json.dumps({"SPEAKER_00": "Nobody"}), encoding="utf-8")
        monkeypatch.setattr(vocabulary, "USE_CUSTOM_VOCABULARY", True)
        monkeypatch.setattr(vocabulary, "VOCABULARY_FILE", str(path))
        from whycast.pipeline.attribution import Turn

        segment = FakeSegment(0.0, 2.0, "", [FakeWord(0.0, 2.0, "Hello")])
        clean, _ = write(tmp_path, [segment], [Turn(0.0, 2.0, "SPEAKER_00")])
        assert "[SPEAKER_00]" in clean


class TestWithoutVocabulary:
    def test_disabled_vocabulary_leaves_the_text_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vocabulary, "USE_CUSTOM_VOCABULARY", False)
        segment = FakeSegment(0.0, 1.0, "", [FakeWord(0.0, 1.0, "YCast")])
        clean, _ = write(tmp_path, [segment])
        assert "YCast" in clean
