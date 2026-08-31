"""Attributing transcribed words to diarized speakers (ADR-002, ADR-004).

Whisper says what was said and when; pyannote says who was speaking and when.
Neither knows about the other, and their boundaries do not line up: Whisper cuts
on its own rhythm - breath, punctuation, its 30-second window - while pyannote
cuts on speaker turns. Joining the two is this module's whole job.

It used to be done by taking the midpoint of a Whisper segment and asking which
diarization turn contained it. Two things were wrong with that, both measured on
podcasts/episode_0.mp3 with pyannote/speaker-diarization-3.1 on 2026-08-30:

* The midpoint often lands in silence. That episode has 215 gaps between turns
  totalling 334 seconds, and 23 of 234 segments (10%) had no speaker at all.
  Taking the speaker with the most *overlap* instead leaves 0 of 234 unassigned.

* A Whisper segment can span a speaker change, and 53 of 234 segments (23%) did.
  The whole segment then gets one label, so one person's words are filed under
  another's name. The opening line of that episode - "Welcome to the WHYcast. I
  am Nancy. I'm Chantal. And I'm Ad." - is three people under one label.

The second problem cannot be fixed at segment level at all: no single label is
right for a segment covering two speakers. It has to be cut where the speaker
changes, which needs word timings. Those were already being computed
(``word_timestamps=True`` in transcribe_audio, enabled with the comment "for
better alignment with diarization") and then thrown away - nothing in the
package ever read ``segment.words``. This module reads them.
"""

import logging
from typing import Iterable, List, NamedTuple, Optional, Sequence

logger = logging.getLogger(__name__)


class Turn(NamedTuple):
    """One stretch of speech by one speaker, as diarization heard it."""

    start: float
    end: float
    speaker: str


class Span(NamedTuple):
    """Consecutive words that belong to one speaker."""

    start: float
    end: float
    speaker: Optional[str]
    text: str


def turns_from_diarization(speaker_segments) -> List[Turn]:
    """Normalise whatever diarization handed us into a list of turns.

    Accepts a pyannote ``Annotation`` (what the pipeline actually passes) or a
    list of dicts (what the type hints in transcription.py claim it passes).
    Both shapes appear in this codebase, so both are handled here rather than
    at every call site.
    """
    if not speaker_segments:
        return []

    if hasattr(speaker_segments, "itertracks"):
        return [
            Turn(segment.start, segment.end, label)
            for segment, _, label in speaker_segments.itertracks(yield_label=True)
        ]

    turns: List[Turn] = []
    for entry in speaker_segments:
        if isinstance(entry, dict):
            speaker = entry.get("speaker") or entry.get("label")
            if speaker is None:
                continue
            turns.append(Turn(float(entry["start"]), float(entry["end"]), speaker))
        elif isinstance(entry, Turn):
            turns.append(entry)
        else:
            start, end, speaker = entry
            turns.append(Turn(float(start), float(end), speaker))
    return turns


def speaker_for_interval(start: float, end: float, turns: Sequence[Turn]) -> Optional[str]:
    """The speaker who talks through most of ``[start, end]``.

    Returns None only when the interval overlaps no turn at all, which for a
    word means it fell entirely inside silence that diarization skipped.

    Overlap rather than midpoint containment: a word or segment straddling the
    edge of a turn still belongs to whoever was speaking for most of it, and a
    midpoint that lands in a gap between two turns is not evidence of anything.
    """
    best: Optional[str] = None
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best, best_overlap = turn.speaker, overlap
    return best


def _words_of(segment) -> List:
    """The word-level timings of a segment, or nothing when they are absent.

    ``word_timestamps`` can be off, and a segment can carry no words even when
    it is on (a filtered-out non-speech stretch). Callers fall back to segment
    level in that case rather than dropping the text.
    """
    words = getattr(segment, "words", None)
    return [w for w in (words or []) if getattr(w, "word", "").strip()]


def attribute_words_to_speakers(segments: Iterable, speaker_segments) -> List[Span]:
    """Cut the transcript at speaker changes instead of at Whisper's boundaries.

    Every word is given to the speaker who overlaps it most; consecutive words
    with the same speaker become one span. A segment covering two speakers
    therefore comes out as two spans, each with the right name.

    Args:
        segments: Whisper segments, ideally carrying ``.words``.
        speaker_segments: Diarization output, in any shape
            :func:`turns_from_diarization` accepts.

    Returns:
        Spans in time order. When there is no diarization the whole transcript
        comes back as spans with ``speaker=None``, which is how the caller
        knows to write no label at all - the same as before this module.
    """
    turns = turns_from_diarization(speaker_segments)

    # [start, end, text, speaker, segment_index] for every unit we can
    # attribute. A segment without word timings stays whole; it is better to
    # label it as one block than to lose it.
    units = []
    segments_without_words = 0
    for index, segment in enumerate(segments):
        text = (segment.text or "").strip()
        words = _words_of(segment)
        if words:
            for word in words:
                units.append([word.start, word.end, word.word, None, index])
        elif text:
            segments_without_words += 1
            units.append([segment.start, segment.end, text, None, index])

    if segments_without_words:
        logger.info(
            "%d segments had no word timings; attributed as whole segments",
            segments_without_words,
        )

    if not units:
        return []

    if turns:
        for unit in units:
            unit[3] = speaker_for_interval(unit[0], unit[1], turns)
        _fill_unattributed(units)

    return _group_into_spans(units)


def _fill_unattributed(units: List[List]) -> None:
    """Give words that overlap no turn to the speaker around them.

    A short word inside a pause - "yeah", a laugh, the tail of a sentence
    running past the end of a turn - can miss every turn. Leaving it as
    SPEAKER_UNKNOWN strands real text; the speaker before it is the far better
    guess, and the speaker after it is the fallback when the gap opens the
    transcript.

    This is the rule the old code meant to apply and never could: its
    continuity branch sat inside ``if speaker:``, so the "no speaker found"
    case it was written for could not reach it.
    """
    previous: Optional[str] = None
    for unit in units:
        if unit[3] is not None:
            previous = unit[3]
        elif previous is not None:
            unit[3] = previous

    following: Optional[str] = None
    for unit in reversed(units):
        if unit[3] is not None:
            following = unit[3]
        elif following is not None:
            unit[3] = following


def _group_into_spans(units: List[List]) -> List[Span]:
    """Group words into spans, breaking on a speaker change or a segment edge.

    Breaking on the segment edge as well as on the speaker keeps this change to
    what it is for. A segment that covers one speaker comes out as exactly the
    line it was before, with the timestamp it had; a segment that covers two
    comes out as two lines. Without that second rule, neighbouring segments by
    the same speaker would fuse into one long line and the timestamped
    transcript would lose most of its resolution - a change nobody asked for.
    Merging whole turns together is what merge_speaker_lines already does,
    downstream, for the file that wants it.
    """
    spans: List[Span] = []
    start, end, pieces = units[0][0], units[0][1], [units[0][2]]
    speaker, segment_index = units[0][3], units[0][4]

    for unit_start, unit_end, text, unit_speaker, unit_index in units[1:]:
        if unit_speaker == speaker and unit_index == segment_index:
            end = unit_end
            pieces.append(text)
        else:
            spans.append(Span(start, end, speaker, "".join(pieces).strip()))
            start, end, pieces = unit_start, unit_end, [text]
            speaker, segment_index = unit_speaker, unit_index

    spans.append(Span(start, end, speaker, "".join(pieces).strip()))
    return [s for s in spans if s.text]
