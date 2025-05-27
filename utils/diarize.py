"""
Speaker diarization utilities for WHYcast-transcribe.
Implements get_speaker_for_segment for mapping time to speaker label.
"""
from typing import List, Dict, Optional

def get_speaker_for_segment(
    time: float,
    speaker_segments: List[Dict],
    segment_duration: Optional[float] = None,
    context_window: float = 0.5
) -> Optional[str]:
    """
    Given a time (in seconds) and a list of speaker segments (with 'start', 'end', 'speaker'),
    return the speaker label for that time. Optionally, use segment_duration and context_window
    to improve robustness for short utterances or ambiguous boundaries.
    """
    if not speaker_segments:
        return None
    # Find all segments that overlap with the given time (with context window)
    candidates = [
        seg for seg in speaker_segments
        if (seg['start'] - context_window) <= time <= (seg['end'] + context_window)
    ]
    if not candidates:
        # Try again with a larger window if segment_duration is very short
        if segment_duration is not None and segment_duration < 1.0:
            candidates = [
                seg for seg in speaker_segments
                if (seg['start'] - 2 * context_window) <= time <= (seg['end'] + 2 * context_window)
            ]
    if not candidates:
        return None
    # Prefer the segment with the smallest distance to the center
    def center_distance(seg):
        center = (seg['start'] + seg['end']) / 2
        return abs(center - time)
    best = min(candidates, key=center_distance)
    return best.get('speaker')
