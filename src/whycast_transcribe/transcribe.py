import logging
import time
import os
import sys
import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from whycast_transcribe.config import BEAM_SIZE, DIARIZATION_MIN_SPEAKERS, DIARIZATION_MAX_SPEAKERS, USE_SPEAKER_DIARIZATION

# Helper function to format timestamp
def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.ms format."""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

# Helper function to find speaker for a given time
def _get_speaker_for_time(timestamp: float, speaker_segments: List[Dict]) -> Optional[str]:
    """Find the speaker label for a specific timestamp based on diarization segments."""
    if not speaker_segments:
        return None
    for segment in speaker_segments:
        if segment['start'] <= timestamp <= segment['end']:
            return segment['speaker']
    return None

def get_speaker_segments(diarization_pipeline: Any, audio_path: str, min_speakers: int = None, max_speakers: int = None) -> Optional[List[Dict]]:
    """
    Process audio file with diarization pipeline to get speaker segments.
    Returns a list of dicts with 'speaker', 'start', 'end' keys or None if diarization is disabled or fails.
    """
    if diarization_pipeline is None:
        logging.info("Speaker diarization is disabled or unavailable")
        return None
    
    try:
        logging.info(f"Performing speaker diarization on {audio_path}")
        min_speakers = min_speakers or DIARIZATION_MIN_SPEAKERS
        max_speakers = max_speakers or DIARIZATION_MAX_SPEAKERS
        
        # Run diarization
        logging.info(f"Using min_speakers={min_speakers}, max_speakers={max_speakers}")
        diarization = diarization_pipeline(
            audio_path,
            num_speakers=max(1, min_speakers),
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Extract speaker segments from diarization result
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'speaker': speaker,
                'start': turn.start,
                'end': turn.end
            })
        
        logging.info(f"Diarization complete: found {len(speaker_segments)} speaker segments")
        return speaker_segments
    except Exception as e:
        logging.error(f"Speaker diarization failed: {e}")
        return None

def transcribe_audio(
    model: WhisperModel,
    audio_file: str,
    diarization_pipeline: Optional[Any] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    skip_diarization: bool = False,
    live_callback: Optional[callable] = None
) -> Tuple[List[Segment], object, Optional[List[Dict]]]:
    """
    Transcribe audio file, return segments and metadata.
    If diarization_pipeline is provided and not skipped, speaker labels are included.
    Prints live transcription output with speaker labels (if available) to the console.
    
    Args:
        model: Initialized WhisperModel
        audio_file: Path to the audio file
        diarization_pipeline: Optional diarization pipeline for speaker identification
        min_speakers: Minimum number of speakers to identify
        max_speakers: Maximum number of speakers to identify
        skip_diarization: Skip diarization even if pipeline is provided
        live_callback: Optional callback function to handle live transcription output
        
    Returns:
        tuple: (segments, info, speaker_segments)
    """
    logging.info(f"Beginning transcription for file: {audio_file}")
    
    # Handle speaker diarization
    speaker_segments = None
    if diarization_pipeline and not skip_diarization:
        speaker_segments = get_speaker_segments(
            diarization_pipeline, 
            audio_file,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
    
    # Proceed with transcription
    start_time = time.time()
    segments_gen, info = model.transcribe(
        audio_file,
        beam_size=BEAM_SIZE,
        word_timestamps=True
    )

    segments = []
    print("--- Live Transcription Output ---", file=sys.stdout)
    if live_callback:
        live_callback("--- Live Transcription Output ---")
        
    for segment in segments_gen:
        # Determine speaker label for the segment's midpoint
        speaker_label = None
        if speaker_segments:
            segment_midpoint = (segment.start + segment.end) / 2
            speaker_label = _get_speaker_for_time(segment_midpoint, speaker_segments)

        # Format output string with speaker label if found
        speaker_prefix = f"[{speaker_label}] " if speaker_label else ""
        # Use the timestamp formatting function
        start_ts = format_timestamp(segment.start)
        end_ts = format_timestamp(segment.end)
        output_line = f"[{start_ts} -> {end_ts}] {speaker_prefix}{segment.text}"

        # Print live output to console
        print(output_line, file=sys.stdout)
        
        # Call the callback if provided
        if live_callback:
            live_callback(output_line)
            
        segments.append(segment)
        
    print("--- End of Live Transcription ---", file=sys.stdout)
    if live_callback:
        live_callback("--- End of Live Transcription ---")

    elapsed = time.time() - start_time
    logging.info(f"Transcription ended; duration: {elapsed:.1f}s, {len(segments)} segments produced")
    logging.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    # Return both segments and speaker segments for downstream processing
    return segments, info, speaker_segments

def write_transcript_files(
    segments: List[Segment],
    output_file: str,
    output_file_timestamped: str,
    speaker_segments: Optional[List[Dict]] = None
) -> str:
    """
    Write transcripts to files (plain and timestamped) and return full transcript text.
    The timestamped file includes start/end times and speaker labels if available.
    """
    # Write plain text transcript
    plain_lines = [seg.text.strip() for seg in segments]
    full_text = "\n".join(plain_lines)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        logging.info(f"Successfully wrote plain transcript to {output_file}")
    except Exception as e:
        logging.error(f"Failed to write plain transcript {output_file}: {e}")

    # Write timestamped transcript
    try:
        with open(output_file_timestamped, 'w', encoding='utf-8') as f:
            for segment in segments:
                # Determine speaker label for the segment's midpoint
                speaker_label = None
                if speaker_segments:
                    segment_midpoint = (segment.start + segment.end) / 2
                    speaker_label = _get_speaker_for_time(segment_midpoint, speaker_segments)

                # Format output string with speaker label if found
                speaker_prefix = f"[{speaker_label}] " if speaker_label else ""
                # Use the new format_timestamp function
                start_ts = format_timestamp(segment.start)
                end_ts = format_timestamp(segment.end)
                timestamped_line = f"[{start_ts} -> {end_ts}] {speaker_prefix}{segment.text.strip()}"
                f.write(timestamped_line + "\n")
        logging.info(f"Successfully wrote timestamped transcript to {output_file_timestamped}")
    except Exception as e:
        logging.error(f"Failed to write timestamped transcript {output_file_timestamped}: {e}")

    return full_text