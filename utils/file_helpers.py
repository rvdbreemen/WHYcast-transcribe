#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File operation utilities for WHYcast-transcribe.

This module contains functions for file operations like checking file existence,
creating output paths, managing file sizes, and writing transcript files.
"""

import os
import logging
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm


def check_file_size(file_path: str, max_file_size_kb: int) -> bool:
    """
    Check if the file size is within acceptable limits.
    
    Args:
        file_path: Path to the file to check
        max_file_size_kb: Maximum acceptable file size in KB
        
    Returns:
        True if file size is acceptable, False if it's too large
    """
    size_kb = os.path.getsize(file_path) / 1024
    if size_kb > max_file_size_kb:
        logging.warning(f"File size ({size_kb:.1f} KB) exceeds recommended limit ({max_file_size_kb} KB). " 
                      f"Processing might be slow or fail.")
        return False
    return True


def create_output_paths(input_file: str) -> Tuple[str, str, str]:
    """
    Create output file paths based on the input filename.
    
    Args:
        input_file: Path to the input audio file
        
    Returns:
        Tuple of (plain_text_path, timestamped_path, summary_path)   
    """
    base = os.path.splitext(input_file)[0]
    return (
        f"{base}.txt",             # Without timestamps
        f"{base}_ts.txt",          # With timestamps
        f"{base}_summary.txt",     # For summary
    )


def setup_output_base(input_file: str, output_dir: Optional[str] = None) -> str:
    """
    Set up the base path for output files.
    
    Args:
        input_file: Path to the input file
        output_dir: Optional output directory
        
    Returns:
        Base path for output files
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(os.path.splitext(input_file)[0])
        output_base = os.path.join(output_dir, base_name)
    else:
        output_base = os.path.splitext(input_file)[0]
    
    return output_base


def transcription_exists(input_file: str) -> bool:
    """
    Check if a transcription already exists for the given input file.
    
    Args:
        input_file: Path to the input file
        
    Returns:
        True if transcription exists, False otherwise
    """
    base = os.path.splitext(input_file)[0]
    transcript_file = f"{base}.txt"
    timestamped_file = f"{base}_ts.txt"
    return os.path.exists(transcript_file) and os.path.exists(timestamped_file)


def normalize_path(path: str) -> str:
    """
    Normalize a file path by removing trailing slashes.
    
    Args:
        path: File path to normalize
        
    Returns:
        Normalized path
    """
    return path.rstrip(os.path.sep)


def get_base_filename(file_path: str, strip_suffixes: bool = False) -> str:
    """
    Get the base filename without path and extension.
    
    Args:
        file_path: Path to the file
        strip_suffixes: Whether to strip known suffixes (_cleaned, _ts, etc.)
        
    Returns:
        Base filename
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    if strip_suffixes:
        known_suffixes = ["_cleaned", "_ts", "_summary", "_blog", "_blog_alt1", "_history"]
        for suffix in known_suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
    
    return base_name


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")


def write_transcript_files(segments: List, output_file: str, output_file_timestamped: str, 
                          speaker_segments: Optional[List[Dict]] = None) -> str:
    """
    Write transcript files with and without timestamps, integrating speaker info.
    
    Args:
        segments: List of transcription segments from Whisper
        output_file: Path to save the clean transcript
        output_file_timestamped: Path to save the timestamped transcript
        speaker_segments: Optional list of speaker segments from diarization
        
    Returns:
        Full transcript text
    """
    try:
        from utils.diarize import get_speaker_for_segment
        
        # Get logger
        logger = logging.getLogger(__name__)
        
        # Prepare for writing the transcripts
        full_transcript = []
        timestamped_transcript = []
        
        # Track current speaker to avoid repeating speaker tags for consecutive segments
        current_speaker = None
        
        print(f"\nCreating transcript files...")
        print(f"- Clean transcript: {os.path.basename(output_file)}")
        print(f"- Timestamped transcript: {os.path.basename(output_file_timestamped)}")
        
        # Process each segment from Whisper
        for i, segment in enumerate(tqdm(segments, desc="Processing transcript", unit="segment")):
            # Get text and timestamp
            start, end = segment.start, segment.end
            text = segment.text.strip()
            
            # Add speaker identification if available
            if speaker_segments:
                # Get speaker for this segment
                speaker = get_speaker_for_segment(
                    start, speaker_segments, segment_duration=(end - start), context_window=1.0
                )
                
                # Always include the speaker tag when speaker information is available
                if speaker:
                    text = f"[{speaker}] {text}"
                    current_speaker = speaker
                # Reset current speaker when there's a significant time gap or no speaker identified
                elif end - start > 1.5 or speaker is None:
                    if speaker_segments:
                        # If we have speaker segments but couldn't identify one for this segment
                        text = "[SPEAKER_UNKNOWN] " + text
                    current_speaker = None
            
            # Add to transcript collections
            full_transcript.append(text)
            timestamped_transcript.append(f"{format_timestamp(start, end)} {text}")
        
        # Join all lines
        full_text = "\n".join(full_transcript)
        timestamped_text = "\n".join(timestamped_transcript)
        
        # Write to files
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        with open(output_file_timestamped, 'w', encoding='utf-8') as f:
            f.write(timestamped_text)
            
        print(f"Transcript files successfully created:")
        print(f"- {output_file}")
        print(f"- {output_file_timestamped}")
        
        return full_text
    except Exception as e:
        logger.error(f"Error writing transcript files: {str(e)}")
        return ""


def format_timestamp(start: float, end: Optional[float] = None) -> str:
    """
    Format a timestamp in seconds to HH:MM:SS.SSS format using the format_time function.
    
    Args:
        start: Start timestamp in seconds
        end: Optional end timestamp in seconds
        
    Returns:
        Formatted timestamp string
    """
    # Import format_time from whisper_model to maintain consistency
    from utils.whisper_model import format_time
    
    if end is None:
        return f"[{format_time(start)}]"
    else:
        return f"[{format_time(start)} --> {format_time(end)}]"


def read_prompt_file(file_path: str, default_prompt: str = "") -> str:
    """
    Read a prompt from a text file.
    
    Args:
        file_path: Path to the prompt file
        default_prompt: Default prompt to use if file doesn't exist or is empty
        
    Returns:
        The prompt text from the file or the default prompt
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                if prompt:
                    return prompt
        
        if default_prompt:
            logging.warning(f"Prompt file not found or empty: {file_path}, using default prompt")
            # If the directory exists but the file doesn't, create it with the default prompt
            if os.path.exists(os.path.dirname(file_path)):
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(default_prompt)
                    logging.info(f"Created prompt file with default prompt: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to create prompt file: {str(e)}")
            return default_prompt
        else:
            logging.error(f"Prompt file not found or empty and no default provided: {file_path}")
            return "Please process the following text:"
    except Exception as e:
        logging.error(f"Error reading prompt file {file_path}: {str(e)}")
        return default_prompt if default_prompt else "Please process the following text:"