#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio processing utilities for WHYcast-transcribe.

This module provides functions for:
- Detecting audio file formats
- Converting audio to supported formats for diarization and transcription
- Checking for ffmpeg availability and handling temporary files

Intended for use in all modules that require robust audio format handling and conversion for ML workflows.
"""

import os
import logging
import subprocess
import shutil
from typing import Optional, Tuple

# Set up logger
logger = logging.getLogger(__name__)

# List of audio formats that are known to work well with pyannote diarization
SUPPORTED_DIARIZATION_FORMATS = ['.wav', '.flac', '.mp3']

def detect_audio_format(file_path: str) -> str:
    """
    Detect the audio format from the file extension.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        The file extension (lowercase) including the dot
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def is_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is available on the system
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        # Check if ffmpeg is installed
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               encoding='utf-8')
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Error checking for ffmpeg: {str(e)}")
        return False

def convert_audio_to_wav(input_file: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Convert an audio file to WAV format using ffmpeg
    
    Args:
        input_file: Path to the input audio file
        output_dir: Directory to save the converted file (defaults to same directory)
        
    Returns:
        Path to the converted WAV file or None if conversion failed
    """
    if not is_ffmpeg_available():
        logger.warning("ffmpeg is not available. Cannot convert audio file.")
        logger.warning("Please install ffmpeg and make sure it's in your PATH")
        return None
    
    try:
        # If no output directory specified, use the same directory as the input file
        if not output_dir:
            output_dir = os.path.dirname(input_file)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}.converted.wav")
        
        # Run ffmpeg to convert the file
        logger.info(f"Converting {input_file} to WAV format")
        print(f"Converting {os.path.basename(input_file)} to WAV format for compatibility...")
        
        # Use subprocess to call ffmpeg
        result = subprocess.run([
            'ffmpeg', 
            '-i', input_file,  # Input file
            '-acodec', 'pcm_s16le',  # Convert to PCM WAV
            '-ar', '44100',  # Sample rate
            '-ac', '1',  # Mono audio
            '-y',  # Overwrite output file if it exists
            output_file
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            logger.error(f"Error converting audio: {result.stderr}")
            return None
            
        logger.info(f"Successfully converted audio to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error converting audio file: {str(e)}")
        return None

def prepare_audio_for_diarization(audio_file: str) -> Tuple[str, bool]:
    """
    Prepare an audio file for diarization by converting it to a supported format if needed
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        A tuple containing:
        - Path to the prepared audio file 
        - Boolean indicating if a temporary file was created (needs cleanup)
    """
    # Check if the format is already supported
    audio_format = detect_audio_format(audio_file)
    
    if (audio_format in SUPPORTED_DIARIZATION_FORMATS):
        logger.info(f"Audio format {audio_format} is already supported")
        return audio_file, False
    
    # Convert to WAV if needed
    logger.info(f"Audio format {audio_format} needs conversion for diarization")
    logger.info(f"Attempting to convert {audio_file} to WAV format for diarization")
    converted_file = convert_audio_to_wav(audio_file)
    
    if converted_file:
        logger.info(f"Using converted audio file for diarization: {converted_file}")
        return converted_file, True
    else:
        # If conversion failed, return the original file and let diarization try to handle it
        logger.warning(f"Conversion failed, using original file: {audio_file}")
        logger.warning(f"This may cause diarization to fail - check if ffmpeg is installed correctly")
        return audio_file, False

def cleanup_temp_audio(temp_file: str) -> None:
    """
    Clean up a temporary audio file
    
    Args:
        temp_file: Path to the temporary file to delete
    """
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            logger.info(f"Removed temporary audio file: {temp_file}")
        except Exception as e:
            logger.warning(f"Could not remove temporary file {temp_file}: {str(e)}")