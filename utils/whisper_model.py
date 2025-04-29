python .\transcribe.py --force podcasts/#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Whisper model functionality for WHYcast-transcribe.

This module provides functions for setting up and using the Whisper model
for audio transcription with various optimizations.
"""

import os
import sys
import time
import json
import logging
import torch
import torchaudio
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# Import configuration
from config import (
    MODEL_SIZE, DEVICE, COMPUTE_TYPE, BEAM_SIZE,
    USE_CUSTOM_VOCABULARY, VOCABULARY_FILE
)

# Import from check_cuda for diagnostics
from utils.check_cuda import show_cuda_diagnostics


def get_audio_duration(audio_file: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Duration in seconds, or 0 if an error occurs
    """
    try:
        info = torchaudio.info(audio_file)
        return info.num_frames / info.sample_rate
    except Exception as e:
        logging.warning(f"Could not determine audio duration: {str(e)}")
        return 0.0


def setup_model(model_size: str = MODEL_SIZE) -> "WhisperModel":
    """
    Initialize and return the Whisper model.
    Automatically uses CUDA if available with optimized settings.
    
    Args:
        model_size: The model size to use
    
    Returns:
        The initialized WhisperModel
    """
    try:
        # First ensure the required modules are imported
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logging.error("faster-whisper not installed. Use pip install faster-whisper")
            sys.exit(1)
            
        # Log system information
        logging.info("=== System Information ===")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            logging.info(f"GPU device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logging.info(f"GPU {i}: {props.name}")
                logging.info(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
                logging.info(f"  CUDA capability: {props.major}.{props.minor}")
                logging.info(f"  Multi-processor count: {props.multi_processor_count}")
            
            # Optimize CUDA settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set number of workers based on GPU memory
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            if gpu_mem >= 16:  # High-end GPU
                num_workers = 4
            elif gpu_mem >= 8:  # Mid-range GPU
                num_workers = 2
            else:  # Lower-end GPU
                num_workers = 1
                
            logging.info(f"Using {num_workers} workers for data loading")
            device = "cuda"
            compute_type = COMPUTE_TYPE
            
            # Enable TensorFloat-32 on Ampere GPUs (compute capability 8.0+)
            compute_capability = torch.cuda.get_device_capability(0)
            if compute_capability[0] >= 8:  # Ampere has compute capability 8.0+
                logging.info("Enabling TensorFloat-32 for better performance on Ampere GPU")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            device = "cpu"
            num_workers = 8  # More workers for CPU
            compute_type = "int8"
            logging.warning("CUDA is not available - using CPU which may be significantly slower")
        
        # Show CUDA diagnostics
        show_cuda_diagnostics()
        
        # Create model with optimized parameters
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "whisper")
        
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            cpu_threads=8 if device == "cpu" else 0,
            num_workers=num_workers,
            download_root=models_dir
        )
        
        logging.info(f"Initialized {model_size} model on {device} with compute type {compute_type}")
        return model
        
    except Exception as e:
        logging.error(f"Error setting up model: {str(e)}")
        # Fallback to CPU if CUDA setup fails
        logging.warning("Falling back to CPU model")
        
        from faster_whisper import WhisperModel
        return WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
            num_workers=4,
            download_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       "models", "whisper")
        )


def transcribe_audio(model, audio_file: str, speaker_segments: Optional[List[Dict]] = None) -> Tuple[List, object]:
    """
    Transcribe audio file using the Whisper model.
    
    Args:
        model: Initialized WhisperModel instance
        audio_file: Path to the audio file to transcribe
        speaker_segments: Optional list of speaker segments from diarization
        
    Returns:
        Tuple of (segments, info)
    """
    logging.info(f"Transcribing audio file: {audio_file}")
    print(f"\nStart transcription of {os.path.basename(audio_file)}...")
    start_time = time.time()
    
    # Load custom vocabulary if enabled
    word_list = None
    word_replacements = {}
    if USE_CUSTOM_VOCABULARY and os.path.exists(VOCABULARY_FILE):
        try:
            with open(VOCABULARY_FILE, 'r', encoding='utf-8') as f:
                vocabulary = json.load(f)
            
            # Create a list of words for the word_list parameter
            word_list = []
            
            for original, replacement in vocabulary.items():
                word_replacements[original.lower()] = replacement
                # Also add the replacement to the word_list
                word_list.append(replacement)
            
            if word_list:
                logging.info(f"Loaded {len(word_list)} custom vocabulary words")
                print(f"Custom vocabulary loaded with {len(word_list)} words")
        except Exception as e:
            logging.error(f"Error loading vocabulary: {str(e)}")
    
    # Create basic transcription parameters
    transcription_params = {
        'beam_size': BEAM_SIZE,
        'word_timestamps': True,  # Enable word timestamps for better alignment with diarization
        'vad_filter': True,       # Filter out non-speech parts
        'vad_parameters': dict(min_silence_duration_ms=500),  # Configure VAD for better accuracy
        'initial_prompt': "This is a podcast transcription.",
        'condition_on_previous_text': True,
    }
    
    # Track current speaker to avoid repetition
    current_speaker = None
    
    # Define function for live output
    def process_segment(segment):
        text = segment.text
        
        # Apply vocabulary to each segment
        if word_replacements:
            import re
            for original, replacement in word_replacements.items():
                # Replace whole words with case-insensitive match
                pattern = r'\b' + re.escape(original) + r'\b'
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Add speaker info if available
        speaker_info = ""
        nonlocal current_speaker
        
        if speaker_segments:
            # Use the midpoint of the segment to determine speaker
            segment_middle = (segment.start + segment.end) / 2
            
            # Import here to avoid circular imports
            from utils.diarize import get_speaker_for_segment
            speaker = get_speaker_for_segment(segment_middle, speaker_segments)
            
            # Always include the speaker tag when speaker information is available
            if speaker:
                speaker_info = f"[{speaker}] "
                current_speaker = speaker
            else:
                # Mark unknown speakers
                speaker_info = "[SPEAKER_UNKNOWN] "
                current_speaker = None
        
        # Show the text directly in the console
        print(f"{format_timestamp(segment.start, segment.end)} {speaker_info} {text}")
        
        # Apply the speaker information to the segment text
        if speaker_info:
            segment.text = f"{speaker_info}{text}"
        else:
            segment.text = text
        
        return segment
    
    # Function to collect segments with real-time processing
    def collect_with_live_output(segments_generator):
        print("\n--- Live Transcription Output ---")
        result = []
        for segment in segments_generator:
            # Process each segment for replacements and logging
            segment = process_segment(segment)
            result.append(segment)
        print("--- End of Live Transcription ---\n")
        return result
    
    # Execute transcription with appropriate parameters
    try:
        # Add word_list only if it's defined and not empty
        if word_list:
            try:
                # Try with word_list
                print("Running transcription with custom vocabulary...")
                segments_generator, info = model.transcribe(
                    audio_file,
                    **transcription_params,
                    word_list=word_list
                )
                segments_list = collect_with_live_output(segments_generator)
            except TypeError as e:
                # If word_list is not supported, try without
                logging.warning(f"word_list parameter not supported in this version of faster_whisper: {e}")
                logging.info("Running transcription without custom vocabulary")
                print("word_list not supported, running transcription without custom vocabulary...")
                segments_generator, info = model.transcribe(
                    audio_file, 
                    **transcription_params
                )
                segments_list = collect_with_live_output(segments_generator)
        else:
            # If there's no word_list, use standard parameters
            print("Running transcription...")
            segments_generator, info = model.transcribe(
                audio_file,
                **transcription_params
            )
            segments_list = collect_with_live_output(segments_generator)
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        raise
    
    # Calculate and show processing time
    elapsed_time = time.time() - start_time
    audio_duration = get_audio_duration(audio_file)
    
    print(f"\nTranscription completed in {elapsed_time:.1f} seconds.")
    if audio_duration > 0:
        realtime_factor = audio_duration / elapsed_time
        print(f"Audio duration: {audio_duration:.1f} seconds (processed at {realtime_factor:.1f}x realtime)")
    
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    print(f"Number of segments: {len(segments_list)}")
    
    logging.info(f"Transcription complete: {len(segments_list)} segments")
    
    # Monitor GPU memory after transcription
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            logging.info(f"GPU memory after transcription: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
            # Clear cache immediately after transcription to free memory
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    return segments_list, info


def format_timestamp(start: float, end: Optional[float] = None) -> str:
    """
    Format a timestamp for display.
    
    Args:
        start: Start time in seconds
        end: Optional end time in seconds
        
    Returns:
        Formatted timestamp string
    """
    if end is not None:
        return f"[{format_time(start)} --> {format_time(end)}]"
    return f"[{format_time(start)}]"


def format_time(seconds: float) -> str:
    """
    Format time in seconds to hh:mm:ss format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def cleanup_resources(model) -> None:
    """
    Clean up resources after processing.
    More aggressive memory cleanup to prevent memory leaks.
    
    Args:
        model: The WhisperModel instance to clean up
    """
    try:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            # More aggressive memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Make sure all CUDA operations are complete
            logging.info("CUDA cache cleared")
            
            # Log memory stats
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            logging.info(f"GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        
        # Delete model to free up memory
        if model is not None:
            del model
            logging.info("Model resources released")
        
        # Force garbage collection
        import gc
        gc.collect()
    except Exception as e:
        logging.warning(f"Error during resource cleanup: {str(e)}")