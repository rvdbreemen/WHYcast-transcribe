"""
Utilities for audio transcription using faster-whisper.
"""

import sys
import logging
from typing import Tuple
from datetime import datetime
from faster_whisper import WhisperModel

# Import configuration variables
try:
    from config import (
        MODEL_SIZE, MODEL_PATH, DEVICE, COMPUTE_TYPE, BEAM_SIZE
    )
except ImportError as e:
    raise ImportError(f"Error importing transcription configuration: {e}")

def is_cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Log CUDA device information for better diagnostics
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            logging.info(f"CUDA is available: {device_count} device(s) - {device_name}")
        return cuda_available
    except ImportError:
        return False

def setup_model(model_size: str = MODEL_SIZE) -> WhisperModel:
    """
    Initialize and return the Whisper model.
    Automatically uses CUDA if available with optimized settings.
    
    Args:
        model_size: The model size to use
    
    Returns:
        The initialized WhisperModel
    """
    try:
        import torch
        
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
            compute_type = "float16"
        else:
            device = "cpu"
            num_workers = 8  # More workers for CPU
            compute_type = "int8"
            logging.warning("CUDA is not available - using CPU which may be significantly slower")
        
        # Create model with optimized parameters
        # If model_size is None or matches MODEL_SIZE, use MODEL_PATH
        # Otherwise use model_size directly as the model name/path
        if model_size is None or model_size == MODEL_SIZE:
            model_path = MODEL_PATH
            logging.info(f"Using configured model path: {model_path}")
        else:
            model_path = model_size
            logging.info(f"Using specified model size: {model_path}")
        
        model = WhisperModel(
            model_path, 
            device=device, 
            compute_type=compute_type,
            cpu_threads=8 if device == "cpu" else 0,
            num_workers=num_workers
        )
        
        logging.info(f"Initialized {'default' if model_size is None else model_size} model on {device} with compute type {compute_type}")
        return model
        
    except Exception as e:
        logging.error(f"Error setting up model: {str(e)}")
        # Fallback to CPU if CUDA setup fails
        logging.warning("Falling back to CPU model")
        try:
            # Try with direct model path for fallback as well
            if model_size is None or model_size == MODEL_SIZE:
                model_path = MODEL_PATH
            else:
                model_path = model_size
                
            return WhisperModel(
                model_path,
                device="cpu",
                compute_type="int8",
                cpu_threads=8,
                num_workers=4
            )
        except Exception as e2:
            logging.error(f"Critical error in fallback: {str(e2)}")
            raise

def transcribe_audio(model: WhisperModel, audio_file: str) -> Tuple:
    """
    Transcribe an audio file using the provided model with optimized settings.
    
    Args:
        model: The WhisperModel to use
        audio_file: Path to the audio file
        
    Returns:
        Tuple of (segments, info)
    """
    logging.info(f"Transcribing {audio_file}...")
    
    # Monitor GPU memory before transcription if possible
    try:
        if model.device == "cuda":
            import torch
            before_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            logging.info(f"GPU memory in use before transcription: {before_mem:.2f} GB")
    except Exception:
        pass
    
    start_time = datetime.now()
    
    # Use optimized transcription parameters
    result = model.transcribe(
        audio_file,
        beam_size=BEAM_SIZE,
        best_of=5,         # Consider more candidates for better results
        vad_filter=True,   # Voice activity detection to skip silence
        vad_parameters={"min_silence_duration_ms": 500},  # Adjust silence detection
        initial_prompt=None,  # Can set an initial prompt if needed for better context
        condition_on_previous_text=True,  # Use previous text as context
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Temperature fallback for difficult audio
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    audio_duration = get_audio_duration(audio_file)
    if audio_duration > 0:
        speed_factor = audio_duration / duration
        logging.info(f"Transcription completed in {duration:.2f} seconds (audio length: {audio_duration:.2f}s, {speed_factor:.2f}x real-time speed)")
    else:
        logging.info(f"Transcription completed in {duration:.2f} seconds")
    
    # Monitor GPU memory after transcription
    try:
        if model.device == "cuda":
            import torch
            after_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            logging.info(f"GPU memory in use after transcription: {after_mem:.2f} GB")
            # Clear cache immediately after transcription to free memory
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    return result

def get_audio_duration(audio_file: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Duration in seconds or 0 if cannot be determined
    """
    try:
        import librosa
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception:
        # Try with pydub if librosa fails
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_file)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception:
            return 0  # Return 0 if duration cannot be determined

def cleanup_resources(model=None):
    """
    Clean up resources after processing.
    More aggressive memory cleanup to prevent memory leaks.
    
    Args:
        model: The WhisperModel instance to clean up (if provided)
    """
    try:
        # Clear CUDA cache if available
        import torch
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