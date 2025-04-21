from faster_whisper import WhisperModel
import torch
import logging
import os
from typing import Optional, Tuple, Any

from whycast_transcribe.config import (
    MODEL_SIZE, DEVICE, COMPUTE_TYPE, BEAM_SIZE,
    USE_SPEAKER_DIARIZATION, DIARIZATION_MODEL,
    DIARIZATION_ALTERNATIVE_MODEL, DIARIZATION_MIN_SPEAKERS,
    DIARIZATION_MAX_SPEAKERS
)
from whycast_transcribe.utils.device_utils import get_best_device


def setup_whisper_model(model_size: str = None) -> WhisperModel:
    """
    Initialize and return a WhisperModel using faster-whisper.
    """
    size = model_size or MODEL_SIZE
    # Determine device
    device = get_best_device()
    # Optionally, adjust compute_type for device
    compute_type = COMPUTE_TYPE
    if device == "mps":
        compute_type = "float16"  # Recommended for MPS
    logging.info(f"Initializing Whisper model (size={size}) on device={device} with compute_type={compute_type}")
    model = WhisperModel(
        size,
        device=device,
        compute_type=compute_type
    )
    logging.info(f"Whisper model initialized successfully on {device}")
    return model


def setup_diarization_pipeline():
    """
    Initialize and return a PyAnnote diarization pipeline or None if disabled or unavailable.
    """
    if not USE_SPEAKER_DIARIZATION:
        logging.info("Speaker diarization is disabled via configuration")
        return None
    
    # Check for Hugging Face token (try both common variable names)
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if not hf_token:
        logging.warning("Neither HF_TOKEN nor HUGGINGFACE_TOKEN environment variable is set. Speaker diarization requires authentication.")
        logging.warning("Set either HF_TOKEN or HUGGINGFACE_TOKEN environment variable to your Hugging Face token.")
        return None
        
    logging.info(f"Attempting to load diarization model: {DIARIZATION_MODEL}")
    try:
        from pyannote.audio import Pipeline
        # Attempt to load primary diarization model with explicit token
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
        logging.info("Primary diarization model loaded successfully")
    except Exception as e:
        logging.warning(f"Speaker diarization unavailable or failed: {e}")
        try:
            # Try alternative model if primary fails
            if DIARIZATION_ALTERNATIVE_MODEL:
                logging.info(f"Attempting to load alternative diarization model: {DIARIZATION_ALTERNATIVE_MODEL}")
                pipeline = Pipeline.from_pretrained(DIARIZATION_ALTERNATIVE_MODEL, use_auth_token=hf_token)
                logging.info("Alternative diarization model loaded successfully")
                return pipeline
        except Exception as e2:
            logging.error(f"Alternative diarization model also failed: {e2}")
        return None

    logging.info("Configuring diarization pipeline for CUDA if available")
    try:
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device('cuda'))
            logging.info("Diarization pipeline moved to CUDA")
    except Exception as e:
        logging.warning(f"Failed to move diarization to CUDA: {e}")
    
    return pipeline


def setup_models(model_size: str = None, enable_diarization: bool = None) -> Tuple[WhisperModel, Optional[Any]]:
    """
    Initialize and return both the WhisperModel and diarization pipeline if enabled.
    
    Args:
        model_size: Size of the Whisper model to use, defaults to config value
        enable_diarization: Whether to enable diarization, defaults to config value
    
    Returns:
        Tuple of (whisper_model, diarization_pipeline)
    """
    # Setup Whisper model
    model = setup_whisper_model(model_size)
    
    # Determine if diarization should be enabled
    use_di = enable_diarization if enable_diarization is not None else USE_SPEAKER_DIARIZATION
    
    # Setup diarization pipeline if enabled
    diarization_pipeline = setup_diarization_pipeline() if use_di else None
    
    return model, diarization_pipeline