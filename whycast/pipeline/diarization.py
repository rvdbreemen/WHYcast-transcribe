"""
Speaker diarization step for the WHYcast pipeline (ADR-008).

Extracted verbatim from transcribe.py (set_huggingface_token,
get_huggingface_token, diarize_audio). The only transformation applied is
print(X) -> emit("diarization", X); all other logic is unchanged.
pyannote/torch imports stay inside the function bodies exactly as in the
original.

One deliberate deviation: set_huggingface_token anchors the .env path to the
repository root via whycast.config.base_dir instead of os.path.dirname(
__file__), so the token lands in the same .env that load_dotenv() reads.
"""

import logging
import os
import re
from typing import Optional

from whycast.config import (
    DIARIZATION_ALTERNATIVE_MODEL,
    DIARIZATION_MAX_SPEAKERS,
    DIARIZATION_MIN_SPEAKERS,
    DIARIZATION_MODEL,
    USE_SPEAKER_DIARIZATION,
    base_dir,
)
from whycast.errors import ConfigurationError
from whycast.events import emit
from whycast.io_utils import atomic_write_text

logger = logging.getLogger(__name__)


def set_huggingface_token(token: str) -> bool:
    """
    Set the HuggingFace token in the .env file.
    
    Args:
        token: The HuggingFace API token
        
    Returns:
        True if token was set successfully, False otherwise
    """
    try:
        # Anchored to the repo root via whycast.config.base_dir: in transcribe.py
        # __file__ was the repo root, but after extraction it would resolve to
        # whycast/pipeline/, writing a .env that load_dotenv() never finds.
        env_file = os.path.join(base_dir, '.env')
        
        # Read current .env file content
        content = ""
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Check if HUGGINGFACE_TOKEN already exists in the file
        if 'HUGGINGFACE_TOKEN=' in content:
            # Replace existing token
            pattern = r'HUGGINGFACE_TOKEN=.*'
            replacement = f'HUGGINGFACE_TOKEN={token}'
            content = re.sub(pattern, replacement, content)
        else:
            # Add new token at the end
            if content and not content.endswith('\n'):
                content += '\n'
            content += f'HUGGINGFACE_TOKEN={token}\n'
        
        # Write back to .env file.
        #
        # Atomically (ADR-008), and this is the one file where it matters most.
        # A plain open(..., 'w') truncates first and writes second, so any
        # failure in between - a full disk, a kill - leaves a zero-byte .env,
        # taking OPENAI_API_KEY and PODCAST_FEED_URL with it, while this
        # function returns False and logs a line about a token. ADR-009 rightly
        # forbids a .env.bak (a backup of a secrets file is a second copy of
        # the secrets), which leaves atomicity as the only protection this file
        # can have: write a temp file next to it, fsync, rename. The rename
        # either happens or it does not.
        atomic_write_text(env_file, content, backup=False)


        # Set in current environment
        os.environ['HUGGINGFACE_TOKEN'] = token
        
        logging.info("HuggingFace token has been set successfully")
        return True
    except Exception as e:
        logging.error(f"Error setting HuggingFace token: {str(e)}")
        return False

def get_huggingface_token(ask_if_missing: bool = True) -> Optional[str]:
    """
    Get the HuggingFace token from environment. Optionally prompt the user if it's missing.
    
    Args:
        ask_if_missing: Whether to prompt the user for the token if it's missing
        
    Returns:
        The token or None if not available
    """
    token = os.environ.get('HUGGINGFACE_TOKEN')
    
    if not token and ask_if_missing:
        try:
            logging.info("HuggingFace token is required for speaker diarization")
            logging.info("You can get one from https://huggingface.co/settings/tokens")
            emit("diarization", "\nPlease enter your HuggingFace token (will be saved to .env file):")
            token = input().strip()
            
            if token:
                set_huggingface_token(token)
            else:
                logging.warning("No token provided. Speaker diarization will be disabled.")
                return None
        except Exception as e:
            logging.error(f"Error getting token from user: {str(e)}")
            return None
            
    return token

def diarize_audio(waveform=None, sample_rate=None, audio_file_path=None, hf_token=None):
    """
    Run speaker diarization using pyannote.audio 3.1 pipeline on GPU if available, using in-memory audio.
    This function implements GPU acceleration for pyannote speaker diarization.
    
    Args:
        waveform: torch.Tensor of shape (channels, samples), optional
        sample_rate: int, optional
        audio_file_path: str, optional, used if waveform/sample_rate not provided
        hf_token: str, Hugging Face access token
    Returns:
        diarization result (pyannote Annotation) or None on failure
    """
    if not USE_SPEAKER_DIARIZATION:
        # The switch existed in config.py and in the web UI config viewer and
        # was read by nothing, so diarization always ran. Turning it off now
        # means the transcript carries no speaker labels at all, which is what
        # every caller already handles (speaker_segments=None).
        emit("diarization", "Speaker diarization disabled (USE_SPEAKER_DIARIZATION)")
        logging.info("Speaker diarization disabled by configuration")
        return None

    try:
        import torch
        import torchaudio
        from pyannote.audio import Pipeline
        
        if hf_token is None:
            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
          # Check CUDA availability and force GPU usage with MAXIMUM utilization
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.device("cuda")
            emit("diarization", "🚀 FORCING MAXIMUM CUDA UTILIZATION for diarization")
            logging.info("FORCING MAXIMUM CUDA UTILIZATION for diarization")
            
            # Clear GPU cache before starting
            torch.cuda.empty_cache()
            
            # AGGRESSIVE GPU optimizations for pyannote
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Set aggressive memory allocation for pyannote
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% for diarization
            except:
                pass
            
            # Log GPU information with enhanced details
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = gpu_props.total_memory / (1024**3)
            multiprocessors = gpu_props.multi_processor_count
            
            emit("diarization", f"🎯 MAXIMUM GPU UTILIZATION: {gpu_name} ({gpu_memory:.1f} GB, {multiprocessors} MPs)")
            logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB, {multiprocessors} multiprocessors)")
            
        else:
            device = torch.device("cpu")
            emit("diarization", "⚠️  CUDA not available for diarization - using CPU (slow)")
            logging.warning("CUDA not available for diarization - using CPU (slow)")
        
        # Load the diarization pipeline. The model comes from config, not from
        # a literal here: DIARIZATION_MODEL was defined, shown in the web UI
        # config viewer and read by nothing, so changing it did nothing at all.
        emit("diarization", f"Loading {DIARIZATION_MODEL} pipeline...")
        logging.info(f"Loading {DIARIZATION_MODEL} pipeline")

        def _load(model_id):
            """Load a pipeline, turning pyannote's silent None into an error.

            Pipeline.from_pretrained does NOT raise when the repository is
            missing, private or gated: it prints a hint and returns None
            (pyannote/audio/core/pipeline.py:107-121, where RepositoryNotFoundError
            - and therefore GatedRepoError - is caught). That is the failure
            ADR-002 names as its risk, so a fallback that only catches
            exceptions would never fire for the case it was written for.
            """
            loaded = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
            if loaded is None:
                raise ConfigurationError(
                    f"Could not load the diarization pipeline '{model_id}'. It is "
                    "missing, private, or gated. Accept the conditions on "
                    f"https://hf.co/{model_id} and set a Hugging Face token."
                )
            return loaded

        try:
            pipeline = _load(DIARIZATION_MODEL)
        except Exception as primary_error:
            # ADR-002 names DIARIZATION_ALTERNATIVE_MODEL as the mitigation for
            # pyannote drifting against torch. Nothing read it, so the mitigation
            # the ADR claims did not exist. It does now, and ships unset: the
            # value must name a diarization PIPELINE, and anything else (a
            # segmentation model, say) can only raise. When it is unset this
            # re-raises the ORIGINAL error, which is the one worth reading.
            if not DIARIZATION_ALTERNATIVE_MODEL or DIARIZATION_ALTERNATIVE_MODEL == DIARIZATION_MODEL:
                raise
            logging.warning(
                "Could not load %s (%s); trying DIARIZATION_ALTERNATIVE_MODEL %s",
                DIARIZATION_MODEL, primary_error, DIARIZATION_ALTERNATIVE_MODEL,
            )
            emit("diarization", f"⚠️  {DIARIZATION_MODEL} failed to load, trying {DIARIZATION_ALTERNATIVE_MODEL}")
            try:
                pipeline = _load(DIARIZATION_ALTERNATIVE_MODEL)
            except Exception as fallback_error:
                logging.error(
                    "DIARIZATION_ALTERNATIVE_MODEL %s also failed (%s). It must name a "
                    "diarization PIPELINE - a repo whose config.yaml has a 'pipeline' "
                    "key - not a segmentation model such as pyannote/segmentation-3.0.",
                    DIARIZATION_ALTERNATIVE_MODEL, fallback_error,
                )
                raise primary_error

        # FORCE the pipeline to use GPU if available
        if cuda_available:
            emit("diarization", "Moving diarization pipeline to GPU...")
            logging.info("Moving diarization pipeline to GPU")
            pipeline.to(device)
            
            # Verify pipeline is on GPU
            logging.info(f"Pipeline device: {device}")
            emit("diarization", f"✅ Diarization pipeline using: {device}")
        else:
            emit("diarization", f"✅ Diarization pipeline using: {device}")
        
        # Load audio if not provided in memory
        if waveform is None or sample_rate is None:
            if audio_file_path is None:
                raise ValueError('No audio data or file path provided for diarization.')
            emit("diarization", "Loading audio for diarization...")
            logging.info(f"Loading audio from: {audio_file_path}")
            waveform, sample_rate = torchaudio.load(audio_file_path)
        
        # Move waveform to GPU if using CUDA
        if cuda_available:
            emit("diarization", "Moving audio waveform to GPU...")
            logging.info("Moving audio waveform to GPU")
            waveform = waveform.to(device)
            logging.info(f"Waveform device: {waveform.device}")
        
        # Run diarization with progress indication
        emit("diarization", "Running speaker diarization...")
        logging.info("Starting diarization inference")
        
        # Monitor GPU memory if using CUDA
        if cuda_available:
            memory_before = torch.cuda.memory_allocated(0) / 1024**2
            logging.info(f"GPU memory before diarization: {memory_before:.2f} MB")
        
        # Run the actual diarization. min/max speakers bound the clustering:
        # without them pyannote decides the speaker count on its own, which on
        # episode 1 produced six numbered labels for what is probably four
        # people. Both were configurable and neither was ever passed.
        speaker_bounds = {}
        if DIARIZATION_MIN_SPEAKERS > 1:
            speaker_bounds["min_speakers"] = DIARIZATION_MIN_SPEAKERS
        if DIARIZATION_MAX_SPEAKERS:
            speaker_bounds["max_speakers"] = DIARIZATION_MAX_SPEAKERS
        if (
            "min_speakers" in speaker_bounds
            and "max_speakers" in speaker_bounds
            and speaker_bounds["min_speakers"] > speaker_bounds["max_speakers"]
        ):
            # Two env-vars that contradict each other. Passing them on gets an
            # error from deep inside pyannote that names neither setting.
            raise ConfigurationError(
                f"DIARIZATION_MIN_SPEAKERS ({DIARIZATION_MIN_SPEAKERS}) is greater "
                f"than DIARIZATION_MAX_SPEAKERS ({DIARIZATION_MAX_SPEAKERS})."
            )
        if speaker_bounds:
            logging.info(f"Diarization speaker bounds: {speaker_bounds}")
        diarization = pipeline(
            {'waveform': waveform, 'sample_rate': sample_rate}, **speaker_bounds
        )
        
        # Log memory usage after processing
        if cuda_available:
            memory_after = torch.cuda.memory_allocated(0) / 1024**2
            logging.info(f"GPU memory after diarization: {memory_after:.2f} MB")
            emit("diarization", f"GPU memory used: {memory_after:.2f} MB")
            
            # Clear GPU cache after processing
            torch.cuda.empty_cache()
            memory_final = torch.cuda.memory_allocated(0) / 1024**2
            logging.info(f"GPU memory after cleanup: {memory_final:.2f} MB")
        
        emit("diarization", "✅ Speaker diarization completed successfully")
        logging.info("Diarization completed successfully")
        return diarization
        
    except Exception as e:
        logging.error(f'Diarization failed: {e}')
        emit("diarization", f"❌ Diarization failed: {e}")
        
        # Clear GPU memory even on failure
        if 'cuda_available' in locals() and cuda_available:
            try:
                torch.cuda.empty_cache()
                logging.info("GPU memory cleared after diarization failure")
            except:
                pass
        
        return None

