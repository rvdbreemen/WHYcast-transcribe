"""
Whisper transcription step for the WHYcast pipeline (ADR-008).

Extracted verbatim from transcribe.py (setup_model, transcribe_audio,
write_transcript_files, format_timestamp); print() calls are routed through
whycast.events.emit per ADR-008.
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import torch

# CTranslate2 (behind faster_whisper) loads the CUDA 12 runtime at import time;
# register the pip-installed CUDA libs first or it cannot find cublas64_12.dll.
from whycast.cuda_setup import ensure_cuda_libs

ensure_cuda_libs()

from faster_whisper import WhisperModel  # noqa: E402  (must follow ensure_cuda_libs)

from whycast._deps import tqdm
from whycast.config import BEAM_SIZE, MODEL_SIZE, USE_CUSTOM_VOCABULARY, VOCABULARY_FILE
from whycast.events import emit
from whycast.io_utils import atomic_write_text
from whycast.pipeline.gpu import force_cuda_device

logger = logging.getLogger(__name__)


def setup_model(
    model_size: str = MODEL_SIZE,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> WhisperModel:
    """
    Initialize and return the Whisper model with optimal GPU batch processing.

    Uses BatchedInferencePipeline for true GPU utilization through batching rather than
    excessive worker threads. This is the correct approach for maximizing GPU throughput.

    Args:
        model_size: The model size to use (default from config)
        device: Force specific device (overrides auto-detection)
        compute_type: Force specific compute type (overrides auto-selection)
        batch_size: Batch size for GPU inference (auto-calculated if None)

    Returns:
        The initialized BatchedInferencePipeline (GPU) or WhisperModel (CPU) with optimal settings    """
    logging.info("=== Whisper Model Setup with GPU Optimization ===")

    try:
        from faster_whisper import WhisperModel

        # Force CUDA device selection for maximum GPU utilization
        target_device = force_cuda_device() if device is None else device

        # For faster-whisper, we need to use "cuda" instead of "cuda:0"
        if target_device.startswith("cuda:"):
            target_device = "cuda"

        # Auto-select compute type based on device if not specified
        if compute_type is None:
            if target_device.startswith("cuda"):
                compute_type = "float16"  # Optimal for GPU
            else:
                compute_type = "int8"     # Optimal for CPU

        # Auto-calculate optimal batch size if not specified
        if batch_size is None and target_device.startswith("cuda") and torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

                # Calculate optimal batch size based on GPU memory and model size
                if model_size in ["large-v3", "large-v2", "large"]:
                    if gpu_mem >= 20:
                        batch_size = 16
                    elif gpu_mem >= 12:
                        batch_size = 8
                    elif gpu_mem >= 8:
                        batch_size = 4
                    else:
                        batch_size = 2
                elif model_size in ["medium"]:
                    if gpu_mem >= 16:
                        batch_size = 24
                    elif gpu_mem >= 8:
                        batch_size = 16
                    else:
                        batch_size = 8
                else:  # small, base, tiny
                    if gpu_mem >= 8:
                        batch_size = 32
                    else:
                        batch_size = 16

                logging.info(f"Auto-calculated batch size: {batch_size} for {model_size} on {gpu_mem:.1f}GB GPU")
            except Exception as e:
                logging.warning(f"Failed to auto-calculate batch size: {e}, using default")
                batch_size = 8
        elif batch_size is None:
            batch_size = 1  # CPU fallback

        # Configure GPU optimizations
        if target_device.startswith("cuda") and torch.cuda.is_available():
            # Clear GPU memory before model loading
            torch.cuda.empty_cache()

            # PyTorch optimizations for GPU inference
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True

            logging.info(f"GPU optimizations enabled for {torch.cuda.get_device_name(0)}")
          # Initialize Whisper model
        logging.info(f"Loading Whisper {model_size} model on {target_device} with {compute_type}")

        model = WhisperModel(
            model_size,
            device=target_device,
            compute_type=compute_type,
            cpu_threads=4 if target_device == "cpu" else 0,
            num_workers=4  # Keep workers reasonable to avoid CPU bottleneck
        )

        # Log successful initialization
        if target_device.startswith("cuda"):
            logging.info(f"✅ Whisper model loaded on GPU with {compute_type} precision")
            emit("transcription", f"✅ Whisper model loaded on GPU ({target_device}) with {compute_type} precision")
        else:
            logging.warning(f"⚠️ Whisper model loaded on CPU with {compute_type} precision (slower)")
            emit("transcription", f"⚠️ Whisper model loaded on CPU (slower). Consider GPU setup for better performance.")

        # Store batch_size as an attribute for transcription functions
        model.optimal_batch_size = batch_size

        return model

    except Exception as e:
        logging.error(f"❌ Error setting up Whisper model: {str(e)}")
        logging.warning("Falling back to CPU model with basic settings")
        emit("transcription", f"❌ Whisper model setup failed: {str(e)}")
        emit("transcription", "🔄 Falling back to CPU model...")

        # Clear any GPU memory that might be allocated
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        # Fallback to CPU with safe settings
        try:
            fallback_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=4
            )
            fallback_model.optimal_batch_size = 1
            logging.info("✅ Fallback CPU model initialized successfully")
            emit("transcription", "✅ Fallback CPU model loaded successfully")
            return fallback_model

        except Exception as fallback_error:
            logging.error(f"❌ Even CPU fallback failed: {fallback_error}")
            emit("transcription", f"❌ Critical error: Even CPU fallback failed: {fallback_error}")
            raise RuntimeError(f"Failed to initialize any Whisper model: {fallback_error}")

def transcribe_audio(model: WhisperModel, audio_file: str, speaker_segments: Optional[List[Dict]] = None) -> Tuple[List, object]:
    """
    Transcribe audio file using the Whisper model with optimal GPU batching.

    Args:
        model: Initialized WhisperModel or BatchedInferencePipeline instance
        audio_file: Path to the audio file to transcribe
        speaker_segments: Optional list of speaker segments from diarization

    Returns:
        Tuple of (segments, info)
    """
    import os  # Actually importing the os module

    logging.info(f"Transcribing audio file: {audio_file}")
    emit("transcription", f"\nStart transcriptie van {os.path.basename(audio_file)}...")
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

            # Read replacements for post-processing
            for original, replacement in vocabulary.items():
                word_replacements[original.lower()] = replacement
                # Also add the replacement to the word_list
                word_list.append(replacement)

            if word_list:
                logging.info(f"Loaded {len(word_list)} custom vocabulary words")
                emit("transcription", f"Custom vocabulary loaded with {len(word_list)} words")
        except Exception as e:
            logging.error(f"Error loading vocabulary: {str(e)}")

    # Get optimal batch size from the model
    batch_size = getattr(model, 'optimal_batch_size', 8)

    # Check if we're using BatchedInferencePipeline
    is_batched = hasattr(model, 'model')  # BatchedInferencePipeline wraps a model

    if is_batched:
        logging.info(f"Using BatchedInferencePipeline with batch_size={batch_size} for maximum GPU utilization")
        emit("transcription", f"🚀 GPU Batch Processing: batch_size={batch_size}")
    else:
        logging.info(f"Using standard model with batch_size={batch_size}")
      # Create basic transcription parameters (common to both models)
    base_transcription_params = {
        'beam_size': BEAM_SIZE,
        'word_timestamps': True,  # Enable word timestamps for better alignment with diarization
        'vad_filter': True,       # Filter out non-speech parts
        'vad_parameters': dict(min_silence_duration_ms=500),  # Configure VAD for better accuracy
        'initial_prompt': "This is a podcast transcription.",
        'condition_on_previous_text': True,
    }
      # Create transcription parameters (without batch_size for now)
    transcription_params = {
        'beam_size': BEAM_SIZE,
        'word_timestamps': True,  # Enable word timestamps for better alignment with diarization
        'vad_filter': True,       # Filter out non-speech parts
        'vad_parameters': dict(min_silence_duration_ms=500),  # Configure VAD for better accuracy
        'initial_prompt': "This is a podcast transcription.",
        'condition_on_previous_text': True,
    }

    # Note: batch_size is stored in model.optimal_batch_size but not used in transcribe() call
    # because the current faster-whisper version doesn't support it in WhisperModel.transcribe()
    logging.info(f"Using model with optimal_batch_size={batch_size} (for future use)")

    # Track the current speaker to avoid repeating speaker tags
    current_speaker = None

    # Define a function to determine the speaker based on diarization segments
    def find_speaker_for_segment(segment_middle, speaker_segments):
        if not speaker_segments:
            return None
        # speaker_segments is expected to be a pyannote Annotation
        # Find the segment that contains segment_middle
        for speech_turn in speaker_segments.itertracks(yield_label=True):
            segment, _, label = speech_turn
            if segment.start <= segment_middle <= segment.end:
                return label
        return None

    # Define a function for live output
    def process_segment(segment):
        text = segment.text

        # Apply word replacements to each segment.
        # The replacement is a function, not a template string, for the same
        # reason as in whycast.pipeline.vocabulary.apply_vocabulary_corrections:
        # these values come from a hand-edited vocabulary.json, and re.sub would
        # otherwise read "\1" as a group reference (raising here, mid-loop,
        # after the GPU work is already paid for) and "C:\temp" as an escape.
        if word_replacements:
            for original, replacement in word_replacements.items():
                # Replace whole words with a case-insensitive match
                pattern = r'\b' + re.escape(original) + r'\b'
                text = re.sub(
                    pattern,
                    lambda _match, value=replacement: value,
                    text,
                    flags=re.IGNORECASE,
                )

        # Add speaker info if available
        speaker_info = ""
        nonlocal current_speaker

        if speaker_segments:
            # Use the middle of the segment to determine the speaker
            segment_middle = (segment.start + segment.end) / 2
            speaker = find_speaker_for_segment(segment_middle, speaker_segments)

            # Always show the speaker, not only when switching
            if speaker:
                speaker_info = f"[{speaker}] "
                current_speaker = speaker
            else:
                speaker_info = "[SPEAKER_UNKNOWN] "

        # Display the text directly in the console
        emit("transcription", f"{format_timestamp(segment.start)} {speaker_info}{text}")

        # Apply the text to the segment
        segment.text = text
        return segment

    # Function to collect segments with real-time processing
    def collect_with_live_output(segments_generator):
        emit("transcription", "\n--- Live Transcription Output ---")
        result = []
        for segment in segments_generator:
            # Process each segment for replacements and logging
            segment = process_segment(segment)
            result.append(segment)
        emit("transcription", "--- End Live Transcription ---\n")
        return result

    # Execute transcription with appropriate parameters
    try:
        # Add word_list only if it is defined and not empty
        if word_list:
            try:
                # Try with word_list
                emit("transcription", "Running transcription with custom vocabulary...")
                segments_generator, info = model.transcribe(
                    audio_file,
                    **transcription_params,
                    word_list=word_list
                )
                segments_list = collect_with_live_output(segments_generator)
            except TypeError as e:
                # If word_list is not supported, try without it
                logging.warning(f"word_list parameter not supported in this version of faster_whisper: {e}")
                logging.info("Running transcription without custom vocabulary")
                emit("transcription", "word_list not supported, running transcription without custom vocabulary...")
                segments_generator, info = model.transcribe(
                    audio_file,
                    **transcription_params
                )
                segments_list = collect_with_live_output(segments_generator)
        else:
            # If there is no word_list, use the default parameters
            emit("transcription", "Running transcription...")
            segments_generator, info = model.transcribe(
                audio_file,
                **transcription_params
            )
            segments_list = collect_with_live_output(segments_generator)

    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        raise

    # Calculate and display processing time
    elapsed_time = time.time() - start_time
    emit("transcription", f"\nTranscription completed in {elapsed_time:.1f} seconds.")
    emit("transcription", f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    emit("transcription", f"Segment count: {len(segments_list)}")

    logging.info(f"Transcription complete: {len(segments_list)} segments")
    return segments_list, info

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
    # Local function to find speaker for a segment midpoint
    def find_speaker_for_segment(segment_middle, speaker_segments):
        if not speaker_segments:
            return None
        for speech_turn in speaker_segments.itertracks(yield_label=True):
            segment, _, label = speech_turn
            if segment.start <= segment_middle <= segment.end:
                return label
        return None

    try:
        # Prepare for writing the transcripts
        full_transcript = []
        timestamped_transcript = []
        # Track current speaker to avoid repeating speaker tags for consecutive segments
        current_speaker = None
        emit("transcription", f"\nCreating transcript files...")
        emit("transcription", f"- Clean transcript: {os.path.basename(output_file)}")
        emit("transcription", f"- Timestamped transcript: {os.path.basename(output_file_timestamped)}")
        # Process each segment from Whisper
        for i, segment in enumerate(tqdm(segments, desc="Processing transcript", unit="segment")):
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            if not text:  # Skip empty segments
                continue
            # Calculate segment duration for better speaker detection of short segments
            segment_duration = end - start
            # Format timestamp for the timestamped version
            timestamp = format_timestamp(start)
            # Get speaker info if available
            speaker_info = ""
            speaker_prefix = ""
            if speaker_segments:
                # Use middle of segment to determine speaker
                middle_time = (start + end) / 2
                speaker = find_speaker_for_segment(middle_time, speaker_segments)
                if speaker:
                    is_short_utterance = segment_duration < 1.0 and len(text.split()) <= 5
                    # For very short utterances with no clear speaker, try to maintain speaker continuity
                    if is_short_utterance and not speaker and current_speaker:
                        speaker = current_speaker
                    if speaker:
                        speaker_info = f"[{speaker}] "
                        speaker_prefix = f"[{speaker}] "
                        current_speaker = speaker
                    else:
                        speaker_info = "[SPEAKER_UNKNOWN] "
                        speaker_prefix = "[SPEAKER_UNKNOWN] "
                        current_speaker = None
                else:
                    speaker_info = "[SPEAKER_UNKNOWN] "
                    speaker_prefix = "[SPEAKER_UNKNOWN] "
                    current_speaker = None
            # Add to transcript collections
            clean_line = f"{speaker_prefix}{text}"
            timestamped_line = f"{timestamp} {speaker_info}{text}"
            full_transcript.append(clean_line)
            timestamped_transcript.append(timestamped_line)
            # Store the end time for checking continuity in the next iteration
            prev_end = end
        # Join all lines
        full_text = "\n".join(full_transcript)
        timestamped_text = "\n".join(timestamped_transcript)

        # Write to files. Atomic (ADR-008): cancelling a job or a CUDA OOM kill
        # must never leave a truncated transcript that looks complete to the
        # episode scanner. Byte-for-byte the same output as the old
        # open(..., 'w', encoding='utf-8').
        # backup=False at every artifact call site (the io_utils default stays
        # True): a re-run of the archive would leave a .bak beside every
        # artifact. The atomic replace already guarantees no partial file; the
        # backup only kept the previous version.
        atomic_write_text(output_file, full_text, backup=False)

        atomic_write_text(
            output_file_timestamped, timestamped_text, backup=False
        )

        emit("transcription", f"Transcript files successfully created:")
        emit("transcription", f"- {output_file}")
        emit("transcription", f"- {output_file_timestamped}")

        return full_text
    except Exception as e:
        logging.error(f"Error writing transcript files: {str(e)}")
        return ""

def format_timestamp(start: float, end: Optional[float] = None) -> str:
    """
    Format a timestamp in seconds to HH:MM:SS format.

    Args:
        start: Start timestamp in seconds
        end: Optional end timestamp in seconds

    Returns:
        Formatted timestamp string
    """
    def _format_single(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

    if end is None:
        return _format_single(start)
    else:
        return f"[{_format_single(start)} --> {_format_single(end)}]"
