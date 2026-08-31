"""
Full pipeline workflow for WHYcast (ADR-008).

Extracted verbatim from transcribe.py (full_workflow, lines 2565-2706) during
the ADR-008 library extraction. The only allowed transformations were applied:
print() -> emit("workflow", ...); everything else is unchanged.
"""

import logging
import os
import re

from whycast.config import USE_SPEAKER_DIARIZATION
from whycast.errors import ConfigurationError
from whycast.events import emit
from whycast.pipeline.audio import prepare_audio_for_diarization
from whycast.pipeline.diarization import diarize_audio
from whycast.pipeline.feed import delete_episode_files, podcast_fetching_workflow
from whycast.pipeline.gpu import maximize_gpu_utilization, verify_gpu_setup
from whycast.pipeline.postprocess import process_transcript_workflow
from whycast.pipeline.speakers import write_merged_transcript
from whycast.pipeline.transcription import setup_model, transcribe_audio, write_transcript_files

logger = logging.getLogger(__name__)


def _run_diarization(prepared_audio):
    """Diarize one prepared audio file, or None when it could not be done.

    Split out of full_workflow so the "switched off" path does not have to walk
    past the GPU checks and the audio load that exist only for this step.
    """
    emit("workflow", "[2/4] Running speaker diarization ...")
    verify_gpu_setup()

    try:
        import torch
        import torchaudio

        # Load audio with GPU optimization
        emit("workflow", "Loading audio for diarization...")
        waveform, sample_rate = torchaudio.load(prepared_audio)

        # Run diarization with GPU acceleration
        speaker_segments = diarize_audio(waveform=waveform, sample_rate=sample_rate)

        if speaker_segments is not None:
            # Count the number of speaker segments
            num_segments = len(list(speaker_segments.itertracks()))
            emit("workflow", f"✅ Diarization complete. Found {num_segments} speaker segments.")

            # Log speaker information
            speakers = set()
            for segment, _, label in speaker_segments.itertracks(yield_label=True):
                speakers.add(label)
            emit("workflow", f"🎤 Detected {len(speakers)} unique speakers: {', '.join(sorted(speakers))}")
            logging.info(f"Diarization found {num_segments} segments with {len(speakers)} speakers")
        else:
            emit("workflow", "❌ Diarization failed or returned no segments.")
            logging.warning("Diarization failed or returned no segments")

    except ConfigurationError:
        # A gated model, a bad token, or contradictory speaker bounds are things
        # a person must fix. Printing "Diarization failed" and carrying on gave
        # them a label-free transcript and exit code 0.
        raise
    except Exception as e:
        emit("workflow", f"❌ Diarization failed: {e}")
        logging.error(f"Diarization failed: {e}")
        speaker_segments = None
          # Clear GPU memory on error
        try:
            if 'torch' in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU memory cleared after diarization error")
        except:
            pass

    
    return speaker_segments


def full_workflow(audio_file=None, output_dir=None, rssfeed=None, force: bool = False, skip_postprocess: bool = False):
    """
    Complete workflow:
    1. If no audio_file, fetch latest episode from RSS feed.
    2. Prepare audio (normalize, etc).
    3. Transcribe audio (with diarization if available).
    4. Process transcript (summary, blog, etc).
    Args:
        audio_file: Path to audio file or None
        output_dir: Output directory (default: './podcasts' or from env)
        rssfeed: RSS feed URL (default: from env or fallback)
        force: If True, delete all related files for this episode before processing
    """    # Step 0: Apply optimal GPU utilization settings at the start
    emit("workflow", "🚀 Applying optimal GPU utilization settings...")
    optimal_batch_size, num_workers = maximize_gpu_utilization()

    # Step 1: Fetch latest episode if needed
    if not audio_file:
        if not rssfeed:
            rssfeed = os.environ.get("WHYCAST_RSSFEED", "https://whycast.podcast.audio/@whycast/feed.xml")
        if not output_dir:
            output_dir = os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
        emit("workflow", f"No audio file provided. Fetching latest episode from {rssfeed} ...")
        result = podcast_fetching_workflow(rssfeed, output_dir, return_base_name=True)
        if not result or not isinstance(result, tuple) or not result[0]:
            emit("workflow", "No episode could be fetched from the feed.")
            return
        audio_file, base_name = result
    else:
        if not output_dir:
            output_dir = os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
        base_name = os.path.splitext(os.path.basename(str(audio_file)))[0] if audio_file else None
        if force and base_name:
            emit("workflow", f"--force: Deleting all files for episode base name '{base_name}' in {output_dir}")
            delete_episode_files(base_name, output_dir)
    if not base_name:
        emit("workflow", "Could not determine base name for episode. Exiting.")
        return
    base_path = os.path.join(output_dir, base_name)
    # Step 2: Prepare audio
    emit("workflow", "[1/4] Preparing audio ...")
    prepared_audio = prepare_audio_for_diarization(str(audio_file))
    emit("workflow", f"Prepared audio: {prepared_audio}")

    # Step 3: Transcribe audio (with diarization)
    if not USE_SPEAKER_DIARIZATION:
        # Checked here rather than around diarize_audio, because everything
        # between the two is work done solely FOR diarization: verify_gpu_setup,
        # importing torch, and loading the whole episode into memory. Doing that
        # and then throwing the result away was slow, and any failure in it -
        # missing torch, no torchaudio backend - was reported as "Diarization
        # failed", which is exactly the message that sends someone hunting for a
        # broken GPU they had deliberately opted out of. faster-whisper runs on
        # ctranslate2, so torch is in this tree for diarization alone.
        emit("workflow", "[2/4] Speaker diarization is switched off; continuing without speaker labels.")
        logging.info("Speaker diarization disabled by configuration")
        speaker_segments = None
    else:
        speaker_segments = _run_diarization(prepared_audio)

    # Step 4: Transcribe
    emit("workflow", "[3/4] Transcribing audio ...")

    # Initialize Whisper model with GPU acceleration
    emit("workflow", "🔧 Setting up Whisper model with GPU acceleration...")
    model = setup_model(batch_size=optimal_batch_size)

    # Run transcription
    emit("workflow", "🎵 Running transcription...")
    segments, _ = transcribe_audio(model, prepared_audio, speaker_segments=speaker_segments)

    # Clear GPU memory after transcription to free up resources
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cleared after transcription")
    except:
        pass

    # Write transcript files
    transcript_text = write_transcript_files(segments, f"{base_path}_transcript.txt", f"{base_path}_ts.txt", speaker_segments=speaker_segments)
    emit("workflow", f"✅ Transcription complete. Transcript saved to {base_path}_transcript.txt")

    # Create merged transcript if speaker labels are present
    merged_transcript_text = None
    if transcript_text and any('[SPEAKER_' in line or re.search(r'\[[^\]]+\]', line) for line in transcript_text.splitlines()):
        emit("workflow", "🔄 Creating merged transcript (grouping consecutive speaker lines)...")
        merged_file = write_merged_transcript(transcript_text, base_path)
        if merged_file:
            emit("workflow", f"✅ Merged transcript created: {os.path.basename(merged_file)}")
            # Read the merged transcript for processing
            try:
                with open(merged_file, 'r', encoding='utf-8') as f:
                    merged_transcript_text = f.read()
                emit("workflow", "📝 Using merged transcript for processing workflow (cleaner format)")
            except Exception as e:
                emit("workflow", f"⚠️ Could not read merged transcript: {e}, using original transcript")
                merged_transcript_text = None

    # Step 4: Process transcript
    if skip_postprocess:
        # A transcript-only run: new audio handling, new diarization, new
        # transcript, and none of the paid language-model steps. What is on disk
        # from an earlier run - summary, blog, history - is left exactly as it
        # was, which means it now describes a transcript that no longer exists.
        # Saying so is the point; the caller decides whether to follow up.
        emit(
            "workflow",
            "[4/4] Skipped: transcript only. The summary, blog and history on "
            "disk still describe the previous transcript - re-run post-processing "
            "to bring them in line.",
            progress=1.0,
        )
        _cleanup_prepared_audio(prepared_audio, audio_file)
        return
    emit("workflow", "[4/4] Processing transcript workflow (summary, blog, history, etc.) ...")
    # Use merged transcript if available, otherwise use original
    text_for_processing = merged_transcript_text if merged_transcript_text else transcript_text
    if text_for_processing:
        if merged_transcript_text:
            emit("workflow", "🔄 Processing with merged transcript (improved readability)")
        process_transcript_workflow(text_for_processing, base_name, output_dir)
    else:
        emit("workflow", "Transcript text is empty, skipping transcript workflow.")
    _cleanup_prepared_audio(prepared_audio, audio_file)


def _cleanup_prepared_audio(prepared_audio, audio_file) -> None:
    """Remove the normalised copy diarization needed, if one was made.

    Pulled out of the tail of :func:`full_workflow` so the transcript-only path,
    which returns earlier, cleans up the same way. Failing to delete a temp file
    is worth a message and nothing more.
    """
    try:
        if os.path.abspath(prepared_audio) != os.path.abspath(str(audio_file)) and os.path.exists(prepared_audio):
            os.remove(prepared_audio)
            emit("workflow", f"Temporary normalized audio file deleted: {prepared_audio}")
    except Exception as e:
        emit("workflow", f"Error deleting temporary audio file: {e}")
