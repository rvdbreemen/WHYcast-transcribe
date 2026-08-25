"""
Audio preparation step for the WHYcast pipeline (ADR-008).

Extracted verbatim from transcribe.py (prepare_audio_for_diarization,
lines 2707-2741). Mechanical extraction: no redesign, no cleanup.
"""

import logging
import os
import subprocess
import warnings
from typing import Optional

logger = logging.getLogger(__name__)

# Suppress specific warnings (mirrors legacy transcribe.py module-level behavior
# for torchaudio warnings; read-only side effect)
warnings.filterwarnings("ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio")
warnings.filterwarnings("ignore", message="The bits_per_sample of .mp3 is set to 0 by default")
warnings.filterwarnings("ignore", message="PySoundFile failed")


def prepare_audio_for_diarization(audio_file: str, output_dir: Optional[str] = None) -> str:
    """
    Ensures the audio file is in a compatible mono 16kHz mp3 format for diarization/transcription.
    Always creates a new file for diarization, never overwrites the original.
    Returns the path to the prepared mp3 file.
    """
    import hashlib
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    os.makedirs(output_dir, exist_ok=True)
    base, _ = os.path.splitext(os.path.basename(audio_file))
    hash_suffix = hashlib.md5(audio_file.encode('utf-8')).hexdigest()[:8]
    output_file = os.path.join(output_dir, f"{base}_mono16k_{hash_suffix}.mp3")

    if os.path.exists(output_file):
        logging.info(f"Prepared audio already exists: {output_file}")
        return output_file

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', audio_file, '-vn', '-ar', '16000', '-ac', '1', '-b:a', '192k', output_file
    ]
    try:
        logging.info(f"Converting {audio_file} to mono 16kHz mp3 using ffmpeg (output: {output_file})...")

        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logging.info(f"ffmpeg output: {result.stdout.decode(errors='ignore')}")
        logging.info(f"Audio converted to {output_file}")


        return output_file
    except Exception as e:
        logging.error(f"ffmpeg conversion failed: {e}\n{getattr(e, 'stderr', b'').decode(errors='ignore')}")
        raise RuntimeError(f"Failed to convert {audio_file} to mono 16kHz mp3: {e}")
