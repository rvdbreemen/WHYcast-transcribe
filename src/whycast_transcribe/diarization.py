import logging
import torch
from pyannote.audio import Pipeline
from whycast_transcribe.config import (
    USE_SPEAKER_DIARIZATION,
    DIARIZATION_MODEL,
    DIARIZATION_ALTERNATIVE_MODEL,
    DIARIZATION_MIN_SPEAKERS,
    DIARIZATION_MAX_SPEAKERS
)


def perform_diarization(audio_file: str):
    """
    Perform speaker diarization and return list of segments with speaker labels.
    """
    if not USE_SPEAKER_DIARIZATION:
        return None

    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=True)
    except Exception:
        logging.warning(f"Primary diarization model failed, using alternative model {DIARIZATION_ALTERNATIVE_MODEL}")
        pipeline = Pipeline.from_pretrained(DIARIZATION_ALTERNATIVE_MODEL, use_auth_token=True)

    if torch.cuda.is_available():
        pipeline.to(torch.device('cuda'))
        logging.info("Diarization pipeline moved to GPU")

    diarization = pipeline(
        audio_file,
        min_speakers=DIARIZATION_MIN_SPEAKERS,
        max_speakers=DIARIZATION_MAX_SPEAKERS
    )

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    logging.info(f"Diarization produced {len(segments)} segments")
    return segments