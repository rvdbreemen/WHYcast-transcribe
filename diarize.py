import torch
import numpy as np
from pyannote.audio import Pipeline

def diarize_audio(audio_file, access_token):
    """
    Performs speaker diarization on the given audio file.

    Args:
        audio_file (str): Path to the audio file.
        access_token (str): Hugging Face access token.

    Returns:
        list: A list of tuples, where each tuple contains the start time, end time, and speaker label.
    """
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=access_token)

        # Set the device to GPU if available, otherwise use CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(device)

        diarization = pipeline(audio_file)

        segments = []
        for segment, track, label in diarization.itertracks(yield_label=True):
            segments.append((segment.start, segment.end, label))

        return segments
    except Exception as e:
        print(f"Error during diarization: {e}")
        return []
