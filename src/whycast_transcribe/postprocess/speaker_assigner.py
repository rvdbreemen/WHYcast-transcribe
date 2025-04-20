import logging
from whycast_transcribe.config import PROMPT_SPEAKER_ASSIGN_FILE, OPENAI_SPEAKER_MODEL, MAX_TOKENS, MAX_INPUT_TOKENS
from whycast_transcribe.utils.file_helpers import read_prompt_file
from whycast_transcribe.utils.text_processing import split_into_chunks
from whycast_transcribe.utils.openai_processor import process_with_openai


def assign_speakers(cleaned_transcript: str) -> str:
    """
    Assign speaker labels to the transcript using OpenAI prompt.
    Returns the speaker-assigned transcript in markdown format.
    """
    logging.info("Postprocess: Starting speaker assignment")
    # Check if transcript includes diarization tags
    if "SPEAKER_" not in cleaned_transcript:
        logging.info("No diarization tags found, skipping speaker assignment")
        return None

    prompt = read_prompt_file(PROMPT_SPEAKER_ASSIGN_FILE)
    if not prompt:
        logging.warning("Speaker assignment prompt file not found or empty, skipping speaker assignment")
        return None

    logging.info("SpeakerAssigner: assigning speakers in transcript...")
    chunks = split_into_chunks(cleaned_transcript, max_chunk_size=MAX_INPUT_TOKENS * 4)
    logging.info(f"SpeakerAssigner: Processing {len(chunks)} chunks for speaker assignment")
    parts = []
    for i, chunk in enumerate(chunks):
        logging.info(f"SpeakerAssigner: Assigning speakers for chunk {i+1}/{len(chunks)}")
        part = process_with_openai(chunk, prompt, OPENAI_SPEAKER_MODEL, max_tokens=MAX_TOKENS)
        if part:
            parts.append(part)
    logging.info("Postprocess: Speaker assignment completed")
    return "\n\n".join(parts)