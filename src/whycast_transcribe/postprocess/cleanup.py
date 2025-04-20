import logging
from whycast_transcribe.config import PROMPT_CLEANUP_FILE, MAX_TOKENS
from whycast_transcribe.utils.file_helpers import read_prompt_file
from whycast_transcribe.utils.text_processing import estimate_token_count
from whycast_transcribe.utils.openai_processor import process_with_openai, choose_appropriate_model


def cleanup_transcript(transcript: str) -> str:
    """
    Clean up the raw transcript using the cleanup prompt and OpenAI.
    Returns the cleaned transcript or the original if cleanup fails.
    """
    logging.info("Postprocess: Starting transcript cleanup")
    cleanup_prompt = read_prompt_file(PROMPT_CLEANUP_FILE)
    if not cleanup_prompt:
        logging.warning("Cleanup prompt file not found or empty, skipping cleanup step")
        return transcript

    logging.info("Cleanup: cleaning up transcript...")
    model_to_use = choose_appropriate_model(transcript)
    estimated_tokens = estimate_token_count(transcript)
    logging.info(f"Transcript size: ~{estimated_tokens} tokens")

    # Process with OpenAI, allowing for extended output
    cleaned = process_with_openai(
        transcript,
        cleanup_prompt,
        model_to_use,
        max_tokens=MAX_TOKENS * 2
    )
    if not cleaned:
        logging.warning("Transcript cleanup failed, using original transcript")
        return transcript
    if len(cleaned) < len(transcript) * 0.5:
        logging.warning("Cleaned transcript is suspiciously short, using original transcript")
        return transcript
    logging.info("Postprocess: Transcript cleanup completed")
    return cleaned