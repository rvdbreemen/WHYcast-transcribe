import logging
from whycast_transcribe.config import PROMPT_HISTORY_EXTRACT_FILE, OPENAI_HISTORY_MODEL, MAX_TOKENS
from whycast_transcribe.utils.file_helpers import read_prompt_file
from whycast_transcribe.utils.openai_processor import process_with_openai


def generate_history(cleaned_transcript: str) -> str:
    """
    Generate a history lesson extraction from the transcript.
    """
    logging.info("Postprocess: Starting history extraction")
    prompt = read_prompt_file(PROMPT_HISTORY_EXTRACT_FILE)
    if not prompt:
        logging.warning("History prompt file not found or empty, skipping history extraction")
        return None

    logging.info("HistoryGenerator: generating history extraction...")
    result = process_with_openai(
        cleaned_transcript,
        prompt,
        OPENAI_HISTORY_MODEL,
        max_tokens=MAX_TOKENS * 2
    )
    if not result:
        logging.error("History extraction failed")
    logging.info("Postprocess: History extraction completed")
    return result