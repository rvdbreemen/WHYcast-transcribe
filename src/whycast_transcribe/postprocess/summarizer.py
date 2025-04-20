import logging
import os
from whycast_transcribe.config import PROMPT_SUMMARY_FILE, OPENAI_MODEL, MAX_TOKENS, MAX_INPUT_TOKENS
from whycast_transcribe.utils.file_helpers import read_prompt_file
from whycast_transcribe.utils.openai_processor import process_with_openai
from whycast_transcribe.utils.text_processing import split_into_chunks


def summarize_transcript(transcript: str) -> str:
    """
    Generate a summary for the transcript using OpenAI prompt.
    Supports recursive summarization if enabled.
    """
    logging.info("Postprocess: Starting summarization")
    prompt = read_prompt_file(PROMPT_SUMMARY_FILE)
    if not prompt:
        logging.warning("Summary prompt file not found or empty, skipping summary generation")
        return None

    logging.info("Summarizer: generating summary...")
    # Check recursive summarization flag from environment
    use_recursive = os.getenv('USE_RECURSIVE_SUMMARIZATION', 'False').lower() in ('true','1','yes')
    if use_recursive:
        chunks = split_into_chunks(transcript, max_chunk_size=MAX_INPUT_TOKENS)
        logging.info(f"Recursive summarization enabled: split into {len(chunks)} chunks")
        parts = []
        for chunk in chunks:
            logging.info("Summarizing chunk...")
            parts.append(process_with_openai(chunk, prompt, OPENAI_MODEL, max_tokens=MAX_TOKENS))
        summary = "\n\n".join(part for part in parts if part)
        logging.info("Recursive summarization completed")
        return summary

    # Single-pass summarization
    summary = process_with_openai(transcript, prompt, OPENAI_MODEL, max_tokens=MAX_TOKENS)
    logging.info("Single-pass summarization completed")
    return summary