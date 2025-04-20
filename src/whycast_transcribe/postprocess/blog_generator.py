import logging
from whycast_transcribe.config import PROMPT_BLOG_FILE, PROMPT_BLOG_ALT1_FILE, OPENAI_MODEL, MAX_TOKENS
from whycast_transcribe.utils.file_helpers import read_prompt_file
from whycast_transcribe.utils.openai_processor import process_with_openai


def generate_blog(cleaned_transcript: str, summary: str) -> str:
    """
    Generate primary blog post based on cleaned transcript and summary.
    """
    logging.info("Postprocess: Starting primary blog generation")
    prompt = read_prompt_file(PROMPT_BLOG_FILE)
    if not prompt:
        logging.warning("Blog prompt file not found or empty, skipping blog generation")
        return None

    logging.info("BlogGenerator: generating primary blog post...")
    # Combine summary and transcript for context
    input_text = f"Summary:\n{summary}\nTranscript:\n{cleaned_transcript}"
    blog = process_with_openai(input_text, prompt, OPENAI_MODEL, max_tokens=MAX_TOKENS)
    logging.info("Postprocess: Primary blog generation completed")
    return blog


def generate_blog_alt(cleaned_transcript: str, summary: str) -> str:
    """
    Generate alternative blog post variant.
    """
    logging.info("Postprocess: Starting alternative blog generation")
    prompt = read_prompt_file(PROMPT_BLOG_ALT1_FILE)
    if not prompt:
        logging.warning("Alternative blog prompt file not found or empty, skipping alt blog generation")
        return None

    logging.info("BlogGenerator: generating alternative blog post...")
    input_text = f"Summary:\n{summary}\nTranscript:\n{cleaned_transcript}"
    alt_blog = process_with_openai(input_text, prompt, OPENAI_MODEL, max_tokens=MAX_TOKENS)
    logging.info("Postprocess: Alternative blog generation completed")
    return alt_blog