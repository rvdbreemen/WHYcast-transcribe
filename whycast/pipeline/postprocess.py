"""
Transcript post-processing steps for the WHYcast pipeline (ADR-008).

Extracted verbatim from transcribe.py (process_transcript_workflow,
cleanup_step, summary_step, blog_step, alt_blog_step, history_step).
"""

import logging
import os
from typing import Dict, Optional

from whycast.config import (
    MAX_TOKENS,
    MAX_INPUT_TOKENS,
    USE_RECURSIVE_SUMMARIZATION,
    OPENAI_HISTORY_MODEL,
    PROMPT_CLEANUP_FILE,
    PROMPT_SUMMARY_FILE,
    PROMPT_BLOG_FILE,
    PROMPT_BLOG_ALT1_FILE,
    PROMPT_HISTORY_EXTRACT_FILE,
)
from whycast.pipeline.llm import (
    read_prompt_file,
    estimate_token_count,
    choose_appropriate_model,
    process_with_openai,
    summarize_large_transcript,
)
from whycast.pipeline.outputs import write_all_format
from whycast.pipeline.speakers import speaker_assignment_step

logger = logging.getLogger(__name__)


def process_transcript_workflow(transcript: str, output_basename: str = None, output_dir: str = None) -> Dict[str, Optional[str]]:
    """
    Orchestrate the full transcript processing workflow:
    1. Speaker assignment
    2. Cleanup
    3. Summary
    4. Blog
    5. Alternative blog
    6. History extraction
    Returns a dictionary of all results.
    """
    results = {}
    speaker_assigned_transcript = speaker_assignment_step(transcript, output_basename, output_dir)
    results['speaker_assignment'] = speaker_assigned_transcript

    # Use speaker-assigned transcript for cleanup if available, otherwise use original
    transcript_for_cleanup = speaker_assigned_transcript if speaker_assigned_transcript else transcript
    cleaned = cleanup_step(transcript_for_cleanup)

    results['cleaned_transcript'] = cleaned
    results['summary'] = summary_step(cleaned, output_basename, output_dir)
    results['blog'] = blog_step(cleaned, results['summary'], output_basename, output_dir)
    results['blog_alt1'] = alt_blog_step(cleaned, results['summary'], output_basename, output_dir)
    results['history_extract'] = history_step(cleaned, output_basename, output_dir)
    return results

def cleanup_step(transcript: str) -> str:
    """
    Step 2: Cleanup transcript via OpenAI.
    Uses a prompt to clean up the transcript, removing filler words, fixing grammar, etc.
    Returns the cleaned transcript as a string. If cleanup fails, returns the original transcript.
    """
    prompt = read_prompt_file(PROMPT_CLEANUP_FILE)
    if not prompt:
        logging.warning("Cleanup prompt missing, skipping cleanup")
        return transcript
    logging.info("Step 2: Cleaning up transcript...")
    model = choose_appropriate_model(transcript)
    cleaned = process_with_openai(transcript, prompt, model, max_tokens=MAX_TOKENS * 2)
    if not cleaned or len(cleaned) < len(transcript) * 0.5:
        logging.warning("Cleanup result invalid, using original transcript")
        return transcript
    return cleaned

def summary_step(cleaned: str, output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """
    Step 3: Generate summary from the cleaned transcript.
    Uses a prompt to summarize the cleaned transcript. Returns the summary as a string, or None if failed.
    """
    prompt = read_prompt_file(PROMPT_SUMMARY_FILE)
    if not prompt:
        logging.warning("Summary prompt missing, skipping summary")
        return None

    logging.info("Step 3: Generating summary...")

    estimated_tokens = estimate_token_count(cleaned)
    logging.info(f"Estimated token count for summary: {estimated_tokens}")

    if USE_RECURSIVE_SUMMARIZATION and estimated_tokens > MAX_INPUT_TOKENS:
        summary = summarize_large_transcript(cleaned, prompt)
    else:
        summary = process_with_openai(
            cleaned,
            prompt,
            choose_appropriate_model(cleaned),
        )
    if summary and output_basename and output_dir:
        write_all_format(summary, f"{output_basename}_summary", output_dir)
    return summary

def blog_step(cleaned: str, summary: Optional[str], output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """
    Step 4: Generate blog post from the cleaned transcript and summary.
    Uses a prompt to generate a blog post. Returns the blog post as a string, or None if failed.
    """
    if not summary:
        return None
    prompt = read_prompt_file(PROMPT_BLOG_FILE)
    if not prompt:
        logging.warning("Blog prompt missing, skipping blog generation")
        return None
    logging.info("Step 4: Generating blog post...")
    input_text = f"CLEANED TRANSCRIPT:\n{cleaned}\n\nSUMMARY:\n{summary}"
    blog = process_with_openai(input_text, prompt, choose_appropriate_model(input_text), max_tokens=MAX_TOKENS * 2)
    if blog and output_basename and output_dir:
        write_all_format(blog, f"{output_basename}_blog", output_dir)
    return blog

def alt_blog_step(cleaned: str, summary: Optional[str], output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """
    Step 5: Generate alternative blog post from the cleaned transcript and summary.
    Uses an alternative prompt to generate a different style of blog post. Returns the alt blog post as a string, or None if failed.
    """
    if not summary or not os.path.isfile(PROMPT_BLOG_ALT1_FILE):
        return None
    prompt = read_prompt_file(PROMPT_BLOG_ALT1_FILE)
    if not prompt:
        logging.warning("Alt blog prompt empty, skipping")
        return None
    logging.info("Step 5: Generating alternative blog post...")
    input_text = f"CLEANED TRANSCRIPT:\n{cleaned}\n\nSUMMARY:\n{summary}"
    alt_blog = process_with_openai(input_text, prompt, choose_appropriate_model(input_text), max_tokens=MAX_TOKENS * 2)
    if alt_blog and output_basename and output_dir:
        write_all_format(alt_blog, f"{output_basename}_blog_alt1", output_dir)
    return alt_blog

def history_step(cleaned: str, output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """
    Step 6: Generate history extraction from the cleaned transcript.
    Uses a prompt to extract historical lessons or context from the transcript. Returns the history extraction as a string, or None if failed.
    """
    prompt = read_prompt_file(PROMPT_HISTORY_EXTRACT_FILE)
    if not prompt:
        logging.warning("History prompt missing, skipping history extraction")
        return None
    logging.info("Step 6: Generating history extraction...")
    history = process_with_openai(cleaned, prompt, OPENAI_HISTORY_MODEL, max_tokens=MAX_TOKENS * 2)
    if history and output_basename and output_dir:
        write_all_format(history, f"{output_basename}_history", output_dir)
    return history
