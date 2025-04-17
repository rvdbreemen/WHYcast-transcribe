#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content generation functionality for WHYcast-transcribe.

This consolidated module provides functions for generating various content
from podcast transcripts:
- Summaries (with support for large/recursive summarization)
- Blog posts (with multiple variants)
- History lesson extractions
- Speaker assignment

The module presents a unified interface while maintaining clear organization
of the different content generation capabilities.
"""

import os
import logging
import glob
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm

# Import configuration
from config import (
    OPENAI_MODEL,
    OPENAI_LARGE_CONTEXT_MODEL,
    OPENAI_HISTORY_MODEL,
    OPENAI_SPEAKER_ASSIGNMENT_MODEL,
    MAX_TOKENS,
    MAX_INPUT_TOKENS,
    USE_RECURSIVE_SUMMARIZATION,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP,
    PROMPT_CLEANUP_FILE,
    PROMPT_SUMMARY_FILE,
    PROMPT_BLOG_FILE,
    PROMPT_BLOG_ALT1_FILE,
    PROMPT_HISTORY_EXTRACT_FILE
)

# Import utilities from other modules
from utils.openai_processor import process_with_openai, choose_appropriate_model
from utils.text_processing import split_into_chunks, estimate_token_count
from utils.file_helpers import read_prompt_file
from utils.format_converters import convert_markdown_to_html, convert_markdown_to_wiki

# ---------------------- UNIFIED WORKFLOW FUNCTIONALITY ----------------------

def process_transcript_workflow(transcript: str) -> Optional[Dict[str, str]]:
    """
    Process transcript through the multi-step workflow: cleanup -> summary -> blog -> history extraction -> speaker assignment
    
    Args:
        transcript: The raw transcript text
        
    Returns:
        Dictionary with cleaned_transcript, summary, blog, history_extract, and speaker_assignment or None if failed
    """
    results = {}
    
    # Step 1: Clean up the transcript
    cleanup_prompt = read_prompt_file(PROMPT_CLEANUP_FILE)
    if not cleanup_prompt:
        logging.warning("Cleanup prompt file not found or empty, skipping cleanup step")
        cleaned_transcript = transcript
    else:
        logging.info("Step 1: Cleaning up transcript...")
        model_to_use = choose_appropriate_model(transcript)
        
        # For cleanup, explicitly use a higher token limit
        estimated_tokens = estimate_token_count(transcript)
        logging.info(f"Transcript size: ~{estimated_tokens} tokens")
        
        # Process with OpenAI, giving plenty of room for output
        cleaned_transcript = process_with_openai(transcript, cleanup_prompt, model_to_use, max_tokens=MAX_TOKENS * 2)
        
        # Check if the cleaning was successful and not truncated
        if not cleaned_transcript:
            logging.warning("Transcript cleanup failed, using original transcript")
            cleaned_transcript = transcript
        elif len(cleaned_transcript) < len(transcript) * 0.5:
            logging.warning(f"Cleaned transcript is suspiciously short ({len(cleaned_transcript)} chars vs original {len(transcript)} chars)")
            logging.warning("This may indicate truncation. Using original transcript instead.")
            cleaned_transcript = transcript
    
    results['cleaned_transcript'] = cleaned_transcript
    
    # Step 2: Generate summary
    summary_prompt = read_prompt_file(PROMPT_SUMMARY_FILE)
    if not summary_prompt:
        logging.warning("Summary prompt file not found or empty, skipping summary generation")
        results['summary'] = None
    else:
        logging.info("Step 2: Generating summary...")
        summary = generate_summary(cleaned_transcript, summary_prompt)
        results['summary'] = summary
    
    # Step 3: Generate blog post
    if results.get('summary'):
        blog_prompt = read_prompt_file(PROMPT_BLOG_FILE)
        if not blog_prompt:
            logging.warning("Blog prompt file not found or empty, skipping blog generation")
            results['blog'] = None
        else:
            logging.info("Step 3: Generating blog post...")
            blog = generate_blog(cleaned_transcript, results['summary'], blog_prompt)
            results['blog'] = blog
    else:
        logging.warning("Skipping blog generation because summary generation failed")
        results['blog'] = None
    
    # Step 4: Generate alternative blog post
    if results.get('summary') and os.path.isfile(PROMPT_BLOG_ALT1_FILE):
        # Only log and attempt to read if the file exists
        logging.debug(f"Alternative blog prompt file found: {PROMPT_BLOG_ALT1_FILE}")
        blog_alt1_prompt = read_prompt_file(PROMPT_BLOG_ALT1_FILE)
        
        # Log the content if it was read successfully
        if blog_alt1_prompt:
            logging.info("Step 4: Generating alternative blog post...")
            blog_alt1 = generate_blog(cleaned_transcript, results['summary'], blog_alt1_prompt)
            results['blog_alt1'] = blog_alt1
        else:
            logging.warning(f"Blog prompt alt1 file exists but couldn't be read or is empty")
            results['blog_alt1'] = None
    else:
        # File doesn't exist or no summary, just log this and skip
        logging.debug(f"Skipping alternative blog generation")
        results['blog_alt1'] = None
    
    # Step 5: Generate history extraction
    history_extract_prompt = read_prompt_file(PROMPT_HISTORY_EXTRACT_FILE)
    if not history_extract_prompt:
        logging.warning("History extraction prompt file not found or empty, skipping history extraction")
        results['history_extract'] = None
    else:
        logging.info("Step 5: Generating history lesson extraction...")
        
        # Generate history extraction
        history_extract = process_with_openai(
            cleaned_transcript,
            history_extract_prompt, 
            OPENAI_HISTORY_MODEL, 
            max_tokens=MAX_TOKENS * 2
        )
        
        if history_extract:
            logging.info(f"Successfully generated history extract ({len(history_extract)} chars)")
            results['history_extract'] = history_extract
        else:
            logging.error("History extraction failed")
            results['history_extract'] = None
    
    # Step 6: Speaker assignment (only if diarization is present)
    try:
        if any("SPEAKER_" in line for line in cleaned_transcript.splitlines()):
            from content_generator import process_speaker_assignment_workflow
            import os
            base_path = os.environ.get("WORKFLOW_OUTPUT_BASE", "output")
            txt_path, html_path, md_path = process_speaker_assignment_workflow(cleaned_transcript, base_path)
            with open(txt_path, "r", encoding="utf-8") as f:
                results['speaker_assignment'] = f.read()
        else:
            results['speaker_assignment'] = None
    except Exception as e:
        import logging
        logging.error(f"Speaker assignment workflow failed: {str(e)}")
        results['speaker_assignment'] = None

    return results


def write_workflow_outputs(results: Dict[str, str], output_base: str) -> None:
    """
    Write the outputs from the workflow to files.
    
    Args:
        results: Dictionary with cleaned_transcript, summary, blog, and history_extract
        output_base: Base path for output files
    """
    # Write cleaned transcript
    if 'cleaned_transcript' in results and results['cleaned_transcript']:
        cleaned_path = f"{output_base}_cleaned.txt"
        with open(cleaned_path, "w", encoding="utf-8") as f:
            f.write(results['cleaned_transcript'])
        logging.info(f"Cleaned transcript saved to: {cleaned_path}")
    
    # Write summary
    if 'summary' in results and results['summary']:
        summary_path = f"{output_base}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(results['summary'])
        logging.info(f"Summary saved to: {summary_path}")
    
    # Write blog post
    if 'blog' in results and results['blog']:
        save_content_outputs(results['blog'], output_base, "blog")
    
    # Write alternative blog post
    if 'blog_alt1' in results and results['blog_alt1']:
        save_content_outputs(results['blog_alt1'], output_base, "blog_alt1")
    
    # Write history extraction
    if 'history_extract' in results and results['history_extract']:
        save_content_outputs(results['history_extract'], output_base, "history")
    
    # Write speaker assignment
    if 'speaker_assignment' in results and results['speaker_assignment']:
        speaker_assignment_path = f"{output_base}_with_speakers.txt"
        with open(speaker_assignment_path, "w", encoding="utf-8") as f:
            f.write(results['speaker_assignment'])
        logging.info(f"Speaker assignment saved to: {speaker_assignment_path}")


def process_directory_workflow(directory: str) -> None:
    """
    Process all transcript files in a directory using the unified workflow.
    Args:
        directory: Directory containing transcript files
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create or access directory {directory}: {str(e)}")
        return
    if not os.path.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return
    files = glob.glob(os.path.join(directory, "*.txt"))
    transcript_files = [f for f in files if all(x not in f for x in ["_ts.txt", "_summary.txt", "_blog.txt", "_cleaned.txt", "_history.txt", "_blog_alt1.txt", "_with_speakers.txt"])]
    if not transcript_files:
        logging.warning(f"No transcript files found in directory: {directory}")
        return
    logging.info(f"Found {len(transcript_files)} transcript files to process")
    for file in tqdm(transcript_files, desc="Unified workflow"):
        logging.info(f"Processing file: {file}")
        try:
            with open(file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            base = os.path.splitext(file)[0]
            results = process_transcript_workflow(transcript)
            if results:
                write_workflow_outputs(results, base)
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
        import gc
        gc.collect()


def process_speaker_assignment_workflow(transcript: str, base_path: str) -> None:
    """
    Process transcript with speaker assignment prompt using o3-mini, chunking if needed.
    Output: <base>_with_speakers.txt, .html, .md
    """
    from config import OPENAI_MODEL, MAX_INPUT_TOKENS
    from utils.text_processing import split_into_chunks
    from utils.openai_processor import process_with_openai
    from utils.file_helpers import read_prompt_file
    from utils.format_converters import convert_markdown_to_html
    import os

    # Check for diarization tags in transcript
    if not any(f"SPEAKER_" in line for line in transcript.splitlines()):
        raise ValueError("Transcript does not contain diarization speaker tags. Speaker assignment prompt requires diarized transcript.")

    prompt = read_prompt_file(os.path.join("prompts", "speaker_assignment.txt"))
    if not prompt:
        raise RuntimeError("Speaker assignment prompt file not found.")

    # Chunk transcript if needed
    chunks = split_into_chunks(transcript, max_chunk_size=MAX_INPUT_TOKENS * 4)  # 4 chars/token
    results = []
    for chunk in chunks:
        result = process_with_openai(chunk, prompt, model="gpt-4o-2024-05-13", max_tokens=4096)
        results.append(result)
    full_result = "\n\n".join(results)

    # Write .txt
    txt_path = f"{base_path}_with_speakers.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_result)
    # Write .html
    html_path = f"{base_path}_with_speakers.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(convert_markdown_to_html(full_result))
    # Write .md
    md_path = f"{base_path}_with_speakers.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(full_result)
    return txt_path, html_path, md_path