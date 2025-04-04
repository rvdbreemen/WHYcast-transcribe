"""
Utilities for transcript summarization and blog generation.
"""

import os
import logging
from typing import Dict, Optional

# Import from other modules
from api_utils import process_with_openai, choose_appropriate_model
from file_utils import read_prompt_file, write_workflow_outputs

# Import configuration variables
try:
    from config import (
        PROMPT_CLEANUP_FILE, PROMPT_SUMMARY_FILE, 
        PROMPT_BLOG_FILE, PROMPT_BLOG_ALT1_FILE,
        MAX_TOKENS
    )
except ImportError as e:
    raise ImportError(f"Error importing summarization configuration: {e}")

def process_transcript_workflow(transcript: str) -> Optional[Dict[str, str]]:
    """
    Process transcript through the multi-step workflow: cleanup -> summary -> blog
    
    Args:
        transcript: The raw transcript text
        
    Returns:
        Dictionary with cleaned_transcript, summary, and blog, or None if failed
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
        estimated_tokens = len(transcript) // 4  # Simple estimation
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
        model_to_use = choose_appropriate_model(cleaned_transcript)
        summary = process_with_openai(cleaned_transcript, summary_prompt, model_to_use)
        results['summary'] = summary
    
    # Step 3: Generate blog post
    blog_prompt = read_prompt_file(PROMPT_BLOG_FILE)
    if not blog_prompt:
        logging.warning("Blog prompt file not found or empty, skipping blog generation")
        results['blog'] = None
    else:
        logging.info("Step 3: Generating blog post...")
        # Use both the cleaned transcript and summary for blog generation
        input_text = f"CLEANED TRANSCRIPT:\n{cleaned_transcript}\n\nSUMMARY:\n{results.get('summary', 'No summary available')}"
        model_to_use = choose_appropriate_model(input_text)
        blog = process_with_openai(input_text, blog_prompt, model_to_use, max_tokens=MAX_TOKENS * 2)
        results['blog'] = blog
        
    # Step 4: Generate alternative blog post
    # Log the filename before reading
    logging.info(f"Reading prompt file: {PROMPT_BLOG_ALT1_FILE}")
    blog_alt1_prompt = read_prompt_file(PROMPT_BLOG_ALT1_FILE)
    
    # Log the content if it was read successfully
    if blog_alt1_prompt:
        logging.info(f"Prompt file content: {blog_alt1_prompt[:100]}..." if len(blog_alt1_prompt) > 100 else blog_alt1_prompt)
    else:
        logging.warning(f"Failed to read prompt file: {PROMPT_BLOG_ALT1_FILE}")
    
    if not blog_alt1_prompt:
        logging.warning("Blog prompt alt1 file not found or empty, skipping alternative blog generation")
        results['blog_alt1'] = None
    else:
        logging.info("Step 4: Generating alternative blog post...")
        # Use both the cleaned transcript and summary for blog generation
        input_text = f"CLEANED TRANSCRIPT:\n{cleaned_transcript}\n\nSUMMARY:\n{results.get('summary', 'No summary available')}"
        model_to_use = choose_appropriate_model(input_text)
        blog_alt1 = process_with_openai(input_text, blog_alt1_prompt, model_to_use, max_tokens=MAX_TOKENS * 2)
        results['blog_alt1'] = blog_alt1
    
    return results

def process_summary(full_transcript: str, output_summary_file: str, output_blog_file: Optional[str] = None) -> bool:
    """
    Process the transcript to generate and save summary and blog.
    
    Args:
        full_transcript: The complete transcript text
        output_summary_file: Path to save the summary
        output_blog_file: Path to save the blog
        
    Returns:
        Success status (True/False)
    """
    # Use the multi-step workflow
    results = process_transcript_workflow(full_transcript)
    
    if not results:
        logging.error("Failed to process transcript")
        return False
    
    # Get the base output path
    output_base = os.path.splitext(output_summary_file)[0].replace('_summary', '')
    
    # Write outputs to files
    write_workflow_outputs(results, output_base)
    
    return True