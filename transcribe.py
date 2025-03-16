#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WHYcast Transcribe - v0.0.6

A tool for transcribing audio files and generating summaries using OpenAI GPT models.
Supports downloading the latest episode from podcast feeds.

Copyright (c) 2025 Robert van den Breemen
License: MIT (see LICENSE file for details)
"""

import os
import sys
import logging
from typing import Tuple, List, Optional, Dict
from dotenv import load_dotenv
from tqdm import tqdm
import glob
import argparse
import json
import re
from openai import OpenAI, BadRequestError
from faster_whisper import WhisperModel
import feedparser
import requests
import hashlib
from urllib.parse import urlparse
from datetime import datetime
import markdown  # New import for markdown to HTML conversion

# Try to import configuration
try:
    from config import (
        VERSION, MODEL_SIZE, DEVICE, COMPUTE_TYPE, BEAM_SIZE,
        OPENAI_MODEL, OPENAI_LARGE_CONTEXT_MODEL, 
        TEMPERATURE, MAX_TOKENS, MAX_INPUT_TOKENS, CHARS_PER_TOKEN,
        PROMPT_CLEANUP_FILE, PROMPT_SUMMARY_FILE, PROMPT_BLOG_FILE, PROMPT_BLOG_ALT1_FILE, MAX_FILE_SIZE_KB,
        USE_RECURSIVE_SUMMARIZATION, MAX_CHUNK_SIZE, CHUNK_OVERLAP,
        VOCABULARY_FILE, USE_CUSTOM_VOCABULARY
    )
except ImportError as e:
    sys.stderr.write(f"Error: Could not import configuration - {str(e)}\n")
    sys.stderr.write("Make sure config.py exists and contains all required settings.\n")
    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"Error in configuration: {str(e)}\n")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Set up logging to both console and file
def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with UTF-8 encoding
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transcribe.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Console handler with proper Unicode handling
    try:
        # Configure console for UTF-8
        if sys.platform == 'win32':
            import locale
            # Use error handler that replaces problematic characters
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            console_handler.setStream(open(os.devnull, 'w', encoding='utf-8'))  # Dummy stream for initial setup
            
            # Custom StreamHandler that handles encoding errors
            class EncodingSafeStreamHandler(logging.StreamHandler):
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        stream = self.stream
                        # Write with error handling for encoding issues
                        try:
                            stream.write(msg + self.terminator)
                        except UnicodeEncodeError:
                            # Fall back to ascii with replacement characters
                            stream.write(msg.encode('ascii', 'replace').decode('ascii') + self.terminator)
                        self.flush()
                    except Exception:
                        self.handleError(record)
            
            # Use our custom handler
            console_handler = EncodingSafeStreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
        else:
            # On non-Windows platforms, standard handler usually works fine
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            
        logger.addHandler(console_handler)
    except Exception as e:
        # Fallback to basic handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        logger.warning(f"Could not set up optimal console logging: {e}")
    
    return logger

logger = setup_logging()

# ==================== API FUNCTIONS ====================
def ensure_api_key() -> str:
    """
    Ensure that the OpenAI API key is available.
    
    Returns:
        str: The API key if available
        
    Raises:
        ValueError: If the API key is not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key

def read_prompt_file(prompt_file: str) -> Optional[str]:
    """
    Read a prompt from file.
    
    Args:
        prompt_file: Path to the prompt file
        
    Returns:
        The prompt text or None if there was an error
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading prompt file {prompt_file}: {str(e)}")
        return None

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    
    Args:
        text: The text to estimate
        
    Returns:
        Estimated number of tokens
    """
    return len(text) // CHARS_PER_TOKEN

def truncate_transcript(transcript: str, max_tokens: int) -> str:
    """
    Truncate a transcript to fit within token limits.
    
    Args:
        transcript: The transcript text
        max_tokens: Maximum token count allowed
        
    Returns:
        Truncated transcript
    """
    estimated_tokens = estimate_token_count(transcript)
    if estimated_tokens <= max_tokens:
        return transcript
        
    # If transcript is too long, keep the first part and last part
    chars_to_keep = max_tokens * CHARS_PER_TOKEN
    first_part_size = chars_to_keep // 2
    last_part_size = chars_to_keep - first_part_size - 100  # Leave room for ellipsis message
    
    first_part = transcript[:first_part_size]
    last_part = transcript[-last_part_size:]
    
    return first_part + "\n\n[...transcript truncated due to length...]\n\n" + last_part

def choose_appropriate_model(transcript: str) -> str:
    """
    Choose the appropriate model based on transcript length.
    
    Args:
        transcript: The transcript text
        
    Returns:
        Model name to use
    """
    estimated_tokens = estimate_token_count(transcript)
    logging.info(f"Estimated transcript length: ~{estimated_tokens} tokens")
    
    # If transcript is long, use large context model
    if estimated_tokens > MAX_INPUT_TOKENS and OPENAI_LARGE_CONTEXT_MODEL:
        logging.info(f"Using large context model: {OPENAI_LARGE_CONTEXT_MODEL} due to transcript length")
        return OPENAI_LARGE_CONTEXT_MODEL
    
    return OPENAI_MODEL

def check_file_size(file_path: str) -> bool:
    """
    Check if the file size is within acceptable limits.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if file size is acceptable, False if it's too large
    """
    size_kb = os.path.getsize(file_path) / 1024
    if size_kb > MAX_FILE_SIZE_KB:
        logging.warning(f"File size ({size_kb:.1f} KB) exceeds recommended limit ({MAX_FILE_SIZE_KB} KB). " 
                      f"Processing might be slow or fail.")
        return False
    return True

def split_into_chunks(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks of specified maximum size with overlap.
    
    Args:
        text: The text to split
        max_chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(text):
        # Find a good breaking point (end of sentence or paragraph)
        end = min(start + max_chunk_size, len(text))
        
        # Try to find paragraph break
        paragraph_break = text.rfind('\n\n', start, end)
        if paragraph_break != -1 and paragraph_break > start + max_chunk_size // 2:
            end = paragraph_break + 2
        else:
            # Try to find sentence break (period followed by space)
            sentence_break = text.rfind('. ', start, end)
            if sentence_break != -1 and sentence_break > start + max_chunk_size // 2:
                end = sentence_break + 2
        
        chunks.append(text[start:end])
        # Start the next chunk with some overlap for context
        start = max(start + 1, end - overlap)
        
    return chunks

def summarize_large_transcript(transcript: str, prompt: str, client: OpenAI) -> Optional[str]:
    """
    Handle very large transcripts by chunking and recursive summarization.
    
    Args:
        transcript: The transcript text
        prompt: The summarization prompt
        client: OpenAI client
        
    Returns:
        Generated summary or None if failed
    """
    estimated_tokens = estimate_token_count(transcript)
    logging.info(f"Starting recursive summarization for large transcript (~{estimated_tokens} tokens)")
    
    # Split into chunks
    chunks = split_into_chunks(transcript)
    logging.info(f"Split transcript into {len(chunks)} chunks")
    
    # Process each chunk
    intermediate_summaries = []
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        try:
            chunk_prompt = "This is part of a longer transcript. Please summarize just this section, focusing on key points."
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": f"{chunk_prompt}\n\n{chunk}"}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS // 2  # Shorter summaries for intermediate steps
            )
            summary = response.choices[0].message.content
            intermediate_summaries.append(summary)
            logging.info(f"Completed summary for chunk {i+1}")
        except Exception as e:
            logging.error(f"Error summarizing chunk {i+1}: {str(e)}")
            # Continue with partial results if available
    
    # If we have intermediate summaries, combine them
    if not intermediate_summaries:
        return None
    
    # Combine intermediate summaries into a final summary
    combined_text = "\n\n".join(intermediate_summaries)
    
    try:
        logging.info("Generating final summary from intermediate summaries")
        response = client.chat.completions.create(
            model=OPENAI_LARGE_CONTEXT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
                {"role": "user", "content": f"{prompt}\n\nHere are summaries from different parts of the transcript. Please combine them into a cohesive summary and blog post:\n\n{combined_text}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating final summary: {str(e)}")
        return None

def process_with_openai(text: str, prompt: str, model_name: str, max_tokens: int = MAX_TOKENS) -> Optional[str]:
    """
    Process text with OpenAI model using the specified prompt.
    
    Args:
        text: The text to process
        prompt: Instructions for the AI
        model_name: Name of the model to use
        max_tokens: Maximum tokens for output
        
    Returns:
        The generated text or None if there was an error
    """
    try:
        api_key = ensure_api_key()
        client = OpenAI(api_key=api_key)
        
        estimated_tokens = estimate_token_count(text)
        logging.info(f"Processing text with OpenAI (~{estimated_tokens} tokens)")
        
        # If text is too long, truncate it
        if estimated_tokens > MAX_INPUT_TOKENS:
            logging.warning(f"Text too long (~{estimated_tokens} tokens > {MAX_INPUT_TOKENS} limit), truncating...")
            text = truncate_transcript(text, MAX_INPUT_TOKENS)
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
                    {"role": "user", "content": f"{prompt}\n\nHere's the text to process:\n\n{text}"}
                ],
                temperature=TEMPERATURE,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except BadRequestError as e:
            if "maximum context length" in str(e).lower():
                # Try with more aggressive truncation
                logging.warning(f"Context length exceeded. Retrying with further truncation...")
                text = truncate_transcript(text, MAX_INPUT_TOKENS // 2)
                logging.info(f"Retrying with reduced text (~{estimate_token_count(text)} tokens)...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
                        {"role": "user", "content": f"{prompt}\n\nHere's the text to process:\n\n{text}"}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                raise
    except Exception as e:
        logging.error(f"Error processing with OpenAI: {str(e)}")
        return None

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
        cleaned_transcript = process_with_openai(transcript, cleanup_prompt, model_to_use)
        if not cleaned_transcript:
            logging.warning("Transcript cleanup failed, using original transcript")
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
        
    # Step 4: Generate blog post
    # Log the filename before reading
    logging.info(f"Reading prompt file: {PROMPT_BLOG_ALT1_FILE}")
    blog_alt1_prompt = read_prompt_file(PROMPT_BLOG_ALT1_FILE)
    
    # Log the content if it was read successfully
    if blog_alt1_prompt:
        logging.info(f"Prompt file content: {blog_alt1_prompt[:100]}..." if len(blog_alt1_prompt) > 100 else blog_alt1_prompt)
    else:
        logging.warning(f"Failed to read prompt file: {PROMPT_BLOG_ALT1_FILE}")
    if not blog_alt1_prompt:
        logging.warning("Blog prompt alt1 file not found or empty, skipping blog generation")
        results['blog_alt1'] = None
    else:
        logging.info("Step 4: Generating blog alt1 post...")
        # Use both the cleaned transcript and summary for blog generation
        input_text = f"CLEANED TRANSCRIPT:\n{cleaned_transcript}\n\nSUMMARY:\n{results.get('summary', 'No summary available')}"
        model_to_use = choose_appropriate_model(input_text)
        blog_alt1 = process_with_openai(input_text, blog_prompt, model_to_use, max_tokens=MAX_TOKENS * 2)
        results['blog_alt1'] = blog_alt1
    
    return results

def write_workflow_outputs(results: Dict[str, str], output_base: str) -> None:
    """
    Write the outputs from the workflow to files.
    
    Args:
        results: Dictionary with cleaned_transcript, summary, and blog
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
        # Get base filename without any special suffixes
        base_filename = os.path.basename(output_base)
        output_dir = os.path.dirname(output_base)
        blog_path = os.path.join(output_dir, f"{base_filename}_blog.txt")
        
        with open(blog_path, "w", encoding="utf-8") as f:
            f.write(results['blog'])
        logging.info(f"Blog post saved to: {blog_path}")
        
        # Generate and write HTML version
        html_content = convert_markdown_to_html(results['blog'])
        html_path = os.path.join(output_dir, f"{base_filename}_blog.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML blog post saved to: {html_path}")
        
        # Generate and write Wiki version
        wiki_content = convert_markdown_to_wiki(results['blog'])
        wiki_path = os.path.join(output_dir, f"{base_filename}_blog.wiki")
        with open(wiki_path, "w", encoding="utf-8") as f:
            f.write(wiki_content)
        logging.info(f"Wiki blog post saved to: {wiki_path}")
        
    # Write blog post
    if 'blog_alt1' in results and results['blog_alt1']:
        # Get base filename without any special suffixes
        base_filename = os.path.basename(output_base)
        output_dir = os.path.dirname(output_base)
        blog_alt1_path = os.path.join(output_dir, f"{base_filename}_blog_alt1.txt")
        
        with open(blog_alt1_path, "w", encoding="utf-8") as f:
            f.write(results['blog_alt1'])
        logging.info(f"Blog alt1 post saved to: {blog_alt1_path}")
        
        # Generate and write HTML version
        html_content = convert_markdown_to_html(results['blog_alt1'])
        html_path = os.path.join(output_dir, f"{base_filename}_blog_alt1.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML blog alt 1post saved to: {html_path}")
        
        # Generate and write Wiki version
        wiki_content = convert_markdown_to_wiki(results['blog_alt1'])
        wiki_path = os.path.join(output_dir, f"{base_filename}_blog_alt1.wiki")
        with open(wiki_path, "w", encoding="utf-8") as f:
            f.write(wiki_content)
        logging.info(f"Wiki blog alt1 post saved to: {wiki_path}")

def generate_summary_and_blog(transcript: str, prompt: str) -> Optional[str]:
    """
    Generate summary and blog post using OpenAI API.
    
    Args:
        transcript: The transcript text to summarize
        prompt: Instructions for the AI
        
    Returns:
        The generated summary and blog or None if there was an error
    """
    try:
        api_key = ensure_api_key()
        client = OpenAI(api_key=api_key)
        
        estimated_tokens = estimate_token_count(transcript)
        logging.info(f"Estimated transcript length: ~{estimated_tokens} tokens")
        
        # For very large transcripts, use recursive summarization
        if USE_RECURSIVE_SUMMARIZATION and estimated_tokens > MAX_INPUT_TOKENS:
            logging.info(f"Transcript is very large ({estimated_tokens} tokens), using recursive summarization")
            return summarize_large_transcript(transcript, prompt, client)
        
        # Choose appropriate model based on length
        model_to_use = choose_appropriate_model(transcript)
        
        # If transcript is still too long, truncate it
        if estimated_tokens > MAX_INPUT_TOKENS:
            logging.warning(f"Transcript too long (~{estimated_tokens} tokens > {MAX_INPUT_TOKENS} limit), truncating...")
            transcript = truncate_transcript(transcript, MAX_INPUT_TOKENS)
        
        logging.info(f"Sending transcript to OpenAI for summary and blog generation using {model_to_use}...")
        logging.info(f"Estimated input size: ~{estimate_token_count(transcript)} tokens")
        
        try:
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
                    {"role": "user", "content": f"{prompt}\n\nHere's the transcript to summarize:\n\n{transcript}"}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except BadRequestError as e:
            if "maximum context length" in str(e).lower():
                # Try with more aggressive truncation
                logging.warning(f"Context length exceeded. Retrying with further truncation...")
                transcript = truncate_transcript(transcript, MAX_INPUT_TOKENS // 2)
                logging.info(f"Retrying with reduced transcript (~{estimate_token_count(transcript)} tokens)...")
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
                        {"role": "user", "content": f"{prompt}\n\nHere's the transcript to summarize:\n\n{transcript}"}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                return response.choices[0].message.content
            else:
                raise
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        return None

# ==================== VOCABULARY PROCESSING FUNCTIONS ====================
# Dictionary to cache vocabulary mappings
_vocabulary_cache = {}

def load_vocabulary_mappings(vocab_file: str) -> Dict[str, str]:
    """
    Load vocabulary mappings from a JSON file.
    
    Args:
        vocab_file: Path to the vocabulary JSON file
        
    Returns:
        Dictionary of {incorrect_term: correct_term} mappings
    """
    # Return cached vocabulary if available
    if vocab_file in _vocabulary_cache:
        return _vocabulary_cache[vocab_file]
        
    if not os.path.exists(vocab_file):
        logging.warning(f"Vocabulary file not found: {vocab_file}")
        return {}
        
    try:
        # Read file content once to avoid reading twice in case of error
        with open(vocab_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            mappings = json.loads(content)
        except json.JSONDecodeError as json_err:
            # Provide more detailed information about the JSON parsing error
            logging.error(f"JSON syntax error in vocabulary file: {str(json_err)}")
            logging.error(f"Check your vocabulary file for issues near character position {json_err.pos}")
            # Show the problematic part of the JSON
            start_pos = max(0, json_err.pos - 20)
            end_pos = min(len(content), json_err.pos + 20)
            context = content[start_pos:end_pos]
            pointer = ' ' * (min(20, json_err.pos - start_pos)) + '^'
            logging.error(f"Context: ...{context}...")
            logging.error(f"Position: ...{pointer}...")
            return {}
        
        if not isinstance(mappings, dict):
            logging.error(f"Vocabulary file must contain a JSON object/dictionary")
            return {}
        
        # Validate entries - ensure keys and values are strings and not empty
        valid_mappings = {}
        for key, value in mappings.items():
            if not isinstance(key, str) or not key.strip():
                logging.warning(f"Skipping invalid vocabulary mapping key: {key}")
                continue
            if not isinstance(value, str):
                logging.warning(f"Skipping non-string value for key '{key}': {value}")
                continue
            valid_mappings[key] = value
            
        if len(valid_mappings) < len(mappings):
            logging.warning(f"Removed {len(mappings) - len(valid_mappings)} invalid mappings")
            
        # Cache the validated mappings
        _vocabulary_cache[vocab_file] = valid_mappings
        logging.info(f"Loaded {len(valid_mappings)} vocabulary mappings")
        return valid_mappings
    except Exception as e:
        logging.error(f"Error loading vocabulary file: {str(e)}")
        return {}

def apply_vocabulary_corrections(text: str, vocab_mappings: Dict[str, str]) -> str:
    """
    Apply vocabulary corrections to the transcribed text.
    
    Args:
        text: The text to correct
        vocab_mappings: Dictionary of word mappings
        
    Returns:
        Corrected text
    """
    if not vocab_mappings:
        return text
    
    corrected_text = text
    
    # Sort mappings by length (descending) to handle longer phrases first
    sorted_mappings = sorted(vocab_mappings.items(), key=lambda x: len(x[0]), reverse=True)
    
    for incorrect, correct in sorted_mappings:
        # Use word boundaries for more accurate replacement
        pattern = r'\b' + re.escape(incorrect) + r'\b'
        corrected_text = re.sub(pattern, correct, corrected_text, flags=re.IGNORECASE)
    
    return corrected_text

def process_transcript_with_vocabulary(transcript: str) -> str:
    """
    Process a transcript with custom vocabulary corrections.
    
    Args:
        transcript: The transcript text
        
    Returns:
        The processed transcript
    """
    if not USE_CUSTOM_VOCABULARY:
        return transcript
    
    vocab_mappings = load_vocabulary_mappings(VOCABULARY_FILE)
    if not vocab_mappings:
        return transcript
    
    return apply_vocabulary_corrections(transcript, vocab_mappings)


# ==================== FORMAT CONVERSION FUNCTIONS ====================
def convert_markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown text to HTML.
    
    Args:
        markdown_text: The markdown text to convert
        
    Returns:
        HTML formatted text
    """
    try:
        # Use the markdown library to convert text to HTML
        html = markdown.markdown(markdown_text, extensions=['extra', 'nl2br', 'sane_lists'])
        
        # Create a complete HTML document with basic styling
        html_document = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WHYcast Blog</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        a {{
            color: #0066cc;
        }}
        blockquote {{
            border-left: 4px solid #ccc;
            padding-left: 16px;
            margin-left: 0;
            color: #555;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""
        return html_document
    except Exception as e:
        logging.error(f"Error converting markdown to HTML: {str(e)}")
        # Return basic HTML with the original text if conversion fails
        return f"<!DOCTYPE html><html><body><pre>{markdown_text}</pre></body></html>"

def convert_markdown_to_wiki(markdown_text: str) -> str:
    """
    Convert markdown text to Wiki markup.
    
    Args:
        markdown_text: The markdown text to convert
        
    Returns:
        Wiki markup formatted text
    """
    try:
        # Basic conversion rules for common markdown to Wiki syntax
        wiki_text = markdown_text
        
        # Headers: Convert markdown headers to wiki headers
        # e.g., "# Heading 1" -> "= Heading 1 ="
        wiki_text = re.sub(r'^# (.+)$', r'= \1 =', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^## (.+)$', r'== \1 ==', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^### (.+)$', r'=== \1 ===', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^#### (.+)$', r'==== \1 ====', wiki_text, flags=re.MULTILINE)
        
        # Bold: Convert **text** or __text__ to '''text'''
        wiki_text = re.sub(r'\*\*(.+?)\*\*', r"'''\1'''", wiki_text)
        wiki_text = re.sub(r'__(.+?)__', r"'''\1'''", wiki_text)
        
        # Italic: Convert *text* or _text_ to ''text''
        wiki_text = re.sub(r'\*([^*]+?)\*', r"''\1''", wiki_text)
        wiki_text = re.sub(r'_([^_]+?)_', r"''\1''", wiki_text)
        
        # Lists: Convert markdown lists to wiki lists
        # Unordered lists: "- item" -> "* item"
        wiki_text = re.sub(r'^- (.+)$', r'* \1', wiki_text, flags=re.MULTILINE)
        
        # Ordered lists: "1. item" -> "# item"
        wiki_text = re.sub(r'^\d+\. (.+)$', r'# \1', wiki_text, flags=re.MULTILINE)
        
        # Links: Convert [text](url) to [url text]
        wiki_text = re.sub(r'\[(.+?)\]\((.+?)\)', r'[\2 \1]', wiki_text)
        
        # Code blocks: Convert ```code``` to <syntaxhighlight>code</syntaxhighlight>
        wiki_text = re.sub(r'```(.+?)```', r'<syntaxhighlight>\1</syntaxhighlight>', wiki_text, flags=re.DOTALL)
        
        # Inline code: Convert `code` to <code>code</code>
        wiki_text = re.sub(r'`(.+?)`', r'<code>\1</code>', wiki_text)
        
        # Blockquotes: Convert > quote to <blockquote>quote</blockquote>
        # First, group consecutive blockquote lines
        blockquote_blocks = re.findall(r'((?:^> .+\n?)+)', wiki_text, flags=re.MULTILINE)
        for block in blockquote_blocks:
            # Remove the > prefix from each line and wrap in blockquote tags
            cleaned_block = re.sub(r'^> (.+)$', r'\1', block, flags=re.MULTILINE).strip()
            wiki_text = wiki_text.replace(block, f'<blockquote>{cleaned_block}</blockquote>\n\n')
        
        return wiki_text
    except Exception as e:
        logging.error(f"Error converting markdown to Wiki markup: {str(e)}")
        return markdown_text  # Return original text if conversion fails

def convert_existing_blogs(directory: str) -> None:
    """
    Convert all existing blog.txt files in the given directory to HTML and Wiki formats.
    
    Args:
        directory: Directory containing blog text files
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create or access directory {directory}: {str(e)}")
        return
    
    if not os.path.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return
    
    # Look for blog files (txt files that have "_blog" in their name)
    blog_pattern = os.path.join(directory, "*_blog.txt")
    blog_files = glob.glob(blog_pattern)
    
    if not blog_files:
        logging.warning(f"No blog files found in directory: {directory}")
        return
    
    logging.info(f"Found {len(blog_files)} blog files to convert")
    converted_count = 0
    
    for blog_file in tqdm(blog_files, desc="Converting blogs"):
        base_filename = os.path.splitext(blog_file)[0]  # Remove .txt extension
        
        # Define output paths
        html_path = f"{base_filename}.html"
        wiki_path = f"{base_filename}.wiki"
        
        try:
            # Read the blog content
            with open(blog_file, 'r', encoding='utf-8') as f:
                blog_content = f.read()
            
            # Convert to HTML and save
            html_content = convert_markdown_to_html(blog_content)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            # Convert to Wiki and save
            wiki_content = convert_markdown_to_wiki(blog_content)
            with open(wiki_path, "w", encoding="utf-8") as f:
                f.write(wiki_content)
            
            logging.info(f"Converted {os.path.basename(blog_file)} to HTML and Wiki formats")
            converted_count += 1
            
        except Exception as e:
            logging.error(f"Error converting {blog_file}: {str(e)}")
    
    logging.info(f"Successfully converted {converted_count} out of {len(blog_files)} blog files")

# ==================== TRANSCRIPTION FUNCTIONS ====================
def is_cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Log CUDA device information for better diagnostics
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            logging.info(f"CUDA is available: {device_count} device(s) - {device_name}")
        return cuda_available
    except ImportError:
        return False

def setup_model(model_size: str = MODEL_SIZE) -> WhisperModel:
    """
    Initialize and return the Whisper model.
    Automatically uses CUDA if available with optimized settings.
    
    Args:
        model_size: The model size to use
    
    Returns:
        The initialized WhisperModel
    """
    # Override device setting if CUDA is available
    device = "cuda" if is_cuda_available() else DEVICE
    
    # If using CUDA, prefer float16 for better performance unless specified otherwise
    compute_type = "float16" if device == "cuda" and COMPUTE_TYPE == "default" else COMPUTE_TYPE
    
    logging.info(f"Using device: {device} with compute type: {compute_type}")
    if device == "cuda":
        logging.info("CUDA is available - GPU acceleration will be used")
        
        # Try to optimize CUDA performance with environment variables
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking CUDA launches
        try:
            # Check available GPU memory
            import torch
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            logging.info(f"GPU has {gpu_mem:.2f} GB of total memory")
        except Exception as e:
            logging.warning(f"Could not query GPU memory: {e}")
    else:
        logging.warning("CUDA is not available - using CPU which may be significantly slower")
    
    # Create model with optimized parameters
    return WhisperModel(
        model_size, 
        device=device, 
        compute_type=compute_type,
        cpu_threads=8 if device == "cpu" else 0,  # More threads for CPU, default for GPU
        num_workers=2  # Number of workers for faster data loading
    )

def transcribe_audio(model: WhisperModel, audio_file: str) -> Tuple[List, object]:
    """
    Transcribe an audio file using the provided model with optimized settings.
    
    Args:
        model: The WhisperModel to use
        audio_file: Path to the audio file
        
    Returns:
        Tuple of (segments, info)
    """
    logging.info(f"Transcribing {audio_file}...")
    if USE_CUSTOM_VOCABULARY:
        logging.info("Vocabulary corrections will be applied during transcription")
    
    # Monitor GPU memory before transcription if possible
    try:
        if model.device == "cuda":
            import torch
            before_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            logging.info(f"GPU memory in use before transcription: {before_mem:.2f} GB")
    except Exception:
        pass
    
    start_time = datetime.now()
    
    # Use optimized transcription parameters
    result = model.transcribe(
        audio_file,
        beam_size=BEAM_SIZE,
        best_of=5,         # Consider more candidates for better results
        vad_filter=True,   # Voice activity detection to skip silence
        vad_parameters={"min_silence_duration_ms": 500},  # Adjust silence detection
        initial_prompt=None,  # Can set an initial prompt if needed for better context
        condition_on_previous_text=True,  # Use previous text as context
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Temperature fallback for difficult audio
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    audio_duration = get_audio_duration(audio_file)
    if audio_duration > 0:
        speed_factor = audio_duration / duration
        logging.info(f"Transcription completed in {duration:.2f} seconds (audio length: {audio_duration:.2f}s, {speed_factor:.2f}x real-time speed)")
    else:
        logging.info(f"Transcription completed in {duration:.2f} seconds")
    
    # Monitor GPU memory after transcription
    try:
        if model.device == "cuda":
            import torch
            after_mem = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            logging.info(f"GPU memory in use after transcription: {after_mem:.2f} GB")
            # Clear cache immediately after transcription to free memory
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    return result

# New function to get audio duration for performance metrics
def get_audio_duration(audio_file: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Duration in seconds or 0 if cannot be determined
    """
    try:
        import librosa
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception:
        # Try with pydub if librosa fails
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_file)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception:
            return 0  # Return 0 if duration cannot be determined

def cleanup_resources(model=None):
    """
    Clean up resources after processing.
    More aggressive memory cleanup to prevent memory leaks.
    
    Args:
        model: The WhisperModel instance to clean up (if provided)
    """
    try:
        # Clear CUDA cache if available
        import torch
        if torch.cuda.is_available():
            # More aggressive memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Make sure all CUDA operations are complete
            logging.info("CUDA cache cleared")
            
            # Log memory stats
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
            logging.info(f"GPU memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
        
        # Delete model to free up memory
        if model is not None:
            del model
            logging.info("Model resources released")
        
        # Force garbage collection
        import gc
        gc.collect()
    except Exception as e:
        logging.warning(f"Error during resource cleanup: {str(e)}")

def create_output_paths(input_file: str) -> Tuple[str, str, str]:
    """
    Create output file paths based on the input filename.
    
    Args:
        input_file: Path to the input audio file
        
    Returns:
        Tuple of (plain_text_path, timestamped_path, summary_path)   
    """
    base = os.path.splitext(input_file)[0]
    return (
        f"{base}.txt",             # Without timestamps
        f"{base}_ts.txt",          # With timestamps
        f"{base}_summary.txt",     # For summary
    )

def write_transcript_files(segments, output_file: str, output_file_timestamped: str) -> str:
    """
    Write transcript files and return the full transcript.
    
    Args:
        segments: Transcript segments from WhisperModel
        output_file: Path for plain text output
        output_file_timestamped: Path for timestamped output
        
    Returns:
        The full transcript as a string
    """
    full_transcript = ""
    
    with open(output_file, "w", encoding="utf-8") as f_plain, open(output_file_timestamped, "w", encoding="utf-8") as f_timestamped:
        
        for segment in segments:
            # Get the original segment text
            segment_text = segment.text
            
            # Apply vocabulary corrections to segment text if enabled
            if USE_CUSTOM_VOCABULARY:
                segment_text = apply_vocabulary_corrections(segment_text, load_vocabulary_mappings(VOCABULARY_FILE))
            
            # Store for OpenAI processing
            full_transcript += segment_text + "\n"
            
            # Write to plain text file without timestamps
            f_plain.write(segment_text + "\n")
            
            # Write to timestamped file with timestamps
            f_timestamped.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment_text))
            
            # Print to console with timestamps - handle potential encoding errors
            try:
                logging.info("[%.2fs -> %.2fs] %s", segment.start, segment.end, segment_text)
            except UnicodeEncodeError:
                # Fall back to ASCII if Unicode fails
                safe_text = segment_text.encode('ascii', 'replace').decode('ascii')
                logging.info("[%.2fs -> %.2fs] %s", segment.start, segment.end, safe_text)
    
    logging.info(f"Transcription saved to: {output_file}")
    logging.info(f"Timestamped transcription saved to: {output_file_timestamped}")
    
    # Apply a final pass of vocabulary corrections to the full transcript if enabled
    # This catches any patterns that might span across segments
    if USE_CUSTOM_VOCABULARY:
        original_transcript = full_transcript
        full_transcript = apply_vocabulary_corrections(full_transcript, load_vocabulary_mappings(VOCABULARY_FILE))
        
        # If the final pass made additional corrections, update the saved files
        if full_transcript != original_transcript:
            logging.info("Applying final vocabulary correction pass to catch cross-segment patterns")
            with open(output_file, "w", encoding="utf-8") as f_plain:
                f_plain.write(full_transcript)
            
            # Simplification: regenerate timestamped file from full transcript
            # This is imperfect since timestamps might not perfectly align with corrected text
            lines = full_transcript.split('\n')
            timestamped_lines = []
            with open(output_file_timestamped, "r", encoding="utf-8") as f_original:
                original_timestamped = f_original.readlines()
                
            # Try to maintain timestamp information while updating text
            for i, line in enumerate(lines):
                if i < len(original_timestamped):
                    ts_line = original_timestamped[i]
                    ts_match = re.match(r'^\[\s*(\d+\.\d+)s\s*->\s*(\d+\.\d+)s\s*\]', ts_line)
                    if ts_match and line.strip():
                        start, end = ts_match.groups()
                        timestamped_lines.append(f"[{start}s -> {end}s] {line}")
                    elif line.strip():
                        timestamped_lines.append(line)
            
            with open(output_file_timestamped, "w", encoding="utf-8") as f_updated:
                f_updated.write('\n'.join(timestamped_lines))
    
    return full_transcript

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

def transcription_exists(input_file: str) -> bool:
    """
    Check if a transcription already exists for the given audio file.
    
    Args:
        input_file: Path to the input audio file    
        
    Returns:
        True if transcription exists, False otherwise
    """
    base = os.path.splitext(input_file)[0]
    transcript_file = f"{base}.txt"
    return os.path.exists(transcript_file)

def regenerate_summary(transcript_file: str) -> bool:
    """
    Regenerate summary and blog from an existing transcript file.
    
    Args:
        transcript_file: Path to the transcript file
        
    Returns:
        Success status (True/False)
    """
    if not os.path.exists(transcript_file):
        logging.error(f"Transcript file does not exist: {transcript_file}")
        return False
    
    try:
        # Read the transcript
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Create paths for output files
        base = os.path.splitext(transcript_file)[0]
        summary_file = f"{base}_summary.txt"
        blog_file = f"{base}_blog.txt"
        
        # Generate and save summary and blog
        return process_summary(transcript, summary_file, blog_file)    
    except Exception as e:
        logging.error(f"Error regenerating summary: {str(e)}")
        return False

def regenerate_all_summaries(directory: str) -> None:
    """
    Regenerate summaries and blogs for all transcript files in the directory.
    
    Args:
        directory: Directory containing transcript files
    """
    # Create directory if it doesn't exist (especially for default 'podcasts' directory)
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create or access directory {directory}: {str(e)}")
        return
    
    if not os.path.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return
    
    # Look for transcript files (txt files that don't have "_ts" or "_summary" in their name)
    files = glob.glob(os.path.join(directory, "*.txt"))
    transcript_files = [f for f in files if "_ts.txt" not in f and "_summary.txt" not in f]
    
    if not transcript_files:
        logging.warning(f"No transcript files found in directory: {directory}")
        return
    
    logging.info(f"Found {len(transcript_files)} transcript files to process")
    for file in tqdm(transcript_files, desc="Regenerating summaries"):
        logging.info(f"Regenerating summary for: {file}")
        regenerate_summary(file)
        # Force garbage collection to release GPU memory
        import gc
        gc.collect()

def regenerate_blog_only(transcript_file: str, summary_file: str) -> bool:
    """
    Regenerate only the blog post from existing transcript and summary files.
    
    Args:
        transcript_file: Path to the transcript file
        summary_file: Path to the summary file
        
    Returns:
        Success status (True/False)
    """
    if not os.path.exists(transcript_file):
        logging.error(f"Transcript file does not exist: {transcript_file}")
        return False
        
    if not os.path.exists(summary_file):
        logging.error(f"Summary file does not exist: {summary_file}")
        return False
    
    try:
        # Read the transcript and summary
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
            
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = f.read()
        
        # Get base filename without path and extension
        base_filename = os.path.splitext(os.path.basename(transcript_file))[0]
        # Remove any _cleaned suffix if present for base name consistency
        if base_filename.endswith('_cleaned'):
            base_filename = base_filename[:-8]  # remove "_cleaned"
        
        # Create path for output directory (same as transcript file directory)
        output_dir = os.path.dirname(transcript_file)
        # Create the consistent blog file path
        blog_file = os.path.join(output_dir, f"{base_filename}_blog.txt")
        
        # Create the consistent blog alt1 file path
        blog_alt1_file = os.path.join(output_dir, f"{base_filename}_blog_alt1.txt")
        
        # Read the blog prompt
        blog_prompt = read_prompt_file(PROMPT_BLOG_FILE)
        if not blog_prompt:
            logging.warning("Blog prompt file not found or empty, cannot regenerate blog")
            return False

        logging.info("Generating blog post...")
        
        # Combine transcript and summary for input
        input_text = f"CLEANED TRANSCRIPT:\n{transcript}\n\nSUMMARY:\n{summary}"
        model_to_use = choose_appropriate_model(input_text)
        

        api_key = ensure_api_key()
        client = OpenAI(api_key=api_key)
            
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates blog posts from transcripts."},
                {"role": "user", "content": f"{blog_prompt}\n\n{input_text}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS * 2  # Allow more tokens for blog generation
        )
        blog = response.choices[0].message.content
        
        # Save the blog post to the correct path
        with open(blog_file, 'w', encoding='utf-8') as f:
            f.write(blog)
            logging.info(f"Blog post saved to: {blog_file}")
        
        
        # Only generate alternative blog post if the alternative prompt file exists
        if os.path.isfile(PROMPT_BLOG_ALT1_FILE):
            # Creating the alternative blogpost
            logging.info("Generating blog alt 1 post...")
            
            # Read the blog prompt
            blog_alt1_prompt = read_prompt_file(PROMPT_BLOG_ALT1_FILE)
            if blog_alt1_prompt:
                # Combine transcript and summary for input
                input_text = f"CLEANED TRANSCRIPT:\n{transcript}\n\nSUMMARY:\n{summary}"
                model_to_use = choose_appropriate_model(input_text)
                
                api_key = ensure_api_key()
                client = OpenAI(api_key=api_key)
                    
                #use the openai api to generate a second blog post
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates blog posts from transcripts."},
                    {"role": "user", "content": f"{blog_alt1_prompt}\n\n{input_text}"}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS * 2  # Allow more tokens for blog generation
                )
                blog_alt1 = response.choices[0].message.content
                
                # Save the blog post to the correct path
                with open(blog_alt1_file, 'w', encoding='utf-8') as f:
                    f.write(blog_alt1)
                    logging.info(f"Blog alt1 post saved to: {blog_alt1_file}")
            else:
                logging.warning("Blog prompt alt1 file exists but is empty, skipping blog alt1 generation")
        else:
            logging.info(f"Blog prompt alt1 file {PROMPT_BLOG_ALT1_FILE} not found, skipping alternative blog generation")
        
        # Generate and save HTML version
        logging.info(f"Converting blog post to HTML format...")    
        html_content = convert_markdown_to_html(blog)
        html_path = os.path.join(output_dir, f"{base_filename}_blog.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            logging.info(f"HTML blog post saved to: {html_path}")
            
        # Generate and save Wiki version
        logging.info(f"Converting blog post to Wiki format...")    
        wiki_content = convert_markdown_to_wiki(blog)
        wiki_path = os.path.join(output_dir, f"{base_filename}_blog.wiki")
        with open(wiki_path, "w", encoding="utf-8") as f:
            f.write(wiki_content)
            logging.info(f"Wiki blog post saved to: {wiki_path}")
        
        return True
                        
    except Exception as e:
        logging.error(f"Error regenerating blog: {str(e)}")
        return False

def regenerate_all_blogs(directory: str) -> None:
    """
    Regenerate blog posts for all transcript/summary pairs in the directory.
    
    Args:
        directory: Directory containing transcript and summary files
    """
    # Create directory if it doesn't exist (especially for default 'podcasts' directory)
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create or access directory {directory}: {str(e)}")
        return
    
    if not os.path.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return
    
    # Look for transcript files (txt files that don't have "_ts", "_summary", or "_blog" in their name)
    files = glob.glob(os.path.join(directory, "*.txt"))
    transcript_files = [f for f in files if "_ts.txt" not in f and "_summary.txt" not in f and "_blog.txt" not in f and "_cleaned.txt" not in f]
    
    if not transcript_files:
        logging.warning(f"No transcript files found in directory: {directory}")
        return
    
    logging.info(f"Found {len(transcript_files)} transcript files to process")
    processed_count = 0
    
    for transcript_file in tqdm(transcript_files, desc="Regenerating blogs"):
        base = os.path.splitext(transcript_file)[0]
        summary_file = f"{base}_summary.txt"
        
        # First check for a cleaned transcript if it exists
        cleaned_transcript_file = f"{base}_cleaned.txt"
        if os.path.exists(cleaned_transcript_file):
            transcript_source = cleaned_transcript_file
        else:
            transcript_source = transcript_file
        
        # Check if summary exists
        if os.path.exists(summary_file):
            logging.info(f"Regenerating blog for: {transcript_file}")
            if regenerate_blog_only(transcript_source, summary_file):
                processed_count += 1
            # Force garbage collection to release memory
            import gc
            gc.collect()
        else:
            logging.warning(f"Skipping {transcript_file}: No matching summary file found")
    
    # Also regenerate HTML and Wiki versions if blog regeneration was successful
    if processed_count > 0:
        logging.info("Ensuring HTML and Wiki versions exist for all blog posts")
        blog_files = glob.glob(os.path.join(directory, "*_blog.txt"))
        for blog_file in blog_files:
            base_filename = os.path.splitext(os.path.basename(blog_file))[0].replace('_blog', '')
            try:
                with open(blog_file, 'r', encoding='utf-8') as f:
                    blog_content = f.read()
                
                # Generate HTML version
                html_path = os.path.join(directory, f"{base_filename}_blog.html")
                if not os.path.exists(html_path):
                    html_content = convert_markdown_to_html(blog_content)
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    logging.info(f"Generated missing HTML version: {html_path}")
                
                # Generate Wiki version
                wiki_path = os.path.join(directory, f"{base_filename}_blog.wiki")
                if not os.path.exists(wiki_path):
                    wiki_content = convert_markdown_to_wiki(blog_content)
                    with open(wiki_path, "w", encoding="utf-8") as f:
                        f.write(wiki_content)
                    logging.info(f"Generated missing Wiki version: {wiki_path}")
            except Exception as e:
                logging.error(f"Error generating formats for {blog_file}: {str(e)}")
    
    logging.info(f"Successfully regenerated {processed_count} blog posts out of {len(transcript_files)} transcript files")

# ==================== PODCAST FEED FUNCTIONS ====================
def get_latest_episode(feed_url: str) -> Optional[Dict]:
    """
    Get the latest episode from a podcast RSS feed.
    
    Args:
        feed_url: URL of the RSS feed
        
    Returns:
        Dictionary with episode info or None if failed
    """
    try:
        logging.info(f"Fetching podcast feed from {feed_url}")
        feed = feedparser.parse(feed_url)
        
        if not feed.entries:
            logging.warning("No episodes found in the feed")
            return None
            
        # Get the latest episode (first entry)
        latest = feed.entries[0]
        
        # Find the audio file URL
        audio_url = None
        for enclosure in latest.enclosures:
            if enclosure.type.startswith('audio/'):
                audio_url = enclosure.href
                break
                
        if not audio_url:
            logging.warning("No audio file found in the latest episode")
            return None
            
        return {
            'title': latest.title,
            'published': latest.published if hasattr(latest, 'published') else '',
            'audio_url': audio_url,
            'description': latest.description if hasattr(latest, 'description') else '',
            'guid': latest.id if hasattr(latest, 'id') else ''
        }
    except Exception as e:
        logging.error(f"Error fetching podcast feed: {str(e)}")
        return None

def get_episode_filename(episode: Dict) -> str:
    """
    Generate a filename for the episode based on its title.
    
    Args:
        episode: Episode dictionary from get_latest_episode()
        
    Returns:
        Sanitized filename
    """
    # Clean the title to make it suitable for a filename
    title = episode['title'].lower()
    # Replace special chars with underscores
    title = re.sub(r'[^\w\s-]', '_', title)
    # Replace whitespace with underscores
    title = re.sub(r'\s+', '_', title)
    return f"{title}.mp3"

def episode_already_processed(episode: Dict, download_dir: str) -> bool:
    """
    Check if an episode has already been processed.
    
    Args:
        episode: Episode dictionary
        download_dir: Directory where episodes are stored
        
    Returns:
        True if already processed, False otherwise
    """
    # Check if the episode ID is in our tracking file
    processed_episodes = load_processed_episodes(download_dir)
    episode_id = get_episode_id(episode)
    if episode_id in processed_episodes:
        return True
    
    # Also check if the file exists (legacy method)
    filename = get_episode_filename(episode)
    full_path = os.path.join(download_dir, filename)
    
    if os.path.exists(full_path):
        return True
        
    # Check if a transcription of this file exists
    transcript_file = os.path.splitext(full_path)[0] + ".txt"
    if os.path.exists(transcript_file):
        return True
    
    return False

def download_latest_episode(feed_url: str, download_dir: str) -> Optional[str]:
    """
    Download the latest episode from the podcast feed if not already processed.
    
    Args:
        feed_url: URL of the RSS feed
        download_dir: Directory to save the downloaded file
        
    Returns:
        Path to the downloaded file or None if no new episode or error
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Get latest episode info
    episode = get_latest_episode(feed_url)
    if not episode:
        logging.warning("No episode found to download")
        return None
    
    # Check if already processed
    if episode_already_processed(episode, download_dir):
        logging.info(f"Episode '{episode['title']}' has already been processed, skipping")
        return None
    
    # Generate filename
    filename = get_episode_filename(episode)
    full_path = os.path.join(download_dir, filename)
    
    # Download the file
    try:
        logging.info(f"Downloading episode: {episode['title']}")
        response = requests.get(episode['audio_url'], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(full_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress_bar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
        
        logging.info(f"Downloaded episode to {full_path}")
        return full_path
    except Exception as e:
        logging.error(f"Error downloading episode: {str(e)}")
        # Clean up partial download if it exists
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except:
                pass
        return None

def get_all_episodes(feed_url: str) -> List[Dict]:
    """
    Get all episodes from a podcast RSS feed.
    
    Args:
        feed_url: URL of the RSS feed
        
    Returns:
        List of dictionaries with episode info
    """
    try:
        logging.info(f"Fetching podcast feed from {feed_url}")
        feed = feedparser.parse(feed_url)
        
        if not feed.entries:
            logging.warning("No episodes found in the feed")
            return []
        
        episodes = []
        for entry in feed.entries:
            # Find the audio file URL
            audio_url = None
            for enclosure in entry.enclosures:
                if enclosure.type.startswith('audio/'):
                    audio_url = enclosure.href
                    break
                    
            if not audio_url:
                logging.warning(f"No audio file found for episode: {entry.title}")
                continue
                
            episodes.append({
                'title': entry.title,
                'published': entry.published if hasattr(entry, 'published') else '',
                'audio_url': audio_url,
                'description': entry.description if hasattr(entry, 'description') else '',
                'guid': entry.id if hasattr(entry, 'id') else ''
            })
        
        logging.info(f"Found {len(episodes)} episodes in the feed")
        return episodes
    except Exception as e:
        logging.error(f"Error fetching podcast feed: {str(e)}")
        return []

def load_processed_episodes(download_dir: str) -> set:
    """
    Load the set of already processed episode IDs.
    
    Args:
        download_dir: Directory where episodes are stored
        
    Returns:
        Set of processed episode IDs
    """
    processed_file = os.path.join(download_dir, "processed_episodes.txt")
    processed = set()
    
    if os.path.exists(processed_file):
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    processed.add(line.strip())
            logging.info(f"Loaded {len(processed)} previously processed episodes")
        except Exception as e:
            logging.error(f"Error loading processed episodes: {str(e)}")
    
    return processed

def mark_episode_processed(episode: Dict, download_dir: str) -> None:
    """
    Mark an episode as processed by storing its ID.
    
    Args:
        episode: Episode dictionary
        download_dir: Directory where episodes are stored
    """
    # Create a unique identifier for the episode
    if 'guid' in episode and episode['guid']:
        episode_id = episode['guid']
    else:
        # Create hash from title and published date
        episode_id = hashlib.md5(f"{episode['title']}-{episode['published']}".encode()).hexdigest()
    
    processed_file = os.path.join(download_dir, "processed_episodes.txt")
    try:
        with open(processed_file, 'a', encoding='utf-8') as f:
            f.write(f"{episode_id}\n")
    except Exception as e:
        logging.error(f"Error marking episode as processed: {str(e)}")

def get_episode_id(episode: Dict) -> str:
    """
    Get a unique ID for an episode.
    
    Args:
        episode: Episode dictionary
        
    Returns:
        Unique identifier for the episode
    """
    if 'guid' in episode and episode['guid']:
        return episode['guid']
    return hashlib.md5(f"{episode['title']}-{episode['published']}".encode()).hexdigest()

def process_all_episodes(feed_url: str, download_dir: str, **kwargs) -> None:
    """
    Download and process all episodes from the podcast feed.
    
    Args:
        feed_url: URL of the RSS feed
        download_dir: Directory to save downloaded files
        **kwargs: Additional arguments to pass to main()
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Get all episodes
    episodes = get_all_episodes(feed_url)
    if not episodes:
        logging.warning("No episodes found to process")
        return
    
    # Load list of processed episodes
    processed_episodes = load_processed_episodes(download_dir)
    
    # Filter out already processed episodes
    new_episodes = []
    for episode in episodes:
        episode_id = get_episode_id(episode)
        if episode_id not in processed_episodes:
            new_episodes.append(episode)
    
    if not new_episodes:
        logging.info("All episodes have already been processed")
        return
    
    logging.info(f"Found {len(new_episodes)} episodes to process")
    
    # Sort episodes by publication date if available (newest first is typical feed order)
    try:
        from dateutil import parser as date_parser
        new_episodes.sort(key=lambda ep: date_parser.parse(ep['published']) if ep['published'] else datetime.now(), 
                        reverse=True)
    except ImportError:
        logging.warning("dateutil module not available, episodes will be processed in feed order")
    
    # Initialize the model once and reuse it for all episodes
    model_size = kwargs.get('model_size') or MODEL_SIZE
    model = setup_model(model_size)
    logging.info(f"Initialized {model_size} model for processing {len(new_episodes)} episodes")
    
    # Process each episode
    for i, episode in enumerate(new_episodes, 1):
        logging.info(f"Processing episode {i}/{len(new_episodes)}: {episode['title']}")
        
        # Generate filename
        filename = get_episode_filename(episode)
        full_path = os.path.join(download_dir, filename)
        
        # Track processing status
        processing_success = False
        
        # Download the file
        try:
            logging.info(f"Downloading episode: {episode['title']}")
            response = requests.get(episode['audio_url'], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(full_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress_bar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            progress_bar.update(len(chunk))
            
            logging.info(f"Downloaded episode to {full_path}")
            
            # Process the downloaded file - pass the model instance
            kwargs_with_model = dict(kwargs)
            kwargs_with_model['model'] = model  # Pass the model to main()
            main(full_path, is_batch_mode=True, **kwargs_with_model)
            
            # If we got this far, processing was successful
            processing_success = True
            
            # Clean GPU memory but keep the model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logging.info("CUDA cache cleared between episodes")
            except Exception as e:
                logging.warning(f"Error clearing CUDA cache: {str(e)}")
                
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logging.error(f"Error processing episode {episode['title']}: {str(e)}")
            # Clean up partial download if it exists
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except:
                    pass
        
        # Only mark as processed if we were successful
        if processing_success:
            mark_episode_processed(episode, download_dir)
            logging.info(f"Successfully processed episode: {episode['title']}")
        else:
            logging.warning(f"Failed to process episode: {episode['title']}, will retry next time")
    
    # Final cleanup of model after all episodes are processed
    cleanup_resources(model)
    logging.info("Completed processing all episodes")

def main(input_file: str, model_size: Optional[str] = None, output_dir: Optional[str] = None, 
         skip_summary: bool = False, force: bool = False, is_batch_mode: bool = False,
         regenerate_summary_only: bool = False, skip_vocabulary: bool = False,
         model: Optional[WhisperModel] = None) -> None:
    """
    Main function to process an audio file.
    
    Args:
        input_file: Path to the input audio file
        model_size: Override default model size if provided
        output_dir: Directory to save output files (defaults to input file directory)
        skip_summary: Flag to skip summary generation
        force: Flag to force regeneration of transcription even if it exists
        is_batch_mode: Flag indicating if running as part of batch processing
        regenerate_summary_only: Flag to only regenerate summary from existing transcript file
        skip_vocabulary: Flag to skip vocabulary corrections
        model: Optional pre-initialized WhisperModel to use (to avoid reloading)
    """
    try:
        # Process for summary regeneration
        if regenerate_summary_only:
            if input_file.endswith('.txt') and 'summary' not in input_file:
                # Input is already a transcript file
                transcript_file = input_file
                logging.info(f"Regenerating summary from transcript: {transcript_file}")
                success = regenerate_summary(transcript_file)
                if not success and not is_batch_mode:
                    sys.exit(1)
                return
            else:
                # Input is an audio file, construct the transcript filename
                transcript_file = os.path.splitext(input_file)[0] + ".txt"
                if not os.path.exists(transcript_file):
                    logging.error(f"Cannot regenerate summary: transcript file {transcript_file} does not exist")
                    if not is_batch_mode:
                        sys.exit(1)
                    return
                logging.info(f"Regenerating summary from transcript: {transcript_file}")
                success = regenerate_summary(transcript_file)
                if not success and not is_batch_mode:
                    sys.exit(1)
                return
        
        # Check input file
        if not os.path.exists(input_file):
            logging.error(f"Input file does not exist: {input_file}")
            if not is_batch_mode:
                sys.exit(1)
            return
        
        # Skip if transcription already exists and not forcing
        if not force and transcription_exists(input_file):
            logging.info(f"Skipping {input_file} - transcription already exists (use --force to override)")
            return
        
        # Setup output directory and create if necessary
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(os.path.splitext(input_file)[0])
            output_base = os.path.join(output_dir, base_name)
        else:
            output_base = os.path.splitext(input_file)[0]
        
        output_file = f"{output_base}.txt"
        output_file_timestamped = f"{output_base}_ts.txt"
        output_summary_file = f"{output_base}_summary.txt"
        output_blog_file = f"{output_base}_blog.txt"
        
        # Setup model if not provided
        if model is None:
            model_size = model_size or MODEL_SIZE
            model = setup_model(model_size)
        
        # Transcribe
        segments, info = transcribe_audio(model, input_file)
        logging.info(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        # Write transcript files
        full_transcript = write_transcript_files(segments, output_file, output_file_timestamped)
        
        # Check output file size before summary
        if not skip_summary and os.path.exists(output_file):
            file_size_kb = os.path.getsize(output_file) / 1024
            logging.info(f"Transcript file size: {file_size_kb:.1f} KB")
            check_file_size(output_file)
            # Extra warning for extremely large files
            if file_size_kb > MAX_FILE_SIZE_KB * 2:
                logging.warning(f"File is extremely large ({file_size_kb:.1f} KB). Processing may take significant time.")
        
        # Generate and save summary (if not skipped)
        if not skip_summary:
            process_summary(full_transcript, output_summary_file, output_blog_file)
    except Exception as e:
        logging.error(f"An error occurred processing {input_file}: {str(e)}")
        if not is_batch_mode:
            sys.exit(1)
        # In batch mode, we continue to the next file

def process_batch(input_pattern: str, **kwargs) -> None:
    """
    Process multiple files matching the given pattern.
    
    Args:
        input_pattern: Glob pattern for input files
        **kwargs: Additional arguments to pass to main()
    """
    files = glob.glob(input_pattern)
    if not files:
        logging.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logging.info(f"Found {len(files)} files to process")
    model = setup_model(kwargs.get('model_size') or MODEL_SIZE)
    for file in tqdm(files, desc="Processing files"):
        logging.info(f"Processing file: {file}")
        # Pass the model instance to avoid reloading for each file
        kwargs_with_model = dict(kwargs)
        kwargs_with_model['model'] = model
        main(file, is_batch_mode=True, **kwargs_with_model)
        
    # Clean up after processing all files
    cleanup_resources(model)

def cleanup_resources(model=None):
    """
    Clean up resources after processing.
    
    Args:
        model: The WhisperModel instance to clean up (if provided)
    """
    try:
        # Clear CUDA cache if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("CUDA cache cleared")
        
        # Delete model to free up memory
        if model is not None:
            del model
            logging.info("Model resources released")
        
        # Force garbage collection
        import gc
        gc.collect()
    except Exception as e:
        logging.warning(f"Error during resource cleanup: {str(e)}")


def process_all_mp3s(directory: str, **kwargs) -> None:
    """
    Process all MP3 files in the given directory.
    
    Args:
        directory: Directory containing MP3 files
        **kwargs: Additional arguments to pass to main()
    """
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory) or '.'
    mp3_pattern = os.path.join(directory, "*.mp3")
    files = glob.glob(mp3_pattern)
    
    if not files:
        logging.warning(f"No MP3 files found in directory: {directory}")
        return
    
    logging.info(f"Found {len(files)} MP3 files to process")
    for file in tqdm(files, desc="Processing MP3 files"):
        logging.info(f"Processing file: {file}")
        main(file, is_batch_mode=True, **kwargs)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'WHYcast Transcribe v{VERSION} - Transcribe audio files and generate summaries')
    parser.add_argument('input', nargs='?', help='Path to the input audio file, directory, or glob pattern (default: current directory for regeneration options)')
    parser.add_argument('--batch', '-b', action='store_true', help='Process multiple files matching pattern')
    parser.add_argument('--all-mp3s', '-a', action='store_true', help='Process all MP3 files in directory')
    parser.add_argument('--model', '-m', help='Model size (e.g., "large-v3", "medium", "small")')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files')
    parser.add_argument('--skip-summary', '-s', action='store_true', help='Skip summary generation')
    parser.add_argument('--force', '-f', action='store_true', help='Force regeneration of transcriptions even if they exist')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--version', action='version', version=f'WHYcast Transcribe v{VERSION}')
    parser.add_argument('--regenerate-summary', '-r', action='store_true', help='Regenerate summary and blog from existing transcript')
    parser.add_argument('--regenerate-all-summaries', '-R', action='store_true', 
                         help='Regenerate summaries for all transcripts in directory (uses current dir if no input)')
    parser.add_argument('--regenerate-all-blogs', '-B', action='store_true', 
                         help='Regenerate only blog posts for all transcripts in directory (uses current dir if no input)')
    parser.add_argument('--skip-vocabulary', action='store_true', help='Skip custom vocabulary corrections')
    
    # Add podcast feed arguments
    parser.add_argument('--feed', '-F', default="https://whycast.podcast.audio/@whycast/feed.xml", 
                       help='RSS feed URL to download latest episode (default: WHYcast feed)')
    parser.add_argument('--download-dir', '-D', default='podcasts', 
                       help='Directory to save downloaded episodes (default: podcasts)')
    parser.add_argument('--no-download', '-N', action='store_true', 
                       help='Disable automatic podcast download')
    
    # Add new argument for processing all episodes
    parser.add_argument('--all-episodes', '-A', action='store_true', 
                       help='Process all episodes from the podcast feed instead of just the latest')
    
    # Add new argument for converting existing blogs to HTML and Wiki formats
    parser.add_argument('--convert-blogs', '-C', action='store_true',
                       help='Convert existing blog text files to HTML and Wiki formats')
    
    args = parser.parse_args()
    
    logging.info(f"WHYcast Transcribe {VERSION} starting up")
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if we should convert blogs
    if args.convert_blogs:
        directory = args.input if args.input else args.download_dir
        logging.info(f"Converting all blog files in directory: {directory}")
        convert_existing_blogs(directory)
        sys.exit(0)
    
    # Check if we should process all episodes
    should_process_all_episodes = (args.all_episodes and 
                                not args.no_download and
                                not args.input and 
                                not args.regenerate_all_summaries and
                                not args.regenerate_all_blogs and
                                not args.regenerate_summary)
    
    if should_process_all_episodes:
        feed_url = args.feed
        download_dir = args.download_dir
        
        logging.info(f"Processing all episodes from {feed_url}")
        process_all_episodes(feed_url, download_dir, model_size=args.model, 
                           output_dir=args.output_dir, skip_summary=args.skip_summary, 
                           force=args.force)
        sys.exit(0)
    
    # Determine if we should check the podcast feed for latest episode (original behavior)
    should_check_feed = (not args.no_download and 
                         not args.all_episodes and
                         not args.input and 
                         not args.regenerate_all_summaries and
                         not args.regenerate_all_blogs and
                         not args.regenerate_summary)
    
    if should_check_feed:
        feed_url = args.feed
        download_dir = args.download_dir
        
        logging.info(f"Checking for the latest episode from {feed_url}")
        episode_file = download_latest_episode(feed_url, download_dir)
        
        if episode_file:
            logging.info(f"Processing newly downloaded episode: {episode_file}")
            main(episode_file, model_size=args.model, output_dir=args.output_dir, 
                 skip_summary=args.skip_summary, force=args.force)
            sys.exit(0)
        else:
            logging.info("No new episode to download or process")
            sys.exit(0)
    
    # Continue with existing functionality if input is provided or special mode is requested
    if args.regenerate_all_blogs:
        # Allow regenerate_all_blogs without an input by using podcasts directory
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating all blogs in directory: {directory}")
        regenerate_all_blogs(directory)
    elif args.regenerate_all_summaries:
        # Allow regenerate_all_summaries without an input by using podcasts directory
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating all summaries in directory: {directory}")
        regenerate_all_summaries(directory)
    elif not args.input:
        parser.print_help()
        sys.exit(1)
    elif args.regenerate_summary:
        if os.path.isdir(args.input):
            logging.error("Please specify a transcript file when using --regenerate-summary, not a directory")
            sys.exit(1)
        elif not os.path.isfile(args.input):
            logging.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        else:
            main(args.input, regenerate_summary_only=True, skip_vocabulary=args.skip_vocabulary)
    elif args.all_mp3s:
        process_all_mp3s(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
                         force=args.force, skip_vocabulary=args.skip_vocabulary)
    elif args.batch:
        process_batch(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
                      force=args.force, skip_vocabulary=args.skip_vocabulary)
    else:
        main(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
             force=args.force, regenerate_summary_only=args.regenerate_summary,
             skip_vocabulary=args.skip_vocabulary)
