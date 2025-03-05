#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WHYcast Transcribe - v0.0.1

A tool for transcribing audio files and generating summaries using OpenAI GPT models.

Copyright (c) 2025 Robert van den Breemen
License: MIT (see LICENSE file for details)
"""

import os
import sys
import logging
from typing import Tuple, List, Optional
from dotenv import load_dotenv
from tqdm import tqdm
import glob
import argparse
from openai import OpenAI, BadRequestError
from faster_whisper import WhisperModel

# Import configuration
from config import (
    VERSION, MODEL_SIZE, DEVICE, COMPUTE_TYPE, BEAM_SIZE,
    OPENAI_MODEL, OPENAI_LARGE_CONTEXT_MODEL, 
    TEMPERATURE, MAX_TOKENS, MAX_INPUT_TOKENS, CHARS_PER_TOKEN,
    PROMPT_FILE, MAX_FILE_SIZE_KB,
    USE_RECURSIVE_SUMMARIZATION, MAX_CHUNK_SIZE, CHUNK_OVERLAP
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

def read_summary_prompt(prompt_file: str = PROMPT_FILE) -> Optional[str]:
    """
    Read the summary prompt from file.
    
    Args:
        prompt_file: Path to the prompt file
        
    Returns:
        The prompt text or None if there was an error
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading prompt file: {str(e)}")
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

# ==================== TRANSCRIPTION FUNCTIONS ====================
def setup_model(model_size: str = MODEL_SIZE) -> WhisperModel:
    """
    Initialize and return the Whisper model.
    
    Args:
        model_size: The model size to use
    
    Returns:
        The initialized WhisperModel
    """
    return WhisperModel(model_size, device=DEVICE, compute_type=COMPUTE_TYPE)

def transcribe_audio(model: WhisperModel, audio_file: str) -> Tuple[List, object]:
    """
    Transcribe an audio file using the provided model.
    
    Args:
        model: The WhisperModel to use
        audio_file: Path to the audio file
        
    Returns:
        Tuple of (segments, info)
    """
    logging.info(f"Transcribing {audio_file}...")
    return model.transcribe(
        audio_file,
        beam_size=BEAM_SIZE
        # Removed progress_callback parameter as it's not supported in this version of faster-whisper
    )

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
        f"{base}_summary.txt"      # For summary and blog
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
    
    with open(output_file, "w", encoding="utf-8") as f_plain, \
         open(output_file_timestamped, "w", encoding="utf-8") as f_timestamped:
        
        for segment in segments:
            # Print to console with timestamps
            logging.info("[%.2fs -> %.2fs] %s", segment.start, segment.end, segment.text)
            
            # Store for OpenAI processing
            full_transcript += segment.text + "\n"
            
            # Write to plain text file without timestamps
            f_plain.write(segment.text + "\n")
                
            # Write to timestamped file with timestamps
            f_timestamped.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
    
    logging.info(f"Transcription saved to: {output_file}")
    logging.info(f"Timestamped transcription saved to: {output_file_timestamped}")
    
    return full_transcript

def process_summary(full_transcript: str, output_summary_file: str) -> bool:
    """
    Process the transcript to generate and save a summary.
    
    Args:
        full_transcript: The complete transcript text
        output_summary_file: Path to save the summary
        
    Returns:
        Success status (True/False)
    """
    summary_prompt = read_summary_prompt()
    if not summary_prompt:
        logging.warning("Summary prompt file not found or empty, skipping summary generation")
        return False
        
    logging.info("Generating summary and blog post...")
    summary_and_blog = generate_summary_and_blog(full_transcript, summary_prompt)
    
    if not summary_and_blog:
        logging.error("Failed to generate summary and blog post")
        return False
        
    # Save summary and blog to file
    with open(output_summary_file, "w", encoding="utf-8") as f_summary:
        f_summary.write(summary_and_blog)
    logging.info(f"Summary and blog post saved to: {output_summary_file}")
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

# ==================== MAIN FUNCTION ====================
def main(input_file: str, model_size: Optional[str] = None, output_dir: Optional[str] = None, 
         skip_summary: bool = False, force: bool = False) -> None:
    """
    Main function to process an audio file.
    
    Args:
        input_file: Path to the input audio file
        model_size: Override default model size if provided
        output_dir: Directory to save output files (defaults to input file directory)
        skip_summary: Flag to skip summary generation
        force: Flag to force regeneration of transcription even if it exists
    """
    try:
        # Check input file
        if not os.path.exists(input_file):
            logging.error(f"Input file does not exist: {input_file}")
            sys.exit(1)
        
        # Skip if transcription already exists and not forcing
        if not force and transcription_exists(input_file):
            logging.info(f"Skipping {input_file} - transcription already exists (use --force to override)")
            return
            
        # Use provided model size or MODEL_SIZE
        model_size = model_size or MODEL_SIZE
        
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
        
        # Setup model
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
            process_summary(full_transcript, output_summary_file)
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

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
    for file in tqdm(files, desc="Processing files"):
        logging.info(f"Processing file: {file}")
        main(file, **kwargs)

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
        main(file, **kwargs)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'WHYcast Transcribe v{VERSION} - Transcribe audio files and generate summaries')
    parser.add_argument('input', help='Path to the input audio file, directory, or glob pattern')
    parser.add_argument('--batch', '-b', action='store_true', help='Process multiple files matching pattern')
    parser.add_argument('--all-mp3s', '-a', action='store_true', help='Process all MP3 files in directory')
    parser.add_argument('--model', '-m', help='Model size (e.g., "large-v3", "medium", "small")')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files')
    parser.add_argument('--skip-summary', '-s', action='store_true', help='Skip summary generation')
    parser.add_argument('--force', '-f', action='store_true', help='Force regeneration of transcriptions even if they exist')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--version', action='version', version=f'WHYcast Transcribe v{VERSION}')
    
    args = parser.parse_args()
    
    logging.info(f"WHYcast Transcribe v{VERSION} starting up")
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.all_mp3s:
        process_all_mp3s(args.input, model_size=args.model, 
                     output_dir=args.output_dir, skip_summary=args.skip_summary,
                     force=args.force)
    elif args.batch:
        process_batch(args.input, model_size=args.model, 
                     output_dir=args.output_dir, skip_summary=args.skip_summary,
                     force=args.force)
    else:
        if not os.path.isfile(args.input):
            logging.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
            
        main(args.input, model_size=args.model, 
             output_dir=args.output_dir, skip_summary=args.skip_summary,
             force=args.force)
