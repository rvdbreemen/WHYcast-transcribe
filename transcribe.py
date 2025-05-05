#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WHYcast Transcribe - v0.2.0

A tool for transcribing podcast episodes with optional speaker diarization,
summarization, and blog post generation.
"""

import os
import sys
import re
import glob
import json
import logging
import argparse
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import warnings
import requests
import urllib.parse
import markdown  # New import for markdown to HTML conversion
import torch
import torchaudio
import time
import hashlib

# Onderdruk specifieke waarschuwingen
warnings.filterwarnings("ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio")
warnings.filterwarnings("ignore", message="The bits_per_sample of .mp3 is set to 0 by default")
warnings.filterwarnings("ignore", message="PySoundFile failed")

# Schakel TensorFloat-32 in voor betere prestaties op NVIDIA Ampere GPU's
if torch.cuda.is_available():
    # Controleer of we een Ampere of nieuwere GPU hebben
    compute_capability = torch.cuda.get_device_capability(0)
    if compute_capability[0] >= 8:  # Ampere heeft compute capability 8.0+
        logging.info("TensorFloat-32 inschakelen voor betere prestaties op Ampere GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, environment variables must be set manually")

# Import required optional modules
try:
    from faster_whisper import WhisperModel
except ImportError:
    logging.error("faster-whisper not installed. Use pip install faster-whisper")
    sys.exit(1)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    from openai import BadRequestError
except ImportError:
    logging.warning("openai not installed. Summarization and blog generation will not be available.")
    OPENAI_AVAILABLE = False
    class BadRequestError(Exception):
        pass

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    logging.warning("feedparser not installed. RSS feed parsing will not be available.")
    FEEDPARSER_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    logging.warning("tqdm not installed. Progress bars will not be available.")
    TQDM_AVAILABLE = False
    
    # Simple fallback for tqdm if not available
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = len(iterable) if iterable is not None else 0
            self.n = 0
            self.desc = kwargs.get('desc', '')
            
        def __iter__(self):
            for obj in self.iterable:
                yield obj
                self.n += 1
                if self.n % 10 == 0:
                    print(f"{self.desc}: {self.n}/{self.total}")
                    
        def update(self, n=1):
            self.n += n
            
        def close(self):
            pass

# Try to import configuration
try:
    from config import (
        VERSION, MODEL_SIZE, DEVICE, COMPUTE_TYPE, BEAM_SIZE,
        OPENAI_MODEL, OPENAI_LARGE_CONTEXT_MODEL, OPENAI_HISTORY_MODEL,
        TEMPERATURE, MAX_TOKENS, MAX_INPUT_TOKENS, CHARS_PER_TOKEN,
        PROMPT_CLEANUP_FILE, PROMPT_SUMMARY_FILE, PROMPT_BLOG_FILE, PROMPT_BLOG_ALT1_FILE, PROMPT_HISTORY_EXTRACT_FILE, PROMPT_SPEAKER_ASSIGN_FILE, MAX_FILE_SIZE_KB,
        USE_RECURSIVE_SUMMARIZATION, MAX_CHUNK_SIZE, CHUNK_OVERLAP,
        VOCABULARY_FILE, USE_CUSTOM_VOCABULARY,
        USE_SPEAKER_DIARIZATION, DIARIZATION_MODEL, DIARIZATION_ALTERNATIVE_MODEL, 
        DIARIZATION_MIN_SPEAKERS, DIARIZATION_MAX_SPEAKERS,
        OPENAI_SPEAKER_MODEL
    )
except ImportError as e:
    sys.stderr.write(f"Error: Could not import configuration - {str(e)}\n")
    sys.stderr.write("Make sure config.py exists and contains all required settings.\n")
    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"Error in configuration: {str(e)}\n")
    sys.exit(1)

# Set up logging to both console and file
def setup_logging():
    # Enhanced log format with line numbers and function names
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Standaard niveau op INFO
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler with UTF-8 encoding
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transcribe.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.setLevel(logging.INFO)  # Zet file handler op INFO niveau
    logger.addHandler(file_handler)
    
    # Console handler with proper Unicode handling
    try:
        # Configure console for UTF-8
        if sys.platform == 'win32':
            import locale
            # Use error handler that replaces problematic characters
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            console_handler.setLevel(logging.INFO)  # Zet console handler op INFO niveau
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
            console_handler.setLevel(logging.INFO)  # Zet custom handler op INFO niveau
        else:
            # On non-Windows platforms, standard handler usually works fine
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            console_handler.setLevel(logging.INFO)  # Zet console handler op INFO niveau
            
        logger.addHandler(console_handler)
    except Exception as e:
        # Fallback to basic handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.setLevel(logging.INFO)  # Zet fallback handler op INFO niveau
        logger.addHandler(console_handler)
        logger.warning(f"Could not set up optimal console logging: {e}")
    
    # Zet specifieke loggers die verbose kunnen zijn op WARNING niveau
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Suppress PyTorch Inductor messages
    logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)
    logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)
    logging.getLogger("torch._subclasses").setLevel(logging.CRITICAL)
    
    # Additional PyTorch logging suppressions
    logging.getLogger("torch._inductor.remote_cache").setLevel(logging.CRITICAL)
    logging.getLogger("torch._dynamo.eval_frame").setLevel(logging.CRITICAL)
    logging.getLogger("torch._dynamo.utils").setLevel(logging.CRITICAL)
    
    # Suppress PyAnnote Audio warnings
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    
    # Filter specific warnings types
    import warnings
    warnings.filterwarnings("ignore", message="TensorFloat-32 .* has been disabled")
    warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
    warnings.filterwarnings("ignore", message=".*Cache Metrics.*")
    
    # Disable torch inductor and dynamo INFO messages
    os.environ["TORCH_LOGS"] = "ERROR"
    os.environ["TORCH_INDUCTOR_VERBOSE"] = "0"
    
    # Suppress PyTorch Inductor compile_threads warnings
    import logging as py_logging
    class PyTorchFilter(py_logging.Filter):
        def filter(self, record):
            # Filter out the compile_threads messages
            return not (hasattr(record, 'msg') and 
                       isinstance(record.msg, str) and 
                       "compile_threads set to" in record.msg)
                       
    # Apply the filter to all loggers
    root_logger = py_logging.getLogger()
    root_logger.addFilter(PyTorchFilter())
    
    return logger

logger = setup_logging()

# ==================== HUGGINGFACE API FUNCTIONS ====================
def set_huggingface_token(token: str) -> bool:
    """
    Set the HuggingFace token in the .env file.
    
    Args:
        token: The HuggingFace API token
        
    Returns:
        True if token was set successfully, False otherwise
    """
    try:
        env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        
        # Read current .env file content
        content = ""
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Check if HUGGINGFACE_TOKEN already exists in the file
        if 'HUGGINGFACE_TOKEN=' in content:
            # Replace existing token
            pattern = r'HUGGINGFACE_TOKEN=.*'
            replacement = f'HUGGINGFACE_TOKEN={token}'
            content = re.sub(pattern, replacement, content)
        else:
            # Add new token at the end
            if content and not content.endswith('\n'):
                content += '\n'
            content += f'HUGGINGFACE_TOKEN={token}\n'
        
        # Write back to .env file
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Set in current environment
        os.environ['HUGGINGFACE_TOKEN'] = token
        
        logging.info("HuggingFace token has been set successfully")
        return True
    except Exception as e:
        logging.error(f"Error setting HuggingFace token: {str(e)}")
        return False

def get_huggingface_token(ask_if_missing: bool = True) -> Optional[str]:
    """
    Get the HuggingFace token from environment. Optionally prompt the user if it's missing.
    
    Args:
        ask_if_missing: Whether to prompt the user for the token if it's missing
        
    Returns:
        The token or None if not available
    """
    token = os.environ.get('HUGGINGFACE_TOKEN')
    
    if not token and ask_if_missing:
        try:
            logging.info("HuggingFace token is required for speaker diarization")
            logging.info("You can get one from https://huggingface.co/settings/tokens")
            print("\nPlease enter your HuggingFace token (will be saved to .env file):")
            token = input().strip()
            
            if token:
                set_huggingface_token(token)
            else:
                logging.warning("No token provided. Speaker diarization will be disabled.")
                return None
        except Exception as e:
            logging.error(f"Error getting token from user: {str(e)}")
            return None
            
    return token

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
        if (paragraph_break != -1 and paragraph_break > start + max_chunk_size // 2):
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

def summarize_large_transcript(transcript: str, prompt: str) -> Optional[str]:
    """
    Handle very large transcripts by chunking and recursive summarization.
    
    Args:
        transcript: The transcript text
        prompt: The summarization prompt
        
    Returns:
        Generated summary or None if failed
    """
    estimated_tokens = estimate_token_count(transcript)
    logging.info(f"Starting recursive summarization for large transcript (~{estimated_tokens} tokens)")
    
    # Split into chunks
    chunks = split_into_chunks(transcript)
    logging.info(f"Split transcript into {len(chunks)} chunks")
    
    # Process each chunk using process_with_openai
    intermediate_summaries = []
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        try:
            chunk_prompt = "This is part of a longer transcript. Please summarize just this section, focusing on key points."
            summary = process_with_openai(chunk, chunk_prompt, OPENAI_MODEL, max_tokens=MAX_TOKENS // 2)
            
            if summary:
                intermediate_summaries.append(summary)
                logging.info(f"Completed summary for chunk {i+1}")
            else:
                logging.warning(f"Failed to summarize chunk {i+1}")
        except Exception as e:
            logging.error(f"Error summarizing chunk {i+1}: {str(e)}")
            # Continue with partial results if available
    
    # If we have intermediate summaries, combine them
    if not intermediate_summaries:
        return None
    
    # Combine intermediate summaries into a final summary
    combined_text = "\n\n".join(intermediate_summaries)
    
    # Use process_with_openai for final combination
    final_prompt = f"{prompt}\n\nHere are summaries from different parts of the transcript. Please combine them into a cohesive summary and blog post:"
    return process_with_openai(combined_text, final_prompt, OPENAI_LARGE_CONTEXT_MODEL, max_tokens=MAX_TOKENS)

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
        
        # Determine if we're using an o-series model (like o3-mini) that requires special parameter handling
        # NOTE: GPT-4o is NOT considered an o-series model in this context - it uses standard parameters
        is_o_series_model = model_name.startswith("o") and not model_name.startswith("gpt")
        
        # Set up parameters based on model type
        token_param = "max_completion_tokens" if is_o_series_model else "max_tokens"
        
        # Create base parameters dict
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
                {"role": "user", "content": f"{prompt}\n\nHere's the text to process:\n\n{text}"}
            ]
        }
        
        # Add temperature only for models that support it (non-"o" series)
        if not is_o_series_model:
            params["temperature"] = TEMPERATURE
        
        try:
            # For transcript cleanup, we need to ensure we get complete output by using a higher max_tokens limit
            if "clean" in prompt.lower() and "transcript" in prompt.lower():
                # For cleanup, give more tokens for output - use at least text length or double MAX_TOKENS
                cleanup_max_tokens = max(estimate_token_count(text), MAX_TOKENS * 2)
                logging.info(f"Using expanded token limit for cleanup: {cleanup_max_tokens}")
                
                # Check if text needs to be processed in chunks due to size
                if estimated_tokens > MAX_INPUT_TOKENS // 2:
                    return process_large_text_in_chunks(text, prompt, model_name, client)
                
                # Add token parameter
                params[token_param] = cleanup_max_tokens
                
                response = client.chat.completions.create(**params)
            else:
                # Regular processing for non-cleanup tasks
                # Add token parameter
                params[token_param] = max_tokens
                
                response = client.chat.completions.create(**params)
                
            result = response.choices[0].message.content
            
            # Check if result might be truncated (ends abruptly without proper punctuation)
            if len(result) > 100 and not result.rstrip().endswith(('.', '!', '?', '"', ':', ';', ')', ']', '}')):
                logging.warning("Generated text may be truncated (doesn't end with punctuation)")
                
            return result
                
        except BadRequestError as e:
            if "maximum context length" in str(e).lower():
                # Try with more aggressive truncation
                logging.warning(f"Context length exceeded. Retrying with further truncation...")
                text = truncate_transcript(text, MAX_INPUT_TOKENS // 2)
                logging.info(f"Retrying with reduced text (~{estimate_token_count(text)} tokens)...")
                
                # Update the message content with truncated text
                params["messages"][1]["content"] = f"{prompt}\n\nHere's the text to process:\n\n{text}"
                
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content
            else:
                raise
    except Exception as e:
        logging.error(f"Error processing with OpenAI: {str(e)}")
        return None

def process_large_text_in_chunks(text: str, prompt: str, model_name: str, client: OpenAI) -> str:
    """
    Process very large text by breaking it into chunks and reassembling the results.
    
    Args:
        text: The text to process
        prompt: The processing instructions
        model_name: Model to use
        client: OpenAI client
        
    Returns:
        Combined processed text
    """
    logging.info("Text is very large, processing in chunks")
    chunks = split_into_chunks(text, max_chunk_size=MAX_CHUNK_SIZE*2)
    logging.info(f"Split text into {len(chunks)} chunks")
    
    # Determine if we're using an o-series model (like o3-mini) that requires special parameter handling
    # NOTE: GPT-4o is NOT considered an o-series model in this context - it uses standard parameters
    is_o_series_model = model_name.startswith("o") and not model_name.startswith("gpt")
    
    # Set up parameters based on model type
    token_param = "max_completion_tokens" if is_o_series_model else "max_tokens"
    token_limit = MAX_TOKENS * 2  # Use larger output limit for chunks
    
    modified_prompt = f"{prompt}\n\nThis is a chunk of a longer transcript. Process this chunk following the instructions."
    processed_chunks = []
    
    # Add progress bar for chunk processing
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        logging.info(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            # Create parameters dictionary with proper token limit parameter
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that processes transcript chunks."},
                    {"role": "user", "content": f"{modified_prompt}\n\nChunk {i+1}/{len(chunks)}:\n\n{chunk}"}
                ]
            }
            
            # Add temperature only for models that support it (non-"o" series)
            if not is_o_series_model:
                params["temperature"] = TEMPERATURE
                
            # Add token parameter
            params[token_param] = token_limit
            
            response = client.chat.completions.create(**params)
            processed_chunks.append(response.choices[0].message.content)
            logging.info(f"Successfully processed chunk {i+1}")
        except Exception as e:
            logging.error(f"Error processing chunk {i+1}: {str(e)}")
            # If processing fails, include original chunk to avoid data loss
            processed_chunks.append(chunk)
    
    # Combine processed chunks
    combined_text = "\n\n".join(processed_chunks)
    
    # Optionally, run a final pass to ensure consistency across chunk boundaries
    if len(processed_chunks) > 1:
        try:
            logging.info("Running final pass to ensure consistency across chunk boundaries")
            # Estimate token count of combined text
            combined_tokens = estimate_token_count(combined_text)
            
            # If combined text is still too large, just return it as is
            if combined_tokens > MAX_INPUT_TOKENS:
                logging.warning(f"Combined text is too large for final pass (~{combined_tokens} tokens)")
                return combined_text
            
            # Create parameters for the consistency pass
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that ensures transcript consistency."},
                    {"role": "user", "content": f"This is a processed transcript that was handled in chunks. Please ensure consistency across chunk boundaries and fix any obvious issues.\n\n{combined_text}"}
                ]
            }
            
            # Add temperature only for models that support it (non-"o" series)
            if not is_o_series_model:
                params["temperature"] = TEMPERATURE
                
            # Add token parameter
            params[token_param] = MAX_TOKENS
            
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in final consistency pass: {str(e)}")
            return combined_text
    
    return combined_text

def process_transcript_workflow(transcript: str) -> Dict[str, Optional[str]]:
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
    results['speaker_assignment'] = speaker_assignment_step(transcript)
    cleaned = cleanup_step(transcript)
    results['cleaned_transcript'] = cleaned
    results['summary'] = summary_step(cleaned)
    results['blog'] = blog_step(cleaned, results['summary'])
    results['blog_alt1'] = alt_blog_step(cleaned, results['summary'])
    results['history_extract'] = history_step(cleaned)
    return results

def speaker_assignment_step(transcript: str) -> Optional[str]:
    """
    Step 1: Run speaker assignment on the raw transcript.
    Checks for diarization tags and, if present, uses the speaker assignment workflow
    to generate a transcript with speaker names. Returns the speaker-assigned transcript as a string,
    or None if not applicable or failed.
    """
    logging.info("Step 1: Speaker assignment")
    try:
        if any("SPEAKER_" in line for line in transcript.splitlines()):
            base_path = os.environ.get("WORKFLOW_OUTPUT_BASE", "output")
            txt_path = process_speaker_assignment_workflow(transcript, base_path)
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        logging.error(f"Speaker assignment failed: {e}")
    return None

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

def summary_step(cleaned: str) -> Optional[str]:
    """
    Step 3: Generate summary from the cleaned transcript.
    Uses a prompt to summarize the cleaned transcript. Returns the summary as a string, or None if failed.
    """
    prompt = read_prompt_file(PROMPT_SUMMARY_FILE)
    if not prompt:
        logging.warning("Summary prompt missing, skipping summary")
        return None
    logging.info("Step 3: Generating summary...")
    return process_with_openai(cleaned, prompt, choose_appropriate_model(cleaned))

def blog_step(cleaned: str, summary: Optional[str]) -> Optional[str]:
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
    return process_with_openai(input_text, prompt, choose_appropriate_model(input_text), max_tokens=MAX_TOKENS * 2)

def alt_blog_step(cleaned: str, summary: Optional[str]) -> Optional[str]:
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
    return process_with_openai(input_text, prompt, choose_appropriate_model(input_text), max_tokens=MAX_TOKENS * 2)

def history_step(cleaned: str) -> Optional[str]:
    """
    Step 6: Generate history extraction from the cleaned transcript.
    Uses a prompt to extract historical lessons or context from the transcript. Returns the history extraction as a string, or None if failed.
    """
    prompt = read_prompt_file(PROMPT_HISTORY_EXTRACT_FILE)
    if not prompt:
        logging.warning("History prompt missing, skipping history extraction")
        return None
    logging.info("Step 6: Generating history extraction...")
    return process_with_openai(cleaned, prompt, OPENAI_HISTORY_MODEL, max_tokens=MAX_TOKENS * 2)

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
    
    # Write history extraction
    if 'history_extract' in results and results['history_extract']:
        base_filename = os.path.basename(output_base)
        output_dir = os.path.dirname(output_base)
        history_path = os.path.join(output_dir, f"{base_filename}_history.txt")
        with open(history_path, "w", encoding="utf-8") as f:
            f.write(results['history_extract'])
        logging.info(f"History extraction saved to: {history_path}")
        
        # Generate and write HTML version
        html_content = convert_markdown_to_html(results['history_extract'])
        html_path = os.path.join(output_dir, f"{base_filename}_history.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML history extraction saved to: {html_path}")
        
        # Generate and write Wiki version
        wiki_content = convert_markdown_to_wiki(results['history_extract'])
        wiki_path = os.path.join(output_dir, f"{base_filename}_history.wiki")
        with open(wiki_path, "w", encoding="utf-8") as f:
            f.write(wiki_content)
        logging.info(f"Wiki history extraction saved to: {wiki_path}")
    
    # Step 6: Speaker assignment
    speaker_prompt = read_prompt_file(PROMPT_SPEAKER_ASSIGN_FILE)
    if USE_SPEAKER_DIARIZATION and speaker_prompt:
        logging.info("Step 6: Assigning speaker names...")
        try:
            ts_path = f"{output_base}_ts.txt"
            with open(ts_path, 'r', encoding='utf-8') as f:
                ts_text = f.read()
            assignment = process_with_openai(ts_text, speaker_prompt, OPENAI_SPEAKER_MODEL)
            if assignment:
                assign_path = f"{output_base}_speaker_assignment.txt"
                with open(assign_path, 'w', encoding='utf-8') as af:
                    af.write(assignment)
                logging.info(f"Speaker assignment saved to: {assign_path}")
                results['speaker_assignment'] = assignment
                # Generate HTML version of speaker assignment
                html_content = convert_markdown_to_html(assignment)
                html_path = f"{output_base}_speaker_assignment.html"
                with open(html_path, "w", encoding="utf-8") as hf:
                    hf.write(html_content)
                logging.info(f"HTML speaker assignment saved to: {html_path}")
                # Generate Wiki version of speaker assignment
                wiki_content = convert_markdown_to_wiki(assignment)
                wiki_path = f"{output_base}_speaker_assignment.wiki"
                with open(wiki_path, "w", encoding="utf-8") as wf:
                    wf.write(wiki_content)
                logging.info(f"Wiki speaker assignment saved to: {wiki_path}")
            else:
                results['speaker_assignment'] = None
                logging.warning("Speaker assignment returned no result.")
        except Exception as e:
            results['speaker_assignment'] = None
            logging.error(f"Error during speaker assignment: {str(e)}")

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
        estimated_tokens = estimate_token_count(transcript)
        logging.info(f"Estimated transcript length: ~{estimated_tokens} tokens")
        
        # For very large transcripts, use recursive summarization
        if USE_RECURSIVE_SUMMARIZATION and estimated_tokens > MAX_INPUT_TOKENS:
            logging.info(f"Transcript is very large ({estimated_tokens} tokens), using recursive summarization")
            return summarize_large_transcript(transcript, prompt)
        
        # Choose appropriate model based on length
        model_to_use = choose_appropriate_model(transcript)
        
        # Use process_with_openai instead of direct API call for consistent handling
        logging.info(f"Generating summary and blog using {model_to_use}...")
        return process_with_openai(transcript, prompt, model_to_use, max_tokens=MAX_TOKENS)
    
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
    try:
        import torch
        
        # Log system information
        logging.info("=== System Information ===")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            logging.info(f"GPU device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logging.info(f"GPU {i}: {props.name}")
                logging.info(f"  Total memory: {props.total_memory / (1024**3):.2f} GB")
                logging.info(f"  CUDA capability: {props.major}.{props.minor}")
                logging.info(f"  Multi-processor count: {props.multi_processor_count}")
            
            # Optimize CUDA settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set number of workers based on GPU memory
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
            if gpu_mem >= 16:  # High-end GPU
                num_workers = 4
            elif gpu_mem >= 8:  # Mid-range GPU
                num_workers = 2
            else:  # Lower-end GPU
                num_workers = 1
                
            logging.info(f"Using {num_workers} workers for data loading")
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            num_workers = 8  # More workers for CPU
            compute_type = "int8"
            logging.warning("CUDA is not available - using CPU which may be significantly slower")
        
        # Create model with optimized parameters
        model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            cpu_threads=8 if device == "cpu" else 0,
            num_workers=num_workers
        )
        
        logging.info(f"Initialized {model_size} model on {device} with compute type {compute_type}")
        return model
        
    except Exception as e:
        logging.error(f"Error setting up model: {str(e)}")
        # Fallback to CPU if CUDA setup fails
        logging.warning("Falling back to CPU model")
        return WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=8,
            num_workers=4
        )

def transcribe_audio(model: WhisperModel, audio_file: str, speaker_segments: Optional[List[Dict]] = None) -> Tuple[List, object]:
    """
    Transcribe audio file using the Whisper model.
    
    Args:
        model: Initialized WhisperModel instance
        audio_file: Path to the audio file to transcribe
        speaker_segments: Optional list of speaker segments from diarization
        
    Returns:
        Tuple of (segments, info)
    """
    import os  # Actually importing the os module
    
    logging.info(f"Transcribing audio file: {audio_file}")
    print(f"\nStart transcriptie van {os.path.basename(audio_file)}...")
    start_time = time.time()
    
    # Load custom vocabulary if enabled
    word_list = None
    word_replacements = {}
    if USE_CUSTOM_VOCABULARY and os.path.exists(VOCABULARY_FILE):
        try:
            with open(VOCABULARY_FILE, 'r', encoding='utf-8') as f:
                vocabulary = json.load(f)
            
            # Maak een lijst van woorden voor word_list parameter
            word_list = []
            
            # Lees vervangingen voor post-processing
            for original, replacement in vocabulary.items():
                word_replacements[original.lower()] = replacement
                # Voeg ook de vervanging toe aan de word_list
                word_list.append(replacement)
            
            if word_list:
                logging.info(f"Loaded {len(word_list)} custom vocabulary words")
                print(f"Aangepaste woordenlijst geladen met {len(word_list)} woorden")
        except Exception as e:
            logging.error(f"Error loading vocabulary: {str(e)}")
    
    # Maak basic transcriptie parameters
    transcription_params = {
        'beam_size': BEAM_SIZE,
        'word_timestamps': True,  # Enable word timestamps for better alignment with diarization
        'vad_filter': True,       # Filter out non-speech parts
        'vad_parameters': dict(min_silence_duration_ms=500),  # Configure VAD for better accuracy
        'initial_prompt': "This is a podcast transcription.",
        'condition_on_previous_text': True,
    }
    
    # Bijhouden van huidige spreker om herhalingen te vermijden
    current_speaker = None
    
    # Definieer functie voor live output
    def process_segment(segment):
        text = segment.text
        
        # Pas woordenboek toe op elke segment
        if word_replacements:
            for original, replacement in word_replacements.items():
                # Vervang hele woorden met case-insensitive match
                pattern = r'\b' + re.escape(original) + r'\b'
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Voeg sprekerinfo toe als beschikbaar
        speaker_info = ""
        nonlocal current_speaker
        
        if speaker_segments:
            # Gebruik middelpunt van het segment om spreker te bepalen
            segment_middle = (segment.start + segment.end) / 2
            speaker = get_speaker_for_segment(segment_middle, speaker_segments)
            
            # Altijd speaker tonen, niet alleen bij wisseling
            if speaker:
                speaker_info = f"[{speaker}] "
                current_speaker = speaker
            else:
                speaker_info = "[SPEAKER_UNKNOWN] "
        
        # Toon de tekst direct in de console
        print(f"{format_timestamp(segment.start, segment.end)} {speaker_info}{text}")
        
        # Pas de tekst toe in het segment
        segment.text = text
        return segment
    
    # Function to collect segments with real-time processing
    def collect_with_live_output(segments_generator):
        print("\n--- Live Transcriptie Output ---")
        result = []
        for segment in segments_generator:
            # Process each segment for replacements and logging
            segment = process_segment(segment)
            result.append(segment)
        print("--- Einde Live Transcriptie ---\n")
        return result
    
    # Execute transcription with appropriate parameters
    try:
        # Voeg word_list alleen toe als deze is gedefinieerd en niet leeg is
        if word_list:
            try:
                # Probeer met word_list
                print("Transcriptie uitvoeren met aangepaste woordenlijst...")
                segments_generator, info = model.transcribe(
                    audio_file,
                    **transcription_params,
                    word_list=word_list
                )
                segments_list = collect_with_live_output(segments_generator)
            except TypeError as e:
                # Als word_list niet wordt ondersteund, probeer zonder
                logging.warning(f"word_list parameter niet ondersteund in deze versie van faster_whisper: {e}")
                logging.info("Transcriptie uitvoeren zonder aangepaste woordenlijst")
                print("word_list niet ondersteund, transcriptie uitvoeren zonder aangepaste woordenlijst...")
                segments_generator, info = model.transcribe(
                    audio_file, 
                    **transcription_params
                )
                segments_list = collect_with_live_output(segments_generator)
        else:
            # Als er geen word_list is, gebruik de standaard parameters
            print("Transcriptie uitvoeren...")
            segments_generator, info = model.transcribe(
                audio_file,
                **transcription_params
            )
            segments_list = collect_with_live_output(segments_generator)
            
    except Exception as e:
        logging.error(f"Error tijdens transcriptie: {str(e)}")
        raise
    
    # Bereken en toon verwerkingstijd
    elapsed_time = time.time() - start_time
    print(f"\nTranscriptie voltooid in {elapsed_time:.1f} seconden.")
    print(f"Taal gedetecteerd: {info.language} (waarschijnlijkheid: {info.language_probability:.2f})")
    print(f"Aantal segmenten: {len(segments_list)}")
    
    logging.info(f"Transcription complete: {len(segments_list)} segments")
    return segments_list, info

def write_transcript_files(segments: List, output_file: str, output_file_timestamped: str, 
                          speaker_segments: Optional[List[Dict]] = None) -> str:
    """
    Write transcript files with and without timestamps, integrating speaker info.
    
    Args:
        segments: List of transcription segments from Whisper
        output_file: Path to save the clean transcript
        output_file_timestamped: Path to save the timestamped transcript
        speaker_segments: Optional list of speaker segments from diarization
        
    Returns:
        Full transcript text
    """
    try:
        # Prepare for writing the transcripts
        full_transcript = []
        timestamped_transcript = []
        
        # Track current speaker to avoid repeating speaker tags for consecutive segments
        current_speaker = None
        
        print(f"\nTranscriptie bestanden aanmaken...")
        print(f"- Schoon transcript: {os.path.basename(output_file)}")
        print(f"- Tijdgemarkeerd transcript: {os.path.basename(output_file_timestamped)}")
        
        # Process each segment from Whisper
        for i, segment in enumerate(tqdm(segments, desc="Transcriptie verwerken", unit="segment")):
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            
            if not text:  # Skip empty segments
                continue
            
            # Calculate segment duration for better speaker detection of short segments
            segment_duration = end - start
            
            # Format timestamp for the timestamped version
            timestamp = format_timestamp(start, end)
            
            # Get speaker info if available
            speaker_info = ""
            speaker_prefix = ""
            if speaker_segments:
                # Use middle of segment to determine speaker with improved context-awareness
                middle_time = (start + end) / 2
                # Pass segment duration to the improved function
                speaker = get_speaker_for_segment(middle_time, speaker_segments, segment_duration=segment_duration)
                
                if speaker:
                    # Check if this is a short utterance that should inherit speaker from context
                    is_short_utterance = segment_duration < 1.0 and len(text.split()) <= 5
                    
                    # For very short utterances with no clear speaker, try to maintain speaker continuity
                    if is_short_utterance and not speaker and current_speaker:
                        # Check if there's another segment right after this one
                        if i < len(segments) - 1:
                            next_segment = segments[i + 1]
                            if next_segment.start - end < 1.0:  # If next segment starts within 1 second
                                next_middle = (next_segment.start + next_segment.end) / 2
                                next_speaker = get_speaker_for_segment(next_middle, speaker_segments)
                                # If next segment has same speaker as current, keep it for continuity
                                if next_speaker == current_speaker:
                                    speaker = current_speaker
                    
                    if speaker:
                        speaker_info = f"[{speaker}] "
                        speaker_prefix = f"[{speaker}] "
                        current_speaker = speaker
                    else:
                        # More aggressive search for a speaker with a wider context window
                        speaker = get_speaker_for_segment(middle_time, speaker_segments, 
                                                         segment_duration=segment_duration, 
                                                         context_window=2.0)
                        if speaker:
                            speaker_info = f"[{speaker}] "
                            speaker_prefix = f"[{speaker}] "
                            current_speaker = speaker
                        else:
                            # Only mark as unknown if we are sure it's not part of previous speaker's utterance
                            if current_speaker and segment_duration < 1.5 and start - prev_end < 1.0:
                                # This is likely a continuation from previous speaker
                                speaker_info = f"[{current_speaker}] "
                                speaker_prefix = f"[{current_speaker}] "
                            else:
                                speaker_info = "[SPEAKER_UNKNOWN] "
                                speaker_prefix = "[SPEAKER_UNKNOWN] "
                                current_speaker = None
                else:
                    speaker_info = "[SPEAKER_UNKNOWN] "
                    speaker_prefix = "[SPEAKER_UNKNOWN] "
                    current_speaker = None
            
            # Add to transcript collections
            clean_line = f"{speaker_prefix}{text}"
            timestamped_line = f"{timestamp} {speaker_info}{text}"
            
            full_transcript.append(clean_line)
            timestamped_transcript.append(timestamped_line)
            
            # Store the end time for checking continuity in the next iteration
            prev_end = end
        
        # Join all lines
        full_text = "\n".join(full_transcript)
        timestamped_text = "\n".join(timestamped_transcript)
        
        # Write to files
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        with open(output_file_timestamped, 'w', encoding='utf-8') as f:
            f.write(timestamped_text)
            
        print(f"Transcriptie bestanden succesvol aangemaakt:")
        print(f"- {output_file}")
        print(f"- {output_file_timestamped}")
        
        return full_text
    except Exception as e:
        logging.error(f"Error writing transcript files: {str(e)}")
        return ""

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
        
        # Create base path for output files (without any extension)
        base = os.path.splitext(transcript_file)[0]
        
        # Run the complete workflow to generate all outputs
        results = process_transcript_workflow(transcript)
        if not results:
            logging.error("Failed to process transcript workflow")
            return False
        
        # Write all outputs to files
        write_workflow_outputs(results, base)
        
        return True
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
        
        # Use process_with_openai instead of direct API call for consistent handling
        blog = process_with_openai(
            input_text, 
            blog_prompt, 
            model_to_use, 
            max_tokens=MAX_TOKENS * 2
        )
        
        if not blog:
            logging.error("Failed to generate blog post")
            return False
            
        # Save the blog post to the correct path
        with open(blog_file, 'w', encoding='utf-8') as f:
            f.write(blog)
            logging.info(f"Blog post saved to: {blog_file}")
        
        # Only generate alternative blog post if the alternative prompt file exists
        if os.path.isfile(PROMPT_BLOG_ALT1_FILE):
            # Creating the alternative blogpost
            logging.info("Generating blog alt 1 post...")
            
            # Read the blog prompt - only attempt to read if file exists
            blog_alt1_prompt = read_prompt_file(PROMPT_BLOG_ALT1_FILE)
            if blog_alt1_prompt:
                # Use process_with_openai instead of direct API call
                blog_alt1 = process_with_openai(
                    input_text, 
                    blog_alt1_prompt, 
                    model_to_use, 
                    max_tokens=MAX_TOKENS * 2
                )
                
                if blog_alt1:
                    # Save the blog post to the correct path
                    with open(blog_alt1_file, 'w', encoding='utf-8') as f:
                        f.write(blog_alt1)
                        logging.info(f"Blog alt1 post saved to: {blog_alt1_file}")
                        
                    # Generate and write HTML version
                    html_content = convert_markdown_to_html(blog_alt1)
                    html_path = os.path.join(output_dir, f"{base_filename}_blog_alt1.html")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    logging.info(f"HTML blog alt1 post saved to: {html_path}")
                    
                    # Generate and write Wiki version
                    wiki_content = convert_markdown_to_wiki(blog_alt1)
                    wiki_path = os.path.join(output_dir, f"{base_filename}_blog_alt1.wiki")
                    with open(wiki_path, "w", encoding="utf-8") as f:
                        f.write(wiki_content)
                    logging.info(f"Wiki blog alt1 post saved to: {wiki_path}")
                else:
                    logging.error("Failed to generate alternative blog post")
            else:
                logging.warning("Blog prompt alt1 file exists but is empty or couldn't be read, skipping blog alt1 generation")
        else:
            # File doesn't exist, just log at INFO level and skip
            logging.info(f"Blog prompt alt1 file {PROMPT_BLOG_ALT1_FILE} not found, skipping alternative blog generation")
        
        # Generate and write HTML version
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

def regenerate_cleaned_transcript(transcript_file: str) -> bool:
    """
    Regenerate cleaned version of a transcript file.
    
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
        
        # Create path for output file
        base = os.path.splitext(transcript_file)[0]
        cleaned_file = f"{base}_cleaned.txt"
        
        # Read the cleanup prompt
        cleanup_prompt = read_prompt_file(PROMPT_CLEANUP_FILE)
        if not cleanup_prompt:
            logging.warning("Cleanup prompt file not found or empty, cannot regenerate cleaned transcript")
            return False

        logging.info("Generating cleaned transcript...")
        
        # Choose appropriate model based on transcript length
        model_to_use = choose_appropriate_model(transcript)
        
        # Use process_with_openai for consistent handling
        cleaned_transcript = process_with_openai(
            transcript, 
            cleanup_prompt, 
            model_to_use, 
            max_tokens=MAX_TOKENS * 2
        )
        
        if not cleaned_transcript:
            logging.error("Failed to generate cleaned transcript")
            return False
            
        # Save the cleaned transcript
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_transcript)
            logging.info(f"Cleaned transcript saved to: {cleaned_file}")
        
        return True
                        
    except Exception as e:
        logging.error(f"Error regenerating cleaned transcript: {str(e)}")
        return False

def regenerate_all_cleaned(directory: str) -> None:
    """
    Regenerate cleaned transcript for all transcript files in the given directory.
    It processes files that do not already have a _cleaned, _ts, _summary, or _blog suffix.
    
    Args:
        directory: Directory containing transcript files
    """
    import os
    import glob
    import logging

    if not os.path.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return

    transcript_files = glob.glob(os.path.join(directory, "*.txt"))
    # Filter out files that are already processed as cleaned or have other suffixes
    original_files = [f for f in transcript_files if not any(suffix in f for suffix in ["_cleaned.txt", "_ts.txt", "_summary.txt", "_blog.txt"])]
    
    if not original_files:
        logging.warning(f"No original transcript files found in directory: {directory}")
        return

    logging.info(f"Found {len(original_files)} transcript files for cleaning regeneration")
    processed_count = 0
    for transcript_file in original_files:
        logging.info(f"Regenerating cleaned transcript for: {transcript_file}")
        success = regenerate_cleaned_transcript(transcript_file)
        if success:
            processed_count += 1

    logging.info(f"Regenerated cleaned transcripts for {processed_count} out of {len(original_files)} files")

def regenerate_full_workflow(input_file: str) -> None:
    """
    Regenerate workflow outputs for an existing transcript file.
    
    Args:
        input_file: Path to the transcript file
    """
    # Normalize path by removing any trailing slashes
    input_file = input_file.rstrip(os.path.sep)
    
    # Verify this is a file, not a directory
    if os.path.isdir(input_file):
        logging.error(f"Input must be a file, not a directory: {input_file}")
        return
        
    # Continue with existing logic
    base_path, _ = os.path.splitext(input_file)
    ts_file = f"{base_path}_ts.txt"
    if not os.path.exists(ts_file):
        logging.error(f"No timestamped file found: {ts_file}, skipping re-transcription.")
        return
    transcript_file = f"{base_path}.txt"
    if not os.path.exists(transcript_file):
        logging.error(f"No transcript file found: {transcript_file}, cannot proceed.")
        return
    
    # Get the base name without any suffixes for consistent output file naming
    base_name = os.path.basename(base_path)
    # Remove any known suffixes to get the pure base name
    known_suffixes = ["_cleaned", "_ts", "_summary", "_blog", "_blog_alt1"]
    for suffix in known_suffixes:
        if (base_name.endswith(suffix)):
            base_name = base_name[:-len(suffix)]
            break
    
    # Reconstruct full output base path
    output_dir = os.path.dirname(base_path)
    output_base = os.path.join(output_dir, base_name)
    
    logging.info(f"Running full workflow on existing transcript: {transcript_file}")
    with open(transcript_file, "r", encoding="utf-8") as f:
        full_transcript = f.read()
    results = process_transcript_workflow(full_transcript)
    if not results:
        logging.error("Failed to process transcript workflow.")
        return
    
    write_workflow_outputs(results, output_base)

def regenerate_all_full_workflow(directory: str) -> None:
    """
    Run the full workflow regeneration on all transcript files in the given directory.
    
    Args:
        directory: Directory containing transcript files
    """
    if not os.path.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return

    # Find all transcript files (txt files that don't have special suffixes)
    transcript_files = glob.glob(os.path.join(directory, "*.txt"))
    base_transcripts = [f for f in transcript_files 
                       if not any(suffix in f for suffix in 
                                ["_ts.txt", "_summary.txt", "_blog.txt", "_cleaned.txt", "_blog_alt1.txt"])]
    
    if not base_transcripts:
        logging.warning(f"No original transcript files found in directory: {directory}")
        return

    logging.info(f"Found {len(base_transcripts)} transcript files to process with full workflow")
    processed_count = 0
    
    for transcript_file in tqdm(base_transcripts, desc="Regenerating full workflow"):
        try:
            logging.info(f"Running full workflow on: {transcript_file}")
            regenerate_full_workflow(transcript_file)
            processed_count += 1
        except Exception as e:
            logging.error(f"Error processing {transcript_file}: {str(e)}")

    logging.info(f"Completed full workflow regeneration for {processed_count} out of {len(base_transcripts)} files")

def regenerate_blogs_from_cleaned(directory: str) -> None:
    """
    Regenerate blog posts specifically from cleaned transcripts in the given directory.
    
    Args:
        directory: Directory containing cleaned transcript files
    """
    # Create directory if it doesn't exist
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create or access directory {directory}: {str(e)}")
        return
    
    if not os.path.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return
    
    # Look specifically for cleaned transcript files
    cleaned_files = glob.glob(os.path.join(directory, "*_cleaned.txt"))
    
    # If no cleaned files, look for regular transcript files
    if not cleaned_files:
        files = glob.glob(os.path.join(directory, "*.txt"))
        transcript_files = [f for f in files 
                           if not any(suffix in f for suffix in 
                                    ["_ts.txt", "_summary.txt", "_blog.txt", "_history.txt", "_blog_alt1.txt"])]
    else:
        transcript_files = cleaned_files
    
    if not transcript_files:
        logging.warning(f"No suitable transcript files found in directory: {directory}")
        return
    
    logging.info(f"Found {len(transcript_files)} transcript files to process for history extraction")
    processed_count = 0
    skipped_count = 0
    
    for transcript_file in tqdm(transcript_files, desc="Generating history extractions"):
        logging.info(f"Processing file: {transcript_file}")
        
        # Get base filename to check if history already exists
        base_filename = os.path.splitext(os.path.basename(transcript_file))[0]
        if base_filename.endswith("_cleaned"):
            base_filename = base_filename[:-8]  # Remove "_cleaned" from the end
            
        output_dir = os.path.dirname(transcript_file)
        history_path = os.path.join(output_dir, f"{base_filename}_history.txt")
        
        # Skip if history exists
        if os.path.exists(history_path):
            logging.info(f"Skipping {transcript_file} - history extraction already exists")
            skipped_count += 1
            continue
        
        # Placeholder: history extraction function is not defined
        logging.warning(f"History extraction function not implemented. Skipping {transcript_file}.")
        
        # If implemented, increment processed_count
        # processed_count += 1
        
        # Force garbage collection to release memory
        import gc
        gc.collect()
    
    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} files (history extractions already exist)")
    
    logging.info(f"Successfully generated {processed_count} history extractions out of {len(transcript_files) - skipped_count} files processed")

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

def download_latest_episode(feed_url: str, download_dir: str, force: bool = False) -> Optional[str]:
    """
    Download the latest episode from the podcast feed if not already processed.
    If force is True, delete all related files and processed marker before downloading.
    """
    os.makedirs(download_dir, exist_ok=True)
    episode = get_latest_episode(feed_url)
    if not episode:
        logging.warning("No episode found to download")
        return None

    # If force, delete all related files and remove from processed_episodes.txt
    if force:
        filename = get_episode_filename(episode)
        base = os.path.splitext(filename)[0]
        patterns = [f"{base}*", f"{base.lower()}*", f"{base.upper()}*"]
        for pattern in patterns:
            for f in glob.glob(os.path.join(download_dir, pattern)):
                try:
                    os.remove(f)
                    logging.info(f"Deleted file: {f}")
                except Exception as e:
                    logging.warning(f"Could not delete file {f}: {e}")
        # Remove from processed_episodes.txt
        processed_file = os.path.join(download_dir, "processed_episodes.txt")
        episode_id = get_episode_id(episode)
        if os.path.exists(processed_file):
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                with open(processed_file, 'w', encoding='utf-8') as f:
                    for line in lines:
                        if line.strip() != episode_id:
                            f.write(line)
                logging.info(f"Removed episode ID {episode_id} from processed_episodes.txt")
            except Exception as e:
                logging.warning(f"Could not update processed_episodes.txt: {e}")

    # Check if already processed (after force cleanup)
    if episode_already_processed(episode, download_dir):
        logging.info(f"Episode '{episode['title']}' has already been processed, skipping")
        return None

    filename = get_episode_filename(episode)
    full_path = os.path.join(download_dir, filename)
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
    
    if (os.path.exists(processed_file)):
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

def perform_speaker_diarization(
    audio_file: str,
    min_speakers: int = DIARIZATION_MIN_SPEAKERS,
    max_speakers: int = DIARIZATION_MAX_SPEAKERS,
    num_speakers: int = None,
    huggingface_token: str = None
) -> Optional[List[Dict]]:
    """
    Speaker diarization using pyannote/speaker-diarization-3.1.
    Always converts input to 16kHz mono MP3 with ffmpeg, loads in memory, and runs diarization.
    """
    import os, subprocess, torch, torchaudio
    from pyannote.audio import Pipeline

    temp_mp3 = audio_file + ".ffmpeg.16k.mp3"
    try:
        # Step 1: Always convert to 16kHz mono MP3
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', audio_file,
            '-vn', '-acodec', 'libmp3lame', '-ar', '16000', '-ac', '1', temp_mp3
        ]
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            return None
        # Step 2: Load audio in memory
        waveform, sample_rate = torchaudio.load(temp_mp3)
        # Step 3: Setup pipeline
        token = huggingface_token or os.environ.get('HUGGINGFACE_TOKEN')
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "diarization")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
            cache_dir=cache_dir
        )
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        # Step 4: Run diarization
        diarization_kwargs = {'min_speakers': min_speakers, 'max_speakers': max_speakers}
        if num_speakers is not None:
            diarization_kwargs = {'num_speakers': num_speakers}
        diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **diarization_kwargs)
        # Step 5: Format output
        segments = [
            {'start': float(turn.start), 'end': float(turn.end), 'speaker': str(speaker)}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        return segments
    except Exception:
        return None
    finally:
        try:
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
        except Exception:
            pass

def get_speaker_for_segment(timestamp: float, speaker_segments: List[Dict], 
                         segment_duration: float = 0.0, 
                         context_window: float = 1.0) -> Optional[str]:
    """
    Find the speaker for a given timestamp from diarization results using improved context-aware approach.
    
    Args:
        timestamp: The timestamp to find the speaker for
        speaker_segments: List of speaker segments from diarization
        segment_duration: Duration of the segment being analyzed (for short segment handling)
        context_window: Size of context window to consider around the timestamp in seconds
        
    Returns:
        Speaker label or None if no match found
    """
    if not speaker_segments:
        return None
    
    # Determine if this is a short segment (less than 1 second)
    is_short_segment = segment_duration > 0 and segment_duration < 1.0
        
    # For short segments, use wider context window and more sophisticated matching
    if is_short_segment:
        # Use wider context window for short segments
        context_window = max(1.5, segment_duration * 3)
        
        # Create a list of speakers in the vicinity with confidence scores
        speaker_candidates = []
        
        # First phase: Find exact overlaps
        for segment in speaker_segments:
            # Check if timestamp falls within this segment
            if segment['start'] <= timestamp <= segment['end']:
                # Strong match - speaker directly overlaps with the timestamp
                # Calculate how centered the timestamp is in the segment (0.0-1.0 score)
                segment_length = segment['end'] - segment['start']
                if segment_length > 0:
                    position_ratio = 1.0 - 2.0 * abs((timestamp - segment['start']) / segment_length - 0.5)
                    confidence = 0.8 + (position_ratio * 0.2)  # Score between 0.8-1.0
                else:
                    confidence = 0.8
                
                speaker_candidates.append({
                    'speaker': segment['speaker'],
                    'confidence': confidence,
                    'distance': 0.0
                })
        
        # If no direct match, find nearby speakers within context window
        if not speaker_candidates:
            for segment in speaker_segments:
                # Calculate distance to this segment
                if timestamp < segment['start']:
                    distance = segment['start'] - timestamp
                elif timestamp > segment['end']:
                    distance = timestamp - segment['end']
                else:
                    distance = 0  # Inside the segment (shouldn't happen here)
                
                # Include if within context window
                if distance <= context_window:
                    # Calculate confidence based on distance - closer is better
                    confidence = max(0.1, 1.0 - (distance / context_window))
                    speaker_candidates.append({
                        'speaker': segment['speaker'],
                        'confidence': confidence,
                        'distance': distance
                    })
        
        # Second phase: If we have candidates, find most probable speaker
        if speaker_candidates:
            # Sort by confidence (highest first)
            speaker_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Return the most confident speaker
            return speaker_candidates[0]['speaker']
            
        # If no candidates found, return None
        return None
    else:
        # Standard approach for normal-length segments
        # First, try exact match
        for segment in speaker_segments:
            if segment['start'] <= timestamp <= segment['end']:
                return segment['speaker']
    
        # If no exact match, look for the closest segment within the window
        closest_segment = None
        min_distance = float('inf')
        
        for segment in speaker_segments:
            # Calculate distance to segment
            if timestamp < segment['start']:
                distance = segment['start'] - timestamp
            elif timestamp > segment['end']:
                distance = timestamp - segment['end']
            else:
                distance = 0  # Inside segment
            
            # Update if this is closer than previous best match and within context window
            if distance < min_distance and distance <= context_window:
                min_distance = distance
                closest_segment = segment
        
        return closest_segment['speaker'] if closest_segment else None

def transcription_exists(input_file: str) -> bool:
    """
    Check if the main transcript output file exists for the given input audio file.
    """
    import os  # Import os within the function scope
    base = os.path.splitext(input_file)[0]
    transcript_file = f"{base}.txt"
    return os.path.exists(transcript_file)

def main(input_file: str, model_size: Optional[str] = None, output_dir: Optional[str] = None, 
         skip_summary: bool = False, force: bool = False, is_batch_mode: bool = False,
         regenerate_summary_only: bool = False, skip_vocabulary: bool = False,
         model: Optional[WhisperModel] = None, use_diarization: Optional[bool] = None,
         min_speakers: Optional[int] = None, max_speakers: Optional[int] = None,
         diarization_model: Optional[str] = None) -> None:
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
        use_diarization: Override config setting for speaker diarization
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        diarization_model: Specific diarization model to use
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
        
        # Perform speaker diarization if enabled
        speaker_segments = None
        # Determine if diarization should be used (CLI argument overrides config)
        should_use_diarization = use_diarization if use_diarization is not None else USE_SPEAKER_DIARIZATION
        
        if should_use_diarization:
            logging.info("Performing speaker diarization...")
            
            # Temporarily override the global DIARIZATION_MODEL if specified
            original_model = None
            if diarization_model:
                global DIARIZATION_MODEL  # Declare global before using it
                original_model = DIARIZATION_MODEL
                DIARIZATION_MODEL = diarization_model
                logging.info(f"Using custom diarization model: {DIARIZATION_MODEL}")
            
            # Perform diarization
            speaker_segments = perform_speaker_diarization(
                input_file, 
                min_speakers=min_speakers or DIARIZATION_MIN_SPEAKERS,
                max_speakers=max_speakers or DIARIZATION_MAX_SPEAKERS
            )
            
            # Restore original model name if it was changed
            if original_model:
                DIARIZATION_MODEL = original_model
                
            if not speaker_segments:
                logging.warning("Speaker diarization failed or no speakers detected")
                print("Sprekerherkenning mislukt of overgeslagen")
        
        # Transcribe with speaker information if available
        segments, info = transcribe_audio(model, input_file, speaker_segments)
        logging.info(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        # Write transcript files with speaker information
        full_transcript = write_transcript_files(segments, output_file, output_file_timestamped, speaker_segments)
        
        # Check output file size before summary
        if not skip_summary and os.path.exists(output_file):
            file_size_kb = os.path.getsize(output_file) / 1024
            logging.info(f"Transcript file size: {file_size_kb:.1f} KB")
            check_file_size(output_file)
            # Extra warning for extremely large files
            if file_size_kb > MAX_FILE_SIZE_KB * 2:
                logging.warning(f"File is extremely large ({file_size_kb:.1f} KB). Processing may take significant time.")
        
        # Generate and save summary, blog, and history extraction (if not skipped)
        if not skip_summary:
            base_path, _ = os.path.splitext(input_file)
            os.environ["WORKFLOW_OUTPUT_BASE"] = base_path
            logging.info(f"Set WORKFLOW_OUTPUT_BASE to {base_path} for speaker assignment workflow.")
            results = process_transcript_workflow(full_transcript)
            if results:
                logging.info("Workflow results keys: %s", list(results.keys()))
                write_workflow_outputs(results, base_path)
                if results.get('speaker_assignment'):
                    logging.info(f"Speaker assignment output successfully generated for {base_path}.")
                else:
                    logging.warning(f"Speaker assignment output was not generated for {base_path}.")
            else:
                logging.error("Failed to process transcript workflow")
    except Exception as e:
        logging.error(f"An error occurred processing {input_file}: {str(e)}")
        if not is_batch_mode:
            sys.exit(1)
        # In batch mode, we continue to the next file

def process_batch(input_pattern: str, **kwargs) -> None:
    """
    Process multiple files matching a glob pattern.
    
    Args:
        input_pattern: Glob pattern to match files
        **kwargs: Additional arguments to pass to main()
    """
    files = glob.glob(input_pattern)
    logging.info(f"Found {len(files)} files matching pattern: {input_pattern}")
    
    model = setup_model(kwargs.get('model_size', MODEL_SIZE))
    
    for file in files:
        try:
            main(file, model=model, is_batch_mode=True, **kwargs)
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
            
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
    Process all MP3 files in a directory.
    
    Args:
        directory: Directory to search for MP3 files
        **kwargs: Additional arguments to pass to main()
    """
    if not os.path.isdir(directory):
        logging.error(f"Directory does not exist: {directory}")
        return
    
    pattern = os.path.join(directory, "**", "*.mp3")
    files = glob.glob(pattern, recursive=True)
    logging.info(f"Found {len(files)} MP3 files in directory: {directory}")
    
    model = setup_model(kwargs.get('model_size', MODEL_SIZE))
    
    for file in files:
        try:
            main(file, model=model, is_batch_mode=True, **kwargs)
        except Exception as e:
            logging.error(f"Error processing {file}: {str(e)}")
            
    cleanup_resources(model)

# ==================== TRANSCRIPTION FUNCTIONS ====================
def show_cuda_diagnostics():
    """Toon diagnostische informatie over CUDA en GPU configuratie"""
    print("\n=== CUDA Diagnostics ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        # Test CUDA met een eenvoudige operatie
        x = torch.rand(5, 3)
        print("Test CUDA operatie: ", end="")
        try:
            x = x.cuda()
            print("Succesvol")
        except Exception as e:
            print(f"Fout: {e}")
    print("=======================\n")

def setup_model(model_size: str) -> WhisperModel:
    """
    Set up the Whisper model.
    
    Args:
        model_size: Size of the model to use
        
    Returns:
        Initialized WhisperModel
    """
    logging.info(f"Loading Whisper model: {model_size}")
    
    # Toon CUDA diagnostiek
    show_cuda_diagnostics()
    
    # Determine device based on available hardware
    device = DEVICE
    if device.lower() == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    if device.lower() != "cpu":
        logging.info(f"CUDA is available, using device: {device}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("Using CPU for transcription (slower)")
        
    # Initialize model
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=COMPUTE_TYPE,
        download_root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "whisper"),
    )
    logging.info(f"Successfully loaded {model_size} model")
    return model

def transcribe_audio(model: WhisperModel, audio_file: str, speaker_segments: Optional[List[Dict]] = None) -> Tuple[List, object]:
    """
    Transcribe audio file using the Whisper model.
    
    Args:
        model: Initialized WhisperModel instance
        audio_file: Path to the audio file to transcribe
        speaker_segments: Optional list of speaker segments from diarization
        
    Returns:
        Tuple of (segments, info)
    """
    import os  # Add the missing import here
    
    logging.info(f"Transcribing audio file: {audio_file}")
    print(f"\nStart transcriptie van {os.path.basename(audio_file)}...")
    start_time = time.time()
    
    # Load custom vocabulary if enabled
    word_list = None
    word_replacements = {}
    if USE_CUSTOM_VOCABULARY and os.path.exists(VOCABULARY_FILE):
        try:
            with open(VOCABULARY_FILE, 'r', encoding='utf-8') as f:
                vocabulary = json.load(f)
            
            # Maak een lijst van woorden voor word_list parameter
            word_list = []
            
            # Lees vervangingen voor post-processing
            for original, replacement in vocabulary.items():
                word_replacements[original.lower()] = replacement
                # Voeg ook de vervanging toe aan de word_list
                word_list.append(replacement)
            
            if word_list:
                logging.info(f"Loaded {len(word_list)} custom vocabulary words")
                print(f"Aangepaste woordenlijst geladen met {len(word_list)} woorden")
        except Exception as e:
            logging.error(f"Error loading vocabulary: {str(e)}")
    
    # Maak basic transcriptie parameters
    transcription_params = {
        'beam_size': BEAM_SIZE,
        'word_timestamps': True,  # Enable word timestamps for better alignment with diarization
        'vad_filter': True,       # Filter out non-speech parts
        'vad_parameters': dict(min_silence_duration_ms=500),  # Configure VAD for better accuracy
        'initial_prompt': "This is a podcast transcription.",
        'condition_on_previous_text': True,
    }
    
    # Bijhouden van huidige spreker om herhalingen te vermijden
    current_speaker = None
    
    # Definieer functie voor live output
    def process_segment(segment):
        text = segment.text
        
        # Pas woordenboek toe op elke segment
        if word_replacements:
            for original, replacement in word_replacements.items():
                # Vervang hele woorden met case-insensitive match
                pattern = r'\b' + re.escape(original) + r'\b'
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Voeg sprekerinfo toe als beschikbaar
        speaker_info = ""
        nonlocal current_speaker
        
        if speaker_segments:
            # Gebruik middelpunt van het segment om spreker te bepalen
            segment_middle = (segment.start + segment.end) / 2
            speaker = get_speaker_for_segment(segment_middle, speaker_segments)
            
            # Altijd speaker tonen, niet alleen bij wisseling
            if speaker:
                speaker_info = f"[{speaker}] "
                current_speaker = speaker
            else:
                speaker_info = "[SPEAKER_UNKNOWN] "
        
        # Toon de tekst direct in de console
        print(f"{format_timestamp(segment.start, segment.end)} {speaker_info}{text}")
        
        # Pas de tekst toe in het segment
        segment.text = text
        return segment
    
    # Function to collect segments with real-time processing
    def collect_with_live_output(segments_generator):
        print("\n--- Live Transcriptie Output ---")
        result = []
        for segment in segments_generator:
            # Process each segment for replacements and logging
            segment = process_segment(segment)
            result.append(segment)
        print("--- Einde Live Transcriptie ---\n")
        return result
    
    # Execute transcription with appropriate parameters
    try:
        # Voeg word_list alleen toe als deze is gedefinieerd en niet leeg is
        if word_list:
            try:
                # Probeer met word_list
                print("Transcriptie uitvoeren met aangepaste woordenlijst...")
                segments_generator, info = model.transcribe(
                    audio_file,
                    **transcription_params,
                    word_list=word_list
                )
                segments_list = collect_with_live_output(segments_generator)
            except TypeError as e:
                # Als word_list niet wordt ondersteund, probeer zonder
                logging.warning(f"word_list parameter niet ondersteund in deze versie van faster_whisper: {e}")
                logging.info("Transcriptie uitvoeren zonder aangepaste woordenlijst")
                print("word_list niet ondersteund, transcriptie uitvoeren zonder aangepaste woordenlijst...")
                segments_generator, info = model.transcribe(
                    audio_file, 
                    **transcription_params
                )
                segments_list = collect_with_live_output(segments_generator)
        else:
            # Als er geen word_list is, gebruik de standaard parameters
            print("Transcriptie uitvoeren...")
            segments_generator, info = model.transcribe(
                audio_file,
                **transcription_params
            )
            segments_list = collect_with_live_output(segments_generator)
            
    except Exception as e:
        logging.error(f"Error tijdens transcriptie: {str(e)}")
        raise
    
    # Bereken en toon verwerkingstijd
    elapsed_time = time.time() - start_time
    print(f"\nTranscriptie voltooid in {elapsed_time:.1f} seconden.")
    print(f"Taal gedetecteerd: {info.language} (waarschijnlijkheid: {info.language_probability:.2f})")
    print(f"Aantal segmenten: {len(segments_list)}")
    
    logging.info(f"Transcription complete: {len(segments_list)} segments")
    return segments_list, info

def write_transcript_files(segments: List, output_file: str, output_file_timestamped: str, 
                          speaker_segments: Optional[List[Dict]] = None) -> str:
    """
    Write transcript files with and without timestamps, integrating speaker info.
    
    Args:
        segments: List of transcription segments from Whisper
        output_file: Path to save the clean transcript
        output_file_timestamped: Path to save the timestamped transcript
        speaker_segments: Optional list of speaker segments from diarization
        
    Returns:
        Full transcript text
    """
    try:
        # Prepare for writing the transcripts
        full_transcript = []
        timestamped_transcript = []
        
        # Track current speaker to avoid repeating speaker tags for consecutive segments
        current_speaker = None
        
        print(f"\nTranscriptie bestanden aanmaken...")
        print(f"- Schoon transcript: {os.path.basename(output_file)}")
        print(f"- Tijdgemarkeerd transcript: {os.path.basename(output_file_timestamped)}")
        
        # Process each segment from Whisper
        for i, segment in enumerate(tqdm(segments, desc="Transcriptie verwerken", unit="segment")):
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            
            if not text:  # Skip empty segments
                continue
            
            # Calculate segment duration for better speaker detection of short segments
            segment_duration = end - start
            
            # Format timestamp for the timestamped version
            timestamp = format_timestamp(start, end)
            
            # Get speaker info if available
            speaker_info = ""
            speaker_prefix = ""
            if speaker_segments:
                # Use middle of segment to determine speaker with improved context-awareness
                middle_time = (start + end) / 2
                # Pass segment duration to the improved function
                speaker = get_speaker_for_segment(middle_time, speaker_segments, segment_duration=segment_duration)
                
                if speaker:
                    # Check if this is a short utterance that should inherit speaker from context
                    is_short_utterance = segment_duration < 1.0 and len(text.split()) <= 5
                    
                    # For very short utterances with no clear speaker, try to maintain speaker continuity
                    if is_short_utterance and not speaker and current_speaker:
                        # Check if there's another segment right after this one
                        if i < len(segments) - 1:
                            next_segment = segments[i + 1]
                            if next_segment.start - end < 1.0:  # If next segment starts within 1 second
                                next_middle = (next_segment.start + next_segment.end) / 2
                                next_speaker = get_speaker_for_segment(next_middle, speaker_segments)
                                # If next segment has same speaker as current, keep it for continuity
                                if next_speaker == current_speaker:
                                    speaker = current_speaker
                    
                    if speaker:
                        speaker_info = f"[{speaker}] "
                        speaker_prefix = f"[{speaker}] "
                        current_speaker = speaker
                    else:
                        # More aggressive search for a speaker with a wider context window
                        speaker = get_speaker_for_segment(middle_time, speaker_segments, 
                                                         segment_duration=segment_duration, 
                                                         context_window=2.0)
                        if speaker:
                            speaker_info = f"[{speaker}] "
                            speaker_prefix = f"[{speaker}] "
                            current_speaker = speaker
                        else:
                            # Only mark as unknown if we are sure it's not part of previous speaker's utterance
                            if current_speaker and segment_duration < 1.5 and start - prev_end < 1.0:
                                # This is likely a continuation from previous speaker
                                speaker_info = f"[{current_speaker}] "
                                speaker_prefix = f"[{current_speaker}] "
                            else:
                                speaker_info = "[SPEAKER_UNKNOWN] "
                                speaker_prefix = "[SPEAKER_UNKNOWN] "
                                current_speaker = None
                else:
                    speaker_info = "[SPEAKER_UNKNOWN] "
                    speaker_prefix = "[SPEAKER_UNKNOWN] "
                    current_speaker = None
            
            # Add to transcript collections
            clean_line = f"{speaker_prefix}{text}"
            timestamped_line = f"{timestamp} {speaker_info}{text}"
            
            full_transcript.append(clean_line)
            timestamped_transcript.append(timestamped_line)
            
            # Store the end time for checking continuity in the next iteration
            prev_end = end
        
        # Join all lines
        full_text = "\n".join(full_transcript)
        timestamped_text = "\n".join(timestamped_transcript)
        
        # Write to files
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        with open(output_file_timestamped, 'w', encoding='utf-8') as f:
            f.write(timestamped_text)
            
        print(f"Transcriptie bestanden succesvol aangemaakt:")
        print(f"- {output_file}")
        print(f"- {output_file_timestamped}")
        
        return full_text
    except Exception as e:
        logging.error(f"Error writing transcript files: {str(e)}")
        return ""

def format_timestamp(start: float, end: Optional[float] = None) -> str:
    """
    Format a timestamp in seconds to HH:MM:SS format.
    
    Args:
        start: Start timestamp in seconds
        end: Optional end timestamp in seconds
        
    Returns:
        Formatted timestamp string
    """
    def _format_single(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    
    if end is None:
        return _format_single(start)
    else:
        return f"[{_format_single(start)} --> {_format_single(end)}]"

def process_summary(transcript: str, output_summary_file: str, output_blog_file: str) -> bool:
    """
    Process the transcript to generate and save summary and blog.
    
    Args:
        transcript: The complete transcript text
        output_summary_file: Path to save the summary
        output_blog_file: Path to save the blog
        
    Returns:
        Success status (True/False)
    """
    try:
        logging.info("Genereren van samenvatting...")
        print(f"\nGenereren van samenvatting van de transcriptie...")
        
        # Read the summary prompt
        summary_prompt = read_prompt_file(PROMPT_SUMMARY_FILE)
        if not summary_prompt:
            logging.error("Samenvatting prompt bestand niet gevonden of leeg")
            return False
        
        # Choose model based on transcript length
        model_to_use = choose_appropriate_model(transcript)
        logging.info(f"Gekozen model voor samenvatting: {model_to_use}")
        
        # Generate summary using OpenAI
        if USE_RECURSIVE_SUMMARIZATION and estimate_token_count(transcript) > MAX_INPUT_TOKENS:
            logging.info("Gebruiken van recursieve samenvatting vanwege lengte van transcript")
            print("Transcript is lang, recursieve samenvatting wordt gebruikt...")
            summary = summarize_large_transcript(transcript, summary_prompt)
        else:
            print("Samenvatting genereren...")
            summary = process_with_openai(transcript, summary_prompt, model_to_use)
        
        if not summary:
            logging.error("Kon geen samenvatting genereren")
            return False
        
        # Save summary
        with open(output_summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logging.info(f"Samenvatting opgeslagen in: {output_summary_file}")
        print(f"Samenvatting opgeslagen in: {os.path.basename(output_summary_file)}")
        
        # Generate blog post
        logging.info("Genereren van blog post...")
        print(f"Genereren van blog post...")
        
        # Read the blog prompt
        blog_prompt = read_prompt_file(PROMPT_BLOG_FILE)
        if not blog_prompt:
            logging.warning("Blog prompt bestand niet gevonden of leeg, sla blog generatie over")
            return True  # We still succeeded with the summary
        
        # Combine transcript and summary for better blog generation
        input_text = f"TRANSCRIPT:\n{transcript}\n\nSUMMARY:\n{summary}"
        
        # Generate blog using OpenAI
        blog = process_with_openai(input_text, blog_prompt, model_to_use)
        
        if not blog:
            logging.error("Kon geen blog post genereren")
            # We still return True since the summary was successful
            return True
        
        # Save blog
        with open(output_blog_file, 'w', encoding='utf-8') as f:
            f.write(blog)
        logging.info(f"Blog post opgeslagen in: {output_blog_file}")
        print(f"Blog post opgeslagen in: {os.path.basename(output_blog_file)}")
        
        # Generate HTML and Wiki versions of the blog
        try:
            # HTML version
            output_blog_html = output_blog_file.replace('.txt', '.html')
            html_content = convert_markdown_to_html(blog)
            with open(output_blog_html, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logging.info(f"HTML blog opgeslagen in: {output_blog_html}")
            
            # Wiki version
            output_blog_wiki = output_blog_file.replace('.txt', '.wiki')
            wiki_content = convert_markdown_to_wiki(blog)
            with open(output_blog_wiki, 'w', encoding='utf-8') as f:
                f.write(wiki_content)
            logging.info(f"Wiki blog opgeslagen in: {output_blog_wiki}")
        except Exception as e:
            logging.warning(f"Kon geen HTML/Wiki versies genereren: {str(e)}")
        
        return True
        
    except Exception as e:
        logging.error(f"Fout bij genereren van samenvatting/blog: {str(e)}")
        return False

def convert_markdown_to_html(markdown_text: str) -> str:
    """
    Convert Markdown text to HTML.
    
    Args:
        markdown_text: The Markdown text to convert
        
    Returns:
        HTML formatted text
    """
    try:
        html = markdown.markdown(
            markdown_text,
            extensions=['extra', 'codehilite', 'tables']
        )
        
        # Voeg een eenvoudig HTML-sjabloon toe
        html_template = f"""<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WHYcast Blog Post</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        a {{ color: #3498db; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        blockquote {{ border-left: 4px solid #ccc; padding-left: 15px; margin-left: 0; color: #555; }}
        code {{ background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; font-family: monospace; }}
        pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""
        return html_template
    except Exception as e:
        logging.error(f"Fout bij conversie naar HTML: {str(e)}")
        return markdown_text  # Als fallback, geef de oorspronkelijke markdown terug

def convert_markdown_to_wiki(markdown_text: str) -> str:
    """
    Convert Markdown text to MediaWiki markup format.
    This is a simple conversion that handles common elements.
    
    Args:
        markdown_text: The Markdown text to convert
        
    Returns:
        MediaWiki formatted text
    """
    try:
        wiki_text = markdown_text
        
        # Headers (Markdown: # Header, MediaWiki: == Header ==)
        wiki_text = re.sub(r'^# (.*?)$', r'= \1 =', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^## (.*?)$', r'== \1 ==', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^### (.*?)$', r'=== \1 ===', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^#### (.*?)$', r'==== \1 ====', wiki_text, flags=re.MULTILINE)
        
        # Bold (Markdown: **text**, MediaWiki: '''text''')
        wiki_text = re.sub(r'\*\*(.*?)\*\*', r"'''\1'''", wiki_text)
        
        # Italic (Markdown: *text*, MediaWiki: ''text'')
        wiki_text = re.sub(r'\*(.*?)\*', r"''\1''", wiki_text)
        
        # Lists (Markdown: - item, MediaWiki: * item)
        wiki_text = re.sub(r'^- (.*?)$', r'* \1', wiki_text, flags=re.MULTILINE)
        
        # Links (Markdown: [text](url), MediaWiki: [url text])
        wiki_text = re.sub(r'\[(.*?)\]\((.*?)\)', r'[\2 \1]', wiki_text)
        
        return wiki_text
    except Exception as e:
        logging.error(f"Fout bij conversie naar Wiki: {str(e)}")
        return markdown_text  # Als fallback, geef de oorspronkelijke markdown terug

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
    parser.add_argument('--regenerate-cleaned', '-rc', action='store_true', 
                         help='Regenerate cleaned version from existing transcript')
    parser.add_argument('--skip-vocabulary', action='store_true', help='Skip custom vocabulary corrections')
    parser.add_argument('--regenerate-all-cleaned', action='store_true',
                        help='Regenerate cleaned transcripts for all transcript files in directory (uses input dir or download-dir if not provided)')
    parser.add_argument('--regenerate-full-workflow', action='store_true',
                        help='Run a single workflow to generate cleaned, summary, blog, and blog_alt1 from existing transcript')
    parser.add_argument('--regenerate-blogs-from-cleaned', action='store_true',
                       help='Regenerate blog posts using only cleaned transcripts')
    parser.add_argument('--regenerate-all-history', action='store_true',
                       help='Generate history extractions for all transcript files in directory')
    parser.add_argument('--regenerate-speaker-assignment', action='store_true',
                       help='Regenerate speaker assignment from an existing transcript file')
    
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
    
    # Add speaker diarization arguments
    parser.add_argument('--diarize', action='store_true',
                       help='Enable speaker diarization (override config setting)')
    parser.add_argument('--no-diarize', action='store_true',
                       help='Disable speaker diarization (override config setting)')
    parser.add_argument('--min-speakers', type=int,
                       help=f'Minimum number of speakers for diarization (default: {DIARIZATION_MIN_SPEAKERS})')
    parser.add_argument('--max-speakers', type=int,
                       help=f'Maximum number of speakers for diarization (default: {DIARIZATION_MAX_SPEAKERS})')
    parser.add_argument('--diarization-model', 
                       help=f'Diarization model to use (default: {DIARIZATION_MODEL}, alt: {DIARIZATION_ALTERNATIVE_MODEL})')
    parser.add_argument('--huggingface-token', type=str,
                       help='Set HuggingFace API token for speaker diarization')
    
    # Add arguments for history extraction
    parser.add_argument('--generate-history', '-H', action='store_true',
                       help='Generate history lesson extraction from an existing transcript file')
    # Note: --regenerate-all-history is already defined earlier in the code
    
    args = parser.parse_args()
    
    logging.info(f"WHYcast Transcribe {VERSION} starting up")
    
    # Set logging level based on verbosity
    if args.verbose:
        # Alleen het hoofdprogramma logger op debug level zetten,
        # niet alle externe modules
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        logging.info("Verbose modus ingeschakeld: Extra logging voor hoofdprogramma")
    else:
        # Zorg dat alle loggers op INFO of hoger staan
        for name in logging.root.manager.loggerDict:
            if name.startswith('matplotlib') or name.startswith('PIL') or \
               name.startswith('urllib3') or name.startswith('httpx') or \
               name.startswith('huggingface_hub'):
                logging.getLogger(name).setLevel(logging.WARNING)
            else:
                logging.getLogger(name).setLevel(logging.INFO)
    
    # Set HuggingFace token if provided
    if args.huggingface_token:
        set_huggingface_token(args.huggingface_token)
        logging.info("HuggingFace token set from command line argument")
    
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
                                not args.regenerate_blogs_from_cleaned and
                                not args.regenerate_summary)
    
    if should_process_all_episodes:
        feed_url = args.feed
        download_dir = args.download_dir
        
        logging.info(f"Processing all episodes from {feed_url}")
        process_all_episodes(feed_url, download_dir, model_size=args.model, 
                           output_dir=args.output_dir, skip_summary=args.skip_summary, 
                           force=args.force, use_diarization=args.diarize if args.diarize else None if not args.no_diarize else False,
                           min_speakers=args.min_speakers, max_speakers=args.max_speakers,
                           diarization_model=args.diarization_model)
        sys.exit(0)
    
    # Determine if we should check the podcast feed for latest episode (original behavior)
    should_check_feed = (not args.no_download and 
                         not args.all_episodes and
                         not args.input and 
                         not args.regenerate_all_summaries and
                         not args.regenerate_all_blogs and
                         not args.regenerate_blogs_from_cleaned and
                         not args.regenerate_summary)
    
    if should_check_feed:
        feed_url = args.feed
        download_dir = args.download_dir
        
        logging.info(f"Checking for the latest episode from {feed_url}")
        episode_file = download_latest_episode(feed_url, download_dir, force=args.force)
        
        if (episode_file):
            logging.info(f"Processing newly downloaded episode: {episode_file}")
            main(episode_file, model_size=args.model, output_dir=args.output_dir, 
                 skip_summary=args.skip_summary, force=args.force,
                 use_diarization=args.diarize if args.diarize else None if not args.no_diarize else False,
                 min_speakers=args.min_speakers, max_speakers=args.max_speakers,
                 diarization_model=args.diarization_model)
            sys.exit(0)
        else:
            logging.info("No new episode to download or process")
            sys.exit(0)
    
    # Continue with existing functionality if input is provided or special mode is requested
    if args.regenerate_blogs_from_cleaned:
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating blogs from cleaned transcripts in directory: {directory}")
        regenerate_blogs_from_cleaned(directory)
    elif args.regenerate_all_blogs:
        # Allow regenerate_all_blogs without an input by using podcasts directory
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating all blogs in directory: {directory}")
        regenerate_all_blogs(directory)
    elif args.regenerate_all_summaries:
        # Allow regenerate_all_summaries without an input by using podcasts directory
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating all summaries in directory: {directory}")
        regenerate_all_summaries(directory)
    elif args.regenerate_all_cleaned:
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating all cleaned transcripts in directory: {directory}")
        regenerate_all_cleaned(directory)
    elif not args.input:
        parser.print_help()
        sys.exit(1)
    elif args.regenerate_cleaned:
        if os.path.isdir(args.input):
            logging.error("Please specify a transcript file when using --regenerate-cleaned, not a directory")
            sys.exit(1)
        elif not os.path.isfile(args.input):
            logging.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        else:
            success = regenerate_cleaned_transcript(args.input)
            if not success:
                sys.exit(1)
    elif args.regenerate_summary:
        if os.path.isdir(args.input):
            logging.error("Please specify a transcript file when using --regenerate-summary, not a directory")
            sys.exit(1)
        elif not os.path.isfile(args.input):
            logging.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        else:
            main(args.input, regenerate_summary_only=True, skip_vocabulary=args.skip_vocabulary,
                use_diarization=args.diarize if args.diarize else None if not args.no_diarize else False,
                min_speakers=args.min_speakers, max_speakers=args.max_speakers,
                diarization_model=args.diarization_model)
    elif args.regenerate_full_workflow:
        if not args.input:
            logging.error("You must specify an input file or directory with --regenerate-full-workflow.")
            sys.exit(1)
        
        if os.path.isdir(args.input):
            # Process all transcripts in directory
            logging.info(f"Running full workflow for all transcripts in directory: {args.input}")
            regenerate_all_full_workflow(args.input)
        else:
            # Process single file
            regenerate_full_workflow(args.input)
        sys.exit(0)
    elif args.all_mp3s:
        process_all_mp3s(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
                        force=args.force, skip_vocabulary=args.skip_vocabulary,
                        use_diarization=args.diarize if args.diarize else None if not args.no_diarize else False,
                        min_speakers=args.min_speakers, max_speakers=args.max_speakers,
                        diarization_model=args.diarization_model)
    elif args.batch:
        process_batch(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
                    force=args.force, skip_vocabulary=args.skip_vocabulary,
                    use_diarization=args.diarize if args.diarize else None if not args.no_diarize else False,
                    min_speakers=args.min_speakers, max_speakers=args.max_speakers,
                    diarization_model=args.diarization_model)
    elif args.regenerate_all_history:
        directory = args.input if args.input else args.download_dir
        logging.info(f"Generating history extractions for all transcripts in directory: {directory}")
        regenerate_all_history_extractions(directory, force=args.force)
    elif args.generate_history:
        if os.path.isdir(args.input):
            logging.error("Please specify a transcript file when using --generate-history, not a directory")
            sys.exit(1)
        elif not os.path.isfile(args.input):
            logging.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        else:
            success = generate_history_extraction(args.input, force=args.force)
            if not success:
                sys.exit(1)
    elif args.regenerate_speaker_assignment:
        if os.path.isdir(args.input):
            logging.error("Please specify a transcript file when using --regenerate-speaker-assignment, not a directory")
            sys.exit(1)
        elif not os.path.isfile(args.input):
            logging.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        else:
            # Read transcript
            with open(args.input, 'r', encoding='utf-8') as f:
                transcript = f.read()
            base = os.path.splitext(args.input)[0]
            # Set output base for workflow
            os.environ["WORKFLOW_OUTPUT_BASE"] = base
            from content_generator import process_speaker_assignment_workflow
            try:
                txt_path = process_speaker_assignment_workflow(transcript, base)
                logging.info(f"Speaker assignment generated: {txt_path}")
                print(f"Speaker assignment generated: {txt_path}")
            except Exception as e:
                logging.error(f"Speaker assignment failed: {e}")
                print(f"Speaker assignment failed: {e}")
                sys.exit(1)
            sys.exit(0)
    else:
        main(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
            force=args.force, skip_vocabulary=args.skip_vocabulary,
            use_diarization=args.diarize if args.diarize else None if not args.no_diarize else False,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers,
            diarization_model=args.diarization_model)
