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
from typing import List, Dict, Tuple, Optional, Any
import warnings
import requests
import markdown  # New import for markdown to HTML conversion
import torch
import hashlib
import uuid
import feedparser
import subprocess  # Ensure this is always imported at the top

# Initialize availability flags as module-level variables
openai_available = False
feedparser_available = False
tqdm_available = False

# Suppress specific warnings
warnings.filterwarnings("ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio")
warnings.filterwarnings("ignore", message="The bits_per_sample of .mp3 is set to 0 by default")
warnings.filterwarnings("ignore", message="PySoundFile failed")

# Enable TensorFloat-32 for improved performance on NVIDIA Ampere GPUs
if torch.cuda.is_available():
    # Check if we have an Ampere or newer GPU
    compute_capability = torch.cuda.get_device_capability(0)
    if compute_capability[0] >= 8:  # Ampere GPUs have compute capability 8.0+
        logging.info("Enabling TensorFloat-32 for improved performance on Ampere GPU")
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
    from openai import BadRequestError
    openai_available = True
except ImportError:
    logging.warning("openai not installed. Summarization and blog generation will not be available.")
    # Create fallback class for BadRequestError
    class BadRequestError(Exception):
        pass

# Custom security exception
class SecurityError(Exception):
    """Raised when a security-related validation fails"""
    pass

try:
    import feedparser
    feedparser_available = True
except ImportError:
    logging.warning("feedparser not installed. RSS feed parsing will not be available.")

try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    logging.warning("tqdm not installed. Progress bars will not be available.")
    
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

# process_speaker_assignment_workflow is now defined below in this file

# Set up logging to both console and file
def setup_logging():
    # Enhanced log format with line numbers and function names
    log_format = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Default level set to INFO
    
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
    try:        # Configure console for UTF-8
        if sys.platform == 'win32':
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
    Ensure that the OpenAI API key is available and valid.
    
    Returns:
        str: The API key if available and valid
        
    Raises:
        ValueError: If the API key is not set or invalid
        SecurityError: If the API key format is suspicious
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set your OpenAI API key in the .env file or environment variables."
        )
    
    # Security: Basic validation of API key format
    api_key = api_key.strip()
    if len(api_key) < 20:
        raise ValueError("OPENAI_API_KEY appears to be too short to be valid")
    
    # Security: Check for suspicious characters that could indicate injection
    if any(char in api_key for char in [';', '&', '|', '`', '$', '\n', '\r']):
        raise SecurityError("OPENAI_API_KEY contains suspicious characters")
    
    # Security: Basic format check for OpenAI API keys (should start with 'sk-')
    if not api_key.startswith('sk-'):
        logging.warning("OPENAI_API_KEY doesn't start with 'sk-', this may not be a valid OpenAI API key")
    
    return api_key

def validate_file_path(file_path: str, must_exist: bool = True, check_readable: bool = True) -> str:
    """
    Validate and sanitize a file path for security.
    
    Args:
        file_path: The file path to validate
        must_exist: Whether the file must exist
        check_readable: Whether to check if the file is readable
        
    Returns:
        The validated and normalized file path
        
    Raises:
        ValueError: If the file path is invalid
        SecurityError: If the file path contains suspicious elements
        FileNotFoundError: If the file must exist but doesn't
    """
    if not file_path or not isinstance(file_path, str):
        raise ValueError("File path must be a non-empty string")
    
    # Security: Remove any null bytes
    file_path = file_path.replace('\0', '')
    
    # Security: Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/') or '\\' in file_path.replace(os.sep, ''):
        # Allow normal Windows paths but block suspicious ones
        normalized = os.path.normpath(file_path)
        if '..' in normalized:
            raise SecurityError(f"Path traversal detected in file path: {file_path}")
    
    # Security: Check for suspicious characters
    suspicious_chars = ['<', '>', '|', '*', '?', '"']
    if any(char in file_path for char in suspicious_chars):
        raise SecurityError(f"Suspicious characters detected in file path: {file_path}")
    
    # Normalize the path
    normalized_path = os.path.normpath(os.path.abspath(file_path))
    
    # Check existence if required
    if must_exist and not os.path.exists(normalized_path):
        raise FileNotFoundError(f"File not found: {normalized_path}")
    
    # Check if it's actually a file (not a directory) if it exists
    if os.path.exists(normalized_path) and not os.path.isfile(normalized_path):
        raise ValueError(f"Path exists but is not a file: {normalized_path}")
    
    # Check readability if required
    if check_readable and os.path.exists(normalized_path):
        if not os.access(normalized_path, os.R_OK):
            raise PermissionError(f"File is not readable: {normalized_path}")
    
    return normalized_path

def validate_directory_path(dir_path: str, create_if_missing: bool = False) -> str:
    """
    Validate and sanitize a directory path for security.
    
    Args:
        dir_path: The directory path to validate
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        The validated and normalized directory path
        
    Raises:
        ValueError: If the directory path is invalid
        SecurityError: If the directory path contains suspicious elements
    """
    if not dir_path or not isinstance(dir_path, str):
        raise ValueError("Directory path must be a non-empty string")
    
    # Security: Remove any null bytes
    dir_path = dir_path.replace('\0', '')
    
    # Security: Check for path traversal attempts
    if '..' in dir_path:
        normalized = os.path.normpath(dir_path)
        if '..' in normalized:
            raise SecurityError(f"Path traversal detected in directory path: {dir_path}")
    
    # Security: Check for suspicious characters
    suspicious_chars = ['<', '>', '|', '*', '?', '"']
    if any(char in dir_path for char in suspicious_chars):
        raise SecurityError(f"Suspicious characters detected in directory path: {dir_path}")
    
    # Normalize the path
    normalized_path = os.path.normpath(os.path.abspath(dir_path))
    
    # Create directory if requested and it doesn't exist
    if create_if_missing and not os.path.exists(normalized_path):
        try:
            os.makedirs(normalized_path, exist_ok=True)
            logging.info(f"Created directory: {normalized_path}")
        except Exception as e:
            raise PermissionError(f"Cannot create directory {normalized_path}: {str(e)}")
    
    # Check if it's actually a directory if it exists
    if os.path.exists(normalized_path) and not os.path.isdir(normalized_path):
        raise ValueError(f"Path exists but is not a directory: {normalized_path}")
    
    return normalized_path

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
        call_id = str(uuid.uuid4())
        prompt_first_line = prompt.strip().splitlines()[0] if prompt.strip().splitlines() else prompt.strip()
        logging.info(f"[OpenAI Call {call_id}] Model: {model_name}, Prompt first line: {prompt_first_line}")
        print(f"[OpenAI Call {call_id}] Prompt: {prompt_first_line}")
        
        estimated_tokens = estimate_token_count(text)
        logging.info(f"[OpenAI Call {call_id}] Processing text with OpenAI (~{estimated_tokens} tokens)")
        
        # If text is too long, truncate it
        if estimated_tokens > MAX_INPUT_TOKENS:
            logging.warning(f"[OpenAI Call {call_id}] Text too long (~{estimated_tokens} tokens > {MAX_INPUT_TOKENS} limit), truncating...")
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
                logging.info(f"[OpenAI Call {call_id}] Using expanded token limit for cleanup: {cleanup_max_tokens}")
                
                # Check if text needs to be processed in chunks due to size
                if estimated_tokens > MAX_INPUT_TOKENS // 2:
                    return process_large_text_in_chunks(text, prompt, model_name, client, parent_call_id=call_id)
                
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
                logging.warning(f"[OpenAI Call {call_id}] Generated text may be truncated (doesn't end with punctuation)")
                
            return result
                
        except BadRequestError as e:
            if "maximum context length" in str(e).lower():
                # Try with more aggressive truncation
                logging.warning(f"[OpenAI Call {call_id}] Context length exceeded. Retrying with further truncation...")
                text = truncate_transcript(text, MAX_INPUT_TOKENS // 2)
                logging.info(f"[OpenAI Call {call_id}] Retrying with reduced text (~{estimate_token_count(text)} tokens)...")
                
                # Update the message content with truncated text
                params["messages"][1]["content"] = f"{prompt}\n\nHere's the text to process:\n\n{text}"
                
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content
            else:
                raise
    except Exception as e:
        logging.error(f"[OpenAI Call {call_id}] Error processing with OpenAI: {str(e)}")
        return None

def process_large_text_in_chunks(text: str, prompt: str, model_name: str, client: OpenAI, parent_call_id: str = None) -> str:
    """
    Process very large text by breaking it into chunks and reassembling the results.
    
    Args:
        text: The text to process
        prompt: The processing instructions
        model_name: Model to use
        client: OpenAI client
        parent_call_id: Optional parent call ID for traceability
    Returns:
        Combined processed text
    """
    call_id = str(uuid.uuid4())
    logging.info(f"[OpenAI Chunked Call {call_id}] Parent: {parent_call_id} | Model: {model_name} | Chunks incoming")
    print(f"[OpenAI Chunked Call {call_id}] Parent: {parent_call_id} | Prompt: {prompt.strip().splitlines()[0] if prompt.strip().splitlines() else prompt.strip()}")
    is_o_series_model = model_name.startswith("o") and not model_name.startswith("gpt")
    token_param = "max_completion_tokens" if is_o_series_model else "max_tokens"
    token_limit = MAX_TOKENS * 2
    modified_prompt = f"{prompt}\n\nThis is a chunk of a longer transcript. Process this chunk following the instructions."
    processed_chunks = []
    for i, chunk in enumerate(tqdm(split_into_chunks(text, max_chunk_size=MAX_CHUNK_SIZE*2), desc="Processing chunks")):
        chunk_call_id = str(uuid.uuid4())
        chunk_first_line = modified_prompt.strip().splitlines()[0] if modified_prompt.strip().splitlines() else modified_prompt.strip()
        logging.info(f"[OpenAI Chunk {chunk_call_id}] Parent: {call_id} | Chunk {i+1} | Prompt: {chunk_first_line}")
        print(f"[OpenAI Chunk {chunk_call_id}] Parent: {call_id} | Chunk {i+1} | Prompt: {chunk_first_line}")
        try:
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that processes transcript chunks."},
                    {"role": "user", "content": f"{modified_prompt}\n\nChunk {i+1} of text:\n\n{chunk}"}
                ]
            }
            if not is_o_series_model:
                params["temperature"] = TEMPERATURE
            params[token_param] = token_limit
            response = client.chat.completions.create(**params)
            processed_chunks.append(response.choices[0].message.content)
            logging.info(f"[OpenAI Chunk {chunk_call_id}] Successfully processed chunk {i+1}")
        except Exception as e:
            logging.error(f"[OpenAI Chunk {chunk_call_id}] Error processing chunk {i+1}: {str(e)}")
            processed_chunks.append(chunk)
    combined_text = "\n\n".join(processed_chunks)
    if len(processed_chunks) > 1:
        try:
            logging.info(f"[OpenAI Chunked Call {call_id}] Running final pass to ensure consistency across chunk boundaries")
            combined_tokens = estimate_token_count(combined_text)
            if combined_tokens > MAX_INPUT_TOKENS:
                logging.warning(f"[OpenAI Chunked Call {call_id}] Combined text is too large for final pass (~{combined_tokens} tokens)")
                return combined_text
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that ensures transcript consistency."},
                    {"role": "user", "content": f"This is a processed transcript that was handled in chunks. Please ensure consistency across chunk boundaries and fix any obvious issues.\n\n{combined_text}"}
                ]
            }
            if not is_o_series_model:
                params["temperature"] = TEMPERATURE
            params[token_param] = MAX_TOKENS
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"[OpenAI Chunked Call {call_id}] Error in final consistency pass: {str(e)}")
            return combined_text
    return combined_text

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

def speaker_assignment_step(transcript: str, output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """
    Step 1: Run speaker assignment on the raw transcript.
    Checks for diarization tags and, if present, uses the speaker assignment workflow
    to generate a transcript with speaker names. Returns the speaker-assigned transcript as a string,
    or None if not applicable or failed.
    If output_basename and output_dir are provided, writes all formats.
    """
    logging.info("Step 1: Speaker assignment")
    try:
        if any("SPEAKER_" in line for line in transcript.splitlines()):
            # Use absolute path from config to avoid issues with working directory
            prompt = read_prompt_file(PROMPT_SPEAKER_ASSIGN_FILE)
            if not prompt:
                logging.warning("Speaker assignment prompt missing")
                return None
            chunks = split_into_chunks(transcript, max_chunk_size=MAX_INPUT_TOKENS * 4)
            results = []
            for chunk in chunks:
                result = process_with_openai(chunk, prompt, OPENAI_SPEAKER_MODEL, max_tokens=MAX_TOKENS)
                results.append(result)
            full_result = "\n\n".join(results)
            if output_basename and output_dir:
                write_all_format(full_result, f"{output_basename}_speaker_assignment", output_dir)
            return full_result
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
    summary = process_with_openai(cleaned, prompt, choose_appropriate_model(cleaned))
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

def write_all_format(markdown_text: str, output_basename: str, output_dir: str) -> None:
    """
    Write markdown text to .txt, .html, and .wiki files in the specified directory.
    Args:
        markdown_text: The markdown string to write
        output_basename: The base filename (without extension)
        output_dir: The directory to write files to
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"{output_basename}.txt")
    html_path = os.path.join(output_dir, f"{output_basename}.html")
    wiki_path = os.path.join(output_dir, f"{output_basename}.wiki")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(convert_markdown_to_html(markdown_text))
        with open(wiki_path, "w", encoding="utf-8") as f:
            f.write(convert_markdown_to_wiki(markdown_text))
    except Exception as e:
        import logging
        logging.error(f"write_all_format failed: {e}")

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
        logging.error(f"Error converting markdown to HTML: {e}")
        return ""

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
        logging.error(f"Error converting markdown to Wiki markup: {e}")
        return ""  # Return original text if conversion fails

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

def setup_model(
    model_size: str = MODEL_SIZE,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> WhisperModel:
    """
    Initialize and return the Whisper model.
    Always uses CUDA (GPU) and float16 if available, otherwise falls back to CPU.
    
    Args:
        model_size: The model size to use
    
    Returns:
        The initialized WhisperModel
    """
    try:
        logging.info("=== System Information ===")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        logging.info(f"CUDA available: {cuda_available}")

        if device is None:
            device = DEVICE
        if compute_type is None:
            compute_type = COMPUTE_TYPE

        if device == "auto":
            device = "cuda" if cuda_available else "cpu"
        if device.startswith("cuda") and not cuda_available:
            logging.warning("CUDA requested but not available. Falling back to CPU")
            device = "cpu"
        if device.startswith("cuda"):
            if cuda_available:
                cuda_version = getattr(torch.version, 'cuda', None)
                if cuda_version:
                    logging.info(f"CUDA version: {cuda_version}")
                else:
                    logging.info("CUDA version: unknown (torch.version.cuda not found)")
                logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
                device_count = torch.cuda.device_count()
                logging.info(f"GPU device count: {device_count}")
                for i in range(device_count):
                    logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
                if gpu_mem >= 16:
                    num_workers = 8
                elif gpu_mem >= 8:
                    num_workers = 4
                else:
                    num_workers = 2
                if compute_type is None:
                    compute_type = "float16"
                logging.info(f"Whisper will use CUDA (GPU). num_workers={num_workers}, compute_type={compute_type}")
            else:
                logging.warning("CUDA requested but not available. Falling back to CPU")
                device = "cpu"

        if not device.startswith("cuda"):
            num_workers = 8
            compute_type = "int8"
            logging.warning("Whisper will use CPU (slow). Consider installing a compatible GPU and CUDA drivers.")
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
        logging.error(f"Error setting up Whisper model: {str(e)}")
        logging.warning("Falling back to CPU model (int8)")
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
            
            # Create a list of words for the word_list parameter
            word_list = []
            
            # Read replacements for post-processing
            for original, replacement in vocabulary.items():
                word_replacements[original.lower()] = replacement
                # Also add the replacement to the word_list
                word_list.append(replacement)
            
            if word_list:
                logging.info(f"Loaded {len(word_list)} custom vocabulary words")
                print(f"Custom vocabulary loaded with {len(word_list)} words")
        except Exception as e:
            logging.error(f"Error loading vocabulary: {str(e)}")
    
    # Create basic transcription parameters
    transcription_params = {
        'beam_size': BEAM_SIZE,
        'word_timestamps': True,  # Enable word timestamps for better alignment with diarization
        'vad_filter': True,       # Filter out non-speech parts
        'vad_parameters': dict(min_silence_duration_ms=500),  # Configure VAD for better accuracy
        'initial_prompt': "This is a podcast transcription.",
        'condition_on_previous_text': True,
    }
    
    # Track the current speaker to avoid repeating speaker tags
    current_speaker = None

    # Define a function to determine the speaker based on diarization segments
    def find_speaker_for_segment(segment_middle, speaker_segments):
        if not speaker_segments:
            return None
        # speaker_segments is expected to be a pyannote Annotation
        # Find the segment that contains segment_middle
        for speech_turn in speaker_segments.itertracks(yield_label=True):
            segment, _, label = speech_turn
            if segment.start <= segment_middle <= segment.end:
                return label
        return None

    # Define a function for live output
    def process_segment(segment):
        text = segment.text
        
        # Apply word replacements to each segment
        if word_replacements:
            for original, replacement in word_replacements.items():
                # Replace whole words with a case-insensitive match
                pattern = r'\b' + re.escape(original) + r'\b'
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Add speaker info if available
        speaker_info = ""
        nonlocal current_speaker
        
        if speaker_segments:
            # Use the middle of the segment to determine the speaker
            segment_middle = (segment.start + segment.end) / 2
            speaker = find_speaker_for_segment(segment_middle, speaker_segments)
            
            # Always show the speaker, not only when switching
            if speaker:
                speaker_info = f"[{speaker}] "
                current_speaker = speaker
            else:
                speaker_info = "[SPEAKER_UNKNOWN] "
        
        # Display the text directly in the console
        print(f"{format_timestamp(segment.start)} {speaker_info}{text}")

        # Apply the text to the segment
        segment.text = text
        return segment
    
    # Function to collect segments with real-time processing
    def collect_with_live_output(segments_generator):
        print("\n--- Live Transcription Output ---")
        result = []
        for segment in segments_generator:
            # Process each segment for replacements and logging
            segment = process_segment(segment)
            result.append(segment)
        print("--- End Live Transcription ---\n")
        return result
    
    # Execute transcription with appropriate parameters
    try:
        # Add word_list only if it is defined and not empty
        if word_list:
            try:
                # Try with word_list
                print("Running transcription with custom vocabulary...")
                segments_generator, info = model.transcribe(
                    audio_file,
                    **transcription_params,
                    word_list=word_list
                )
                segments_list = collect_with_live_output(segments_generator)
            except TypeError as e:
                # If word_list is not supported, try without it
                logging.warning(f"word_list parameter not supported in this version of faster_whisper: {e}")
                logging.info("Running transcription without custom vocabulary")
                print("word_list not supported, running transcription without custom vocabulary...")
                segments_generator, info = model.transcribe(
                    audio_file, 
                    **transcription_params
                )
                segments_list = collect_with_live_output(segments_generator)
        else:
            # If there is no word_list, use the default parameters
            print("Running transcription...")
            segments_generator, info = model.transcribe(
                audio_file,
                **transcription_params
            )
            segments_list = collect_with_live_output(segments_generator)
            
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        raise
    
    # Calculate and display processing time
    elapsed_time = time.time() - start_time
    print(f"\nTranscription completed in {elapsed_time:.1f} seconds.")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    print(f"Segment count: {len(segments_list)}")
    
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
    # Local function to find speaker for a segment midpoint
    def find_speaker_for_segment(segment_middle, speaker_segments):
        if not speaker_segments:
            return None
        for speech_turn in speaker_segments.itertracks(yield_label=True):
            segment, _, label = speech_turn
            if segment.start <= segment_middle <= segment.end:
                return label
        return None

    try:
        # Prepare for writing the transcripts
        full_transcript = []
        timestamped_transcript = []
        # Track current speaker to avoid repeating speaker tags for consecutive segments
        current_speaker = None
        print(f"\nCreating transcript files...")
        print(f"- Clean transcript: {os.path.basename(output_file)}")
        print(f"- Timestamped transcript: {os.path.basename(output_file_timestamped)}")
        # Process each segment from Whisper
        for i, segment in enumerate(tqdm(segments, desc="Processing transcript", unit="segment")):
            start = segment.start
            end = segment.end
            text = segment.text.strip()
            if not text:  # Skip empty segments
                continue
            # Calculate segment duration for better speaker detection of short segments
            segment_duration = end - start
            # Format timestamp for the timestamped version
            timestamp = format_timestamp(start)
            # Get speaker info if available
            speaker_info = ""
            speaker_prefix = ""
            if speaker_segments:
                # Use middle of segment to determine speaker
                middle_time = (start + end) / 2
                speaker = find_speaker_for_segment(middle_time, speaker_segments)
                if speaker:
                    is_short_utterance = segment_duration < 1.0 and len(text.split()) <= 5
                    # For very short utterances with no clear speaker, try to maintain speaker continuity
                    if is_short_utterance and not speaker and current_speaker:
                        speaker = current_speaker
                    if speaker:
                        speaker_info = f"[{speaker}] "
                        speaker_prefix = f"[{speaker}] "
                        current_speaker = speaker
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
            
        print(f"Transcript files successfully created:")
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

def delete_episode_files(base_name: str, output_dir: str, exclude_files: Optional[List[str]] = None):
    """
    Delete all files related to a given episode base name in the output directory, except those in exclude_files and except .mp3 files.
    """
    patterns = [
        f"{base_name}*.*",  # matches all files for this episode
    ]
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(output_dir, pattern)):
            abs_file_path = os.path.abspath(file_path)
            # Never delete the input file or any .mp3 file
            if exclude_files and abs_file_path in [os.path.abspath(f) for f in exclude_files]:
                continue
            if abs_file_path.lower().endswith('.mp3'):
                continue
            try:
                os.remove(abs_file_path)
                logging.info(f"Deleted file: {abs_file_path}")
            except Exception as e:
                logging.error(f"Failed to delete file {abs_file_path}: {e}")

def podcast_fetching_workflow(rssfeed, output_dir, return_base_name=False):
    """
    Fetch the latest episode from the RSS feed and download the audio file.
    Returns the path to the downloaded audio file, or None if not found.
    If return_base_name is True, also returns the base name for the episode file.
    """
    feed = feedparser.parse(rssfeed)
    if not feed.entries:
        print("No episodes found in RSS feed.")
        return (None, None) if return_base_name else None

    latest = feed.entries[0]
    # Try to find the audio link
    audio_url = None
    for link in latest.get('links', []):
        if isinstance(link, dict) and str(link.get('type', '')).startswith('audio'):
            audio_url = link.get('href')
            break
    if not audio_url or not isinstance(audio_url, str):
        print("No audio file found for the latest episode.")
        return (None, None) if return_base_name else None

    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in latest.get('title', 'episode'))
    audio_ext = os.path.splitext(audio_url)[-1].split('?')[0] if '.' in os.path.basename(audio_url) else '.mp3'
    audio_file = os.path.join(output_dir, f"{safe_title}{audio_ext}")
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    if not os.path.exists(audio_file):
        print(f"Downloading latest episode to {audio_file} ...")
        try:
            with requests.get(audio_url, stream=True) as r:
                r.raise_for_status()
                with open(audio_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded: {audio_file}")
        except Exception as e:
            print(f"Failed to download episode: {e}")
            return (None, None) if return_base_name else None
    else:
        print(f"Audio file already exists: {audio_file}")
    return (audio_file, base_name) if return_base_name else audio_file

def download_all_episodes_from_rssfeed(rssfeed: str, output_dir: str = './podcasts'):
    """
    Download all mp3 audio files from the RSS feed to the output directory.
    Skips files that already exist. Shows a progress bar.
    """
    feed = feedparser.parse(rssfeed)
    if not feed.entries:
        print("No episodes found in RSS feed.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(feed.entries)} episodes in RSS feed.")
    from tqdm import tqdm
    for entry in tqdm(feed.entries, desc="Downloading episodes", unit="episode"):
        audio_url = None
        for link in entry.get('links', []):
            if isinstance(link, dict) and str(link.get('type', '')).startswith('audio'):
                audio_url = link.get('href')
                break
        if not audio_url or not isinstance(audio_url, str):
            print(f"No audio file found for episode: {entry.get('title', 'unknown')}")
            continue
        safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in entry.get('title', 'episode'))
        audio_ext = os.path.splitext(audio_url)[-1].split('?')[0] if '.' in os.path.basename(audio_url) else '.mp3'
        audio_file = os.path.join(output_dir, f"{safe_title}{audio_ext}")
        if os.path.exists(audio_file):
            tqdm.write(f"Already exists: {audio_file}")
            continue
        tqdm.write(f"Downloading: {audio_url}\n  -> {audio_file}")
        try:
            with requests.get(audio_url, stream=True) as r:
                r.raise_for_status()
                with open(audio_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            tqdm.write(f"Downloaded: {audio_file}")
        except Exception as e:
            tqdm.write(f"Failed to download {audio_url}: {e}")
    print("All episodes processed.")

def full_workflow(audio_file=None, output_dir=None, rssfeed=None, force: bool = False):
    """
    Complete workflow:
    1. If no audio_file, fetch latest episode from RSS feed.
    2. Prepare audio (normalize, etc).
    3. Transcribe audio (with diarization if available).
    4. Process transcript (summary, blog, etc).
    Args:
        audio_file: Path to audio file or None
        output_dir: Output directory (default: './podcasts' or from env)
        rssfeed: RSS feed URL (default: from env or fallback)
        force: If True, delete all related files for this episode before processing
    """
    # Step 1: Fetch latest episode if needed
    if not audio_file:
        if not rssfeed:
            rssfeed = os.environ.get("WHYCAST_RSSFEED", "https://whycast.podcast.audio/@whycast/feed.xml")
        if not output_dir:
            output_dir = os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
        print(f"No audio file provided. Fetching latest episode from {rssfeed} ...")
        result = podcast_fetching_workflow(rssfeed, output_dir, return_base_name=True)
        if not result or not isinstance(result, tuple) or not result[0]:
            print("No episode could be fetched from the feed.")
            return
        audio_file, base_name = result
    else:
        if not output_dir:
            output_dir = os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
        base_name = os.path.splitext(os.path.basename(str(audio_file)))[0] if audio_file else None
        if force and base_name:
            print(f"--force: Deleting all files for episode base name '{base_name}' in {output_dir}")
            delete_episode_files(base_name, output_dir)
    if not base_name:
        print("Could not determine base name for episode. Exiting.")
        return
    base_path = os.path.join(output_dir, base_name)
    # Step 2: Prepare audio
    print("[1/4] Preparing audio ...")
    prepared_audio = prepare_audio_for_diarization(str(audio_file))
    print(f"Prepared audio: {prepared_audio}")
    # Step 3: Transcribe audio (with diarization)
    print("[2/4] Running speaker diarization ...")
    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(prepared_audio)
        speaker_segments = diarize_audio(waveform=waveform, sample_rate=sample_rate)
        if speaker_segments is not None:
            print(f"Diarization complete. Found {len(speaker_segments)} speaker segments.")
        else:
            print("Diarization failed or returned no segments.")
    except Exception as e:
        print(f"Diarization failed: {e}")
        speaker_segments = None
    print("[3/4] Transcribing audio ...")
    model = setup_model()
    segments, _ = transcribe_audio(model, prepared_audio, speaker_segments=speaker_segments)
    transcript_text = write_transcript_files(segments, f"{base_path}_transcript.txt", f"{base_path}_ts.txt", speaker_segments=speaker_segments)
    print(f"Transcription complete. Transcript saved to {base_path}_transcript.txt")
    # Step 4: Process transcript
    print("[4/4] Processing transcript workflow (summary, blog, history, etc.) ...")
    if transcript_text:
        process_transcript_workflow(transcript_text, base_name, output_dir)
    else:
        print("Transcript text is empty, skipping transcript workflow.")
    # Delete temp audio file if it was created (prepared_audio != audio_file)
    try:
        if os.path.abspath(prepared_audio) != os.path.abspath(str(audio_file)) and os.path.exists(prepared_audio):
            os.remove(prepared_audio)
            print(f"Temporary normalized audio file deleted: {prepared_audio}")
    except Exception as e:
        print(f"Error deleting temporary audio file: {e}")

def prepare_audio_for_diarization(audio_file: str, output_dir: Optional[str] = None) -> str:
    """
    Ensures the audio file is in a compatible mono 16kHz mp3 format for diarization/transcription.
    Always creates a new file for diarization, never overwrites the original.
    Returns the path to the prepared mp3 file.
    """
    import hashlib
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    os.makedirs(output_dir, exist_ok=True)
    base, _ = os.path.splitext(os.path.basename(audio_file))
    hash_suffix = hashlib.md5(audio_file.encode('utf-8')).hexdigest()[:8]
    output_file = os.path.join(output_dir, f"{base}_mono16k_{hash_suffix}.mp3")

    if os.path.exists(output_file):
        logging.info(f"Prepared audio already exists: {output_file}")
        return output_file

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-i', audio_file, '-vn', '-ar', '16000', '-ac', '1', '-b:a', '192k', output_file
    ]
    try:
        logging.info(f"Converting {audio_file} to mono 16kHz mp3 using ffmpeg (output: {output_file})...")

        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logging.info(f"ffmpeg output: {result.stdout.decode(errors='ignore')}")
        logging.info(f"Audio converted to {output_file}")
        

        return output_file
    except Exception as e:
        logging.error(f"ffmpeg conversion failed: {e}\n{getattr(e, 'stderr', b'').decode(errors='ignore')}")
        raise RuntimeError(f"Failed to convert {audio_file} to mono 16kHz mp3: {e}")


def diarize_audio(waveform=None, sample_rate=None, audio_file_path=None, hf_token=None):
    """
    Run speaker diarization using pyannote.audio 3.1 pipeline on GPU if available, using in-memory audio.
    Args:
        waveform: torch.Tensor of shape (channels, samples), optional
        sample_rate: int, optional
        audio_file_path: str, optional, used if waveform/sample_rate not provided
        hf_token: str, Hugging Face access token
    Returns:
        diarization result (pyannote Annotation) or None on failure
    """
    try:
        import torch
        import torchaudio
        from pyannote.audio import Pipeline
        if hf_token is None:
            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=hf_token)
        if torch.cuda.is_available():
            pipeline.to(torch.device('cuda'))
            logging.info('pyannote pipeline using CUDA')
        else:
            logging.info('pyannote pipeline using CPU')
        if waveform is None or sample_rate is None:
            if audio_file_path is None:
                raise ValueError('No audio data or file path provided for diarization.')
            waveform, sample_rate = torchaudio.load(audio_file_path)
        diarization = pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        return diarization
    except Exception as e:
        logging.error(f'Diarization failed: {e}')
        return None



# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'WHYcast Transcribe v{VERSION} - Transcribe audio files and generate summaries')
    parser.add_argument('input', nargs='?', help='Path to the input audio file, directory, or glob pattern')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files', default='./podcasts')
    parser.add_argument('--rssfeed', '-r', help='RSS feed URL to fetch latest episodes from', default=os.environ.get('WHYCAST_RSSFEED', 'https://whycast.podcast.audio/@whycast/feed.xml'))
    parser.add_argument('--force', action='store_true', help='Force re-download and re-transcribe the latest episode, deleting all related files first')
    parser.add_argument('--version', action='version', version=f'WHYcast Transcribe v{VERSION}')
    parser.add_argument('--fetch-all', action='store_true', help='Download all mp3 episodes from the RSS feed to the output directory and exit')

    args = parser.parse_args()
    logging.info(f"WHYcast Transcribe {VERSION} starting up")

    # Handle --fetch-all: download all mp3s and exit before any other logic
    if args.fetch_all:
        rssfeed = args.rssfeed or os.environ.get("WHYCAST_RSSFEED", "https://whycast.podcast.audio/@whycast/feed.xml")
        output_dir = args.output_dir or os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
        download_all_episodes_from_rssfeed(rssfeed, output_dir)
        exit(0)

    # Handle --force: fetch latest, delete all related files, then run workflow
    if args.force:
        rssfeed = args.rssfeed or os.environ.get("WHYCAST_RSSFEED", "https://whycast.podcast.audio/@whycast/feed.xml")
        output_dir = args.output_dir or os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
        if args.input:
            base_name = os.path.splitext(os.path.basename(str(args.input)))[0] if args.input else None
            input_path = os.path.abspath(str(args.input))
            if base_name:
                print(f"--force: Deleting all files for episode base name '{base_name}' in {output_dir} (excluding input file)")
                delete_episode_files(base_name, output_dir, exclude_files=[input_path])
            full_workflow(audio_file=args.input, output_dir=output_dir, rssfeed=rssfeed, force=False)
            exit(0)
        else:
            result = podcast_fetching_workflow(rssfeed, output_dir, return_base_name=True)
            if not result or not isinstance(result, tuple) or not result[0] or not result[1]:
                print("No episode could be fetched from the feed.")
                exit(1)
            audio_file, base_name = result
            input_path = os.path.abspath(audio_file) if audio_file else None
            if base_name:
                print(f"--force: Deleting all files for episode base name '{base_name}' in {output_dir} (excluding input file)")
                delete_episode_files(base_name, output_dir, exclude_files=[input_path] if input_path else None)
            # Re-fetch after deletion to ensure mp3 is present
            result = podcast_fetching_workflow(rssfeed, output_dir, return_base_name=True)
            if not result or not isinstance(result, tuple) or not result[0]:
                print("Failed to download episode after deletion.")
                exit(1)
            audio_file, base_name = result
            full_workflow(audio_file=audio_file, output_dir=output_dir, rssfeed=rssfeed, force=False)
            exit(0)
    # If no parameters, check if latest mp3 is already downloaded and transcribed
    if not args.input:
        rssfeed = args.rssfeed or os.environ.get("WHYCAST_RSSFEED", "https://whycast.podcast.audio/@whycast/feed.xml")
        output_dir = args.output_dir or os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
        result = podcast_fetching_workflow(rssfeed, output_dir, return_base_name=True)
        if result and isinstance(result, tuple) and result[0] and result[1]:
            audio_file, base_name = result
            transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
            if os.path.exists(transcript_path):
                print(f"Transcript for latest episode '{base_name}' already exists at {transcript_path}. Exiting.")
                exit(0)
        # If not found, proceed as normal
        audio_file = result[0] if result and isinstance(result, tuple) else result
        full_workflow(audio_file=audio_file, output_dir=output_dir, rssfeed=rssfeed, force=False)
        exit(0)
    # Always call the main workflow with parsed arguments if input is provided
    full_workflow(
        audio_file=args.input,
        output_dir=args.output_dir,
        rssfeed=args.rssfeed,
        force=False
    )
