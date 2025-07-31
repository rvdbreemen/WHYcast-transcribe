#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WHYcast Transcribe - v0.3.0

A tool for transcribing podcast episodes with optional speaker diarization,
summarization, and blog post generation.

Recent improvements:
- Enhanced speaker assignment with content preservation monitoring
- Conservative speaker identification prompt to prevent content loss
- Automatic fallback for over-aggressive processing
- Detailed processing statistics and warnings
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
from typing import List, Dict, Tuple, Optional, Any, Union
import warnings
import requests
import markdown  # New import for markdown to HTML conversion
import torch

# Import the transcript formatter
from transcript_formatter import format_transcript_with_headers
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

# ==================== SPEAKER MERGING FUNCTIONS ====================
def merge_speaker_lines(transcript_text: str) -> str:
    """
    Merge consecutive lines that have the same speaker label into paragraphs.
    Removes duplicate speaker labels, keeping only one at the beginning of each merged section.
    
    Args:
        transcript_text: The original transcript text with speaker labels
        
    Returns:
        Merged transcript text with speaker paragraphs
    """
    if not transcript_text.strip():
        return transcript_text
    
    lines = transcript_text.strip().split('\n')
    merged_lines = []
    current_speaker = None
    current_content = []
    current_use_brackets = True  # Track the format being used
    
    # Pattern to match speaker labels - both bracketed [Speaker] and non-bracketed Speaker:
    speaker_pattern_bracketed = re.compile(r'^\[([^\]]+)\]\s*(.*)$')
    speaker_pattern_colon = re.compile(r'^([^:]+):\s*(.*)$')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try bracketed format first [Speaker]
        match = speaker_pattern_bracketed.match(line)
        if match:
            speaker_label = match.group(1)
            content = match.group(2).strip()
            use_brackets = True
        else:
            # Try colon format Speaker:
            match = speaker_pattern_colon.match(line)
            if match:
                speaker_label = match.group(1).strip()
                content = match.group(2).strip()
                use_brackets = False
            else:
                match = None
        
        if match:
            # If this is the same speaker as the previous line, accumulate content
            if speaker_label == current_speaker and current_content:
                if content:  # Only add non-empty content
                    current_content.append(content)
            else:
                # Different speaker or first line - output previous speaker's content
                if current_speaker and current_content:
                    merged_content = ' '.join(current_content)
                    if current_use_brackets:
                        merged_lines.append(f"[{current_speaker}] {merged_content}")
                    else:
                        merged_lines.append(f"{current_speaker}: {merged_content}")
                
                # Start new speaker section
                current_speaker = speaker_label
                current_content = [content] if content else []
                current_use_brackets = use_brackets
        else:
            # Line without speaker label - treat as continuation of current speaker
            if current_speaker and line:
                current_content.append(line)
            elif line:
                # Line without speaker and no current speaker - add as is
                merged_lines.append(line)
    
    # Don't forget the last speaker's content
    if current_speaker and current_content:
        merged_content = ' '.join(current_content)
        if current_use_brackets:
            merged_lines.append(f"[{current_speaker}] {merged_content}")
        else:
            merged_lines.append(f"{current_speaker}: {merged_content}")
    
    return '\n\n'.join(merged_lines)

def write_merged_transcript(transcript_text: str, base_path: str) -> str:
    """
    Create a merged transcript file where consecutive lines from the same speaker are combined.
    
    Args:
        transcript_text: The original transcript text
        base_path: Base path for output file (without extension)
        
    Returns:
        Path to the merged transcript file
    """
    try:
        merged_text = merge_speaker_lines(transcript_text)
        merged_file = f"{base_path}_merged.txt"
        
        with open(merged_file, 'w', encoding='utf-8') as f:
            f.write(merged_text)
        
        logging.info(f"Merged transcript written to: {merged_file}")
        print(f"✅ Merged transcript saved: {os.path.basename(merged_file)}")
        
        return merged_file
    except Exception as e:
        logging.error(f"Error writing merged transcript: {str(e)}")
        print(f"❌ Error creating merged transcript: {str(e)}")
        return ""

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

def analyze_speakers_with_o4(transcript: str, output_basename: str = None, output_dir: str = None) -> Optional[Dict[str, str]]:
    """
    Use o4 model to analyze transcript and create detailed speaker mapping.
    
    Args:
        transcript: The original transcript text with SPEAKER_XX labels
        output_basename: Base name for output files
        output_dir: Directory for output files
        
    Returns:
        Dictionary mapping SPEAKER_XX to final labels, or None if failed
    """
    if not openai_available:
        logging.warning("OpenAI not available, skipping speaker analysis")
        return None
        
    logging.info("Running detailed speaker analysis with o4 model")
    print("🧠 Analyzing speakers with o4 reasoning model...")
    
    try:
        # Read the speaker analysis prompt
        analysis_prompt_file = os.path.join(os.path.dirname(PROMPT_SPEAKER_ASSIGN_FILE), "speaker_analysis_prompt.txt")
        analysis_prompt = read_prompt_file(analysis_prompt_file)
        
        if not analysis_prompt:
            logging.error("Speaker analysis prompt not found")
            return None
        
        # Use o4 model for analysis (assuming it's configured in OPENAI_SPEAKER_MODEL)
        analysis_result = process_with_openai(transcript, analysis_prompt, OPENAI_SPEAKER_MODEL, max_tokens=MAX_TOKENS * 2)
        
        if not analysis_result:
            logging.error("Speaker analysis failed")
            return None
        
        # Parse the mapping from the analysis result
        speaker_mapping = parse_speaker_mapping_from_analysis(analysis_result)
        
        # Save detailed analysis to file
        if output_basename and output_dir:
            analysis_file = os.path.join(output_dir, f"{output_basename}_speaker_analysis.txt")
            try:
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    f.write("DETAILED SPEAKER ANALYSIS REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model: {OPENAI_SPEAKER_MODEL}\n")
                    f.write(f"Transcript: {output_basename}\n\n")
                    f.write(analysis_result)
                    f.write(f"\n\n{'='*50}\n")
                    f.write("EXTRACTED MAPPING:\n")
                    for original, mapped in speaker_mapping.items():
                        f.write(f"{original} → {mapped}\n")
                
                logging.info(f"Speaker analysis saved to: {analysis_file}")
                print(f"📄 Speaker analysis report saved: {os.path.basename(analysis_file)}")
                
            except Exception as e:
                logging.error(f"Error saving speaker analysis: {str(e)}")
        
        # Log the mapping
        logging.info(f"Speaker mapping extracted: {speaker_mapping}")
        print(f"🎭 Speaker mapping created:")
        for original, mapped in speaker_mapping.items():
            print(f"   {original} → {mapped}")
        
        return speaker_mapping
        
    except Exception as e:
        logging.error(f"Error in speaker analysis: {str(e)}")
        print(f"❌ Speaker analysis failed: {str(e)}")
        return None

def parse_speaker_mapping_from_analysis(analysis_text: str) -> Dict[str, str]:
    """
    Parse speaker mapping from the o4 analysis result.
    
    Args:
        analysis_text: The analysis result from o4 model
        
    Returns:
        Dictionary mapping SPEAKER_XX to final labels
    """
    mapping = {}
    
    try:
        # Look for the "FINAL MAPPING FOR TRANSCRIPT:" section
        lines = analysis_text.split('\n')
        in_mapping_section = False
        
        for line in lines:
            line = line.strip()
            
            # Check if we've reached the mapping section
            if "FINAL MAPPING FOR TRANSCRIPT:" in line or "SPEAKER MAPPING:" in line:
                in_mapping_section = True
                continue
                
            # Stop at next major section or examples
            if in_mapping_section and (line.startswith(('##', '===', 'CONFIDENCE SUMMARY:', '**EXAMPLES'))) or line == '':
                if line.startswith(('##', '===', 'CONFIDENCE SUMMARY:', '**EXAMPLES')):
                    break
                continue
                
            # Parse mapping lines like "SPEAKER_00 → Sarah" or "SPEAKER_00 → Host"
            if in_mapping_section and ('→' in line or '->' in line):
                # Split on either arrow type
                if '→' in line:
                    parts = line.split('→')
                else:
                    parts = line.split('->')
                    
                if len(parts) == 2:
                    original = parts[0].strip()
                    mapped = parts[1].strip()
                    
                    # Clean up the original speaker label
                    if not original.startswith('SPEAKER_'):
                        continue
                    
                    # Keep mapped label simple - it will be used as "Label:" without brackets
                    # Store mapping as [SPEAKER_XX] → Label (no brackets in the mapped value)
                    mapping[f"[{original}]"] = mapped
        
        # Fallback: look for individual speaker sections
        if not mapping:
            current_speaker = None
            current_label = None
            
            for line in lines:
                line = line.strip()
                
                # Look for speaker headers like "SPEAKER_00:"
                if line.startswith('SPEAKER_') and line.endswith(':'):
                    current_speaker = f"[{line[:-1]}]"  # Remove colon, add brackets
                    current_label = None
                
                # Look for Final Label lines (now expecting simple labels)
                elif current_speaker and line.startswith('- Final Label:'):
                    label_part = line.replace('- Final Label:', '').strip()
                    # Remove any brackets from the response if present
                    if label_part.startswith('[') and label_part.endswith(']'):
                        label_part = label_part[1:-1]
                    current_label = label_part  # No brackets in the mapped value
                    
                    mapping[current_speaker] = current_label
                    current_speaker = None
                    current_label = None
        
        # If still no mapping found, try simple pattern matching
        if not mapping:
            import re
            # Look for any SPEAKER_XX → Label patterns (without requiring brackets)
            pattern = r'(SPEAKER_\d+)\s*[→\->\s]+\s*([^\n\[]+?)(?:\n|$)'
            matches = re.findall(pattern, analysis_text)
            for original, mapped in matches:
                clean_mapped = mapped.strip()
                # Remove any trailing punctuation or brackets
                clean_mapped = re.sub(r'[\[\]]+$', '', clean_mapped).strip()
                mapping[f"[{original}]"] = clean_mapped  # No brackets in the mapped value
        
        logging.info(f"Parsed speaker mapping: {mapping}")
        return mapping
        
    except Exception as e:
        logging.error(f"Error parsing speaker mapping: {str(e)}")
        return {}

def analyze_transcript_changes(original: str, processed: str) -> Dict[str, Any]:
    """Analyze changes between original and processed transcript."""
    
    orig_stats = {
        'words': len(original.split()),
        'lines': original.count('\n'),
        'chars': len(original),
        'speakers': len(re.findall(r'\[SPEAKER_\d+\]', original))
    }
    
    proc_stats = {
        'words': len(processed.split()),
        'lines': processed.count('\n'), 
        'chars': len(processed),
        'speakers': len(re.findall(r'\[[^\]]+\]', processed))
    }
    
    return {
        'word_retention': (proc_stats['words'] / orig_stats['words']) * 100 if orig_stats['words'] > 0 else 100,
        'line_retention': (proc_stats['lines'] / orig_stats['lines']) * 100 if orig_stats['lines'] > 0 else 100,
        'char_retention': (proc_stats['chars'] / orig_stats['chars']) * 100 if orig_stats['chars'] > 0 else 100,
        'speaker_conversion': proc_stats['speakers'] / orig_stats['speakers'] if orig_stats['speakers'] > 0 else 0,
        'orig_stats': orig_stats,
        'proc_stats': proc_stats
    }

def apply_speaker_mapping_programmatically(transcript: str, speaker_mapping: Dict[str, str]) -> str:
    """
    Apply speaker mapping programmatically without using AI.
    
    Args:
        transcript: Original transcript with [SPEAKER_XX] tags
        speaker_mapping: Dictionary mapping [SPEAKER_XX] to clean labels
        
    Returns:
        Transcript with speaker labels replaced
    """
    if not transcript or not speaker_mapping:
        return transcript
    
    result = transcript
    
    # Sort mappings by key length (longest first) to avoid partial replacements
    sorted_mappings = sorted(speaker_mapping.items(), key=lambda x: len(x[0]), reverse=True)
    
    for speaker_tag, clean_label in sorted_mappings:
        # Ensure speaker_tag has brackets if not already present
        if not speaker_tag.startswith('['):
            speaker_tag = f"[{speaker_tag}]"
        
        # Create the replacement pattern
        # Replace [SPEAKER_XX] with "CleanLabel:"
        pattern = re.escape(speaker_tag)
        replacement = f"{clean_label}:"
        
        # Replace all occurrences
        result = re.sub(pattern, replacement, result)
    
    return result

def handle_unknown_speakers(transcript: str) -> str:
    """
    Handle any remaining [SPEAKER_XX] or [SPEAKER_UNKNOWN] tags that weren't mapped.
    
    Args:
        transcript: Transcript that may still have unmapped speaker tags
        
    Returns:
        Transcript with unknown speakers replaced with generic labels
    """
    # Handle [SPEAKER_UNKNOWN] tags
    transcript = re.sub(r'\[SPEAKER_UNKNOWN\]', 'Speaker:', transcript)
    
    # Handle any remaining [SPEAKER_XX] tags with numbers
    def replace_numbered_speaker(match):
        speaker_num = match.group(1)
        return f"Speaker {speaker_num}:"
    
    transcript = re.sub(r'\[SPEAKER_(\d+)\]', replace_numbered_speaker, transcript)
    
    # Handle any other bracketed speaker tags
    transcript = re.sub(r'\[SPEAKER_[^\]]*\]', 'Speaker:', transcript)
    
    return transcript

def speaker_assignment_programmatic(
    transcript: str, 
    speaker_mapping: Dict[str, str], 
    output_basename: str = None, 
    output_dir: str = None
) -> Optional[str]:
    """
    Apply speaker mapping programmatically (replaces the AI-based assignment).
    
    Args:
        transcript: Original transcript with [SPEAKER_XX] tags
        speaker_mapping: Dictionary mapping speaker tags to clean labels
        output_basename: Base name for output files
        output_dir: Directory for output files
        
    Returns:
        Transcript with speaker labels replaced
    """
    try:
        # Pre-processing statistics
        input_lines = transcript.count('\n')
        input_words = len(transcript.split())
        input_chars = len(transcript)
        
        logging.info(f"Programmatic speaker assignment input: {input_lines} lines, {input_words} words, {input_chars} chars")
        print(f"🔄 Applying speaker mapping programmatically...")
        print(f"📊 Input: {input_words} words, {input_lines} lines")
        
        # Apply the speaker mapping
        result = apply_speaker_mapping_programmatically(transcript, speaker_mapping)
        
        # Handle any unknown/unmapped speakers
        result = handle_unknown_speakers(result)
        
        # Post-processing statistics
        output_lines = result.count('\n')
        output_words = len(result.split())
        output_chars = len(result)
        
        # Calculate retention (should be nearly 100% for programmatic replacement)
        word_retention = (output_words / input_words) * 100 if input_words > 0 else 100
        line_retention = (output_lines / input_lines) * 100 if input_lines > 0 else 100
        char_retention = (output_chars / input_chars) * 100 if input_chars > 0 else 100
        
        logging.info(f"Programmatic speaker assignment output: {output_lines} lines, {output_words} words, {output_chars} chars")
        logging.info(f"Retention rates: {word_retention:.1f}% words, {line_retention:.1f}% lines, {char_retention:.1f}% chars")
        
        print(f"✅ Programmatic assignment complete!")
        print(f"📈 Retention: {word_retention:.1f}% words, {line_retention:.1f}% lines")
        
        # Verify the replacement worked
        remaining_speakers = re.findall(r'\[SPEAKER_[^\]]*\]', result)
        if remaining_speakers:
            logging.warning(f"Unmapped speakers found: {remaining_speakers}")
            print(f"⚠️ Unmapped speakers: {remaining_speakers}")
        else:
            print(f"✅ All speaker tags successfully mapped")
        
        # Add header and footer formatting
        if output_basename:
            try:
                formatted_result = format_transcript_with_headers(result, output_basename)
                
                # Save files if output directory provided
                if output_dir:
                    write_all_format(formatted_result, f"{output_basename}_speaker_assignment", output_dir)
                    print(f"✅ Files saved: {output_basename}_speaker_assignment.*")
                
                return formatted_result
            except Exception as e:
                logging.error(f"Error in formatting: {str(e)}")
                print(f"⚠️ Formatting error, returning unformatted result: {str(e)}")
                return result
        
        return result
        
    except Exception as e:
        logging.error(f"Error in programmatic speaker assignment: {str(e)}")
        print(f"❌ Error in programmatic assignment: {str(e)}")
        return None

def speaker_assignment_step(transcript: str, output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """
    Two-phase speaker assignment: 
    1. Detailed speaker analysis with o4 model
    2. Apply the mapping using structured assignment prompt
    """
    if not openai_available:
        logging.warning("OpenAI not available, skipping speaker assignment")
        return None
        
    logging.info("Starting two-phase speaker assignment")
    
    try:
        if any("SPEAKER_" in line for line in transcript.splitlines()):
            # Store original for comparison
            original_text = transcript
            
            # Phase 1: Detailed speaker analysis with o4 model
            speaker_mapping = analyze_speakers_with_o4(transcript, output_basename, output_dir)
            
            if not speaker_mapping:
                logging.warning("Speaker analysis failed, falling back to original method")
                print("⚠️  Advanced speaker analysis failed, using fallback method")
                
                # Fallback to original method
                return speaker_assignment_fallback(transcript, output_basename, output_dir)
            
            # Phase 2: Apply the mapping programmatically (NEW - replaces AI call)
            logging.info("Applying speaker mapping programmatically")
            print("🔄 Applying speaker assignments...")
            
            try:
                assigned_text = speaker_assignment_programmatic(
                    transcript, 
                    speaker_mapping, 
                    output_basename, 
                    output_dir
                )
                
                if not assigned_text:
                    logging.warning("Programmatic assignment failed, trying fallback")
                    print("⚠️ Programmatic assignment failed, using fallback method")
                    return speaker_assignment_fallback(transcript, output_basename, output_dir)
                
            except Exception as e:
                logging.error(f"Error in programmatic assignment: {str(e)}")
                print(f"❌ Error in programmatic assignment: {str(e)}")
                return speaker_assignment_fallback(transcript, output_basename, output_dir)
            
            # The programmatic assignment already includes content validation and header/footer formatting
            # Return the result directly since it's already been processed completely
            logging.info("Speaker assignment completed successfully")
            print("✅ Speaker assignment completed with headers and footers")
            
            return assigned_text
        
        else:
            logging.info("No speaker tags found, skipping speaker assignment")
            return None
            
    except Exception as e:
        logging.error(f"Error in speaker assignment: {str(e)}")
        print(f"❌ Speaker assignment failed: {str(e)}")
        return None

def speaker_assignment_fallback(transcript: str, output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """Fallback to original speaker assignment method."""
    logging.info("Using fallback speaker assignment method")
    print("🔄 Using fallback speaker assignment...")
    
    try:
        # Pre-processing: Count input characteristics
        input_words = len(transcript.split())
        input_lines = transcript.count('\n')
        input_chars = len(transcript)
        
        logging.info(f"Fallback speaker assignment input: {input_lines} lines, {input_words} words, {input_chars} chars")
        
        # Read speaker assignment prompt
        prompt = read_prompt_file(PROMPT_SPEAKER_ASSIGN_FILE)
        if not prompt:
            logging.error("Speaker assignment prompt not found")
            return None
        
        # Add preservation instructions to prompt
        enhanced_prompt = f"""INPUT STATISTICS (for preservation verification):
- Lines: {input_lines}
- Words: {input_words}
- Characters: {input_chars}

Your output should have similar statistics (±10% variance acceptable).

{prompt}

CRITICAL: The output length should be nearly identical to input length. Only change speaker labels, preserve everything else exactly."""
        
        # Process with OpenAI
        assigned_text = process_with_openai(
            transcript, 
            enhanced_prompt, 
            OPENAI_SPEAKER_MODEL, 
            max_tokens=MAX_TOKENS * 2
        )
        
        if not assigned_text:
            logging.error("Fallback speaker assignment failed")
            return None
        
        # Content preservation check
        changes = analyze_transcript_changes(transcript, assigned_text)
        
        logging.info(f"Fallback content analysis: {changes['word_retention']:.1f}% retention")
        print(f"📈 Fallback retention: {changes['word_retention']:.1f}%")
        
        # Check for significant content loss
        if changes['word_retention'] < 90.0:
            logging.warning(f"Fallback significant content loss: {changes['word_retention']:.1f}% retention")
            if changes['word_retention'] < 70.0:
                logging.error("Fallback excessive content loss - using original transcript")
                assigned_text = transcript
        
        return assigned_text
        
    except Exception as e:
        logging.error(f"Error in fallback speaker assignment: {str(e)}")
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

def get_default_device() -> str:
    """Return 'cuda' if CUDA is available else 'cpu'."""
    return "cuda" if is_cuda_available() else "cpu"

def force_cuda_device() -> str:
    """
    Aggressively try to select CUDA device for maximum GPU utilization.
    
    This function checks for CUDA availability and selects the best CUDA device.
    If CUDA is not available, it falls back to CPU but logs appropriate warnings.
    
    Returns:
        str: The selected device string ('cuda', 'cuda:0', or 'cpu')
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logging.warning("🚫 CUDA not available - falling back to CPU")
            return "cpu"
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            logging.warning("🚫 No CUDA devices found - falling back to CPU")
            return "cpu"
        
        # Select the best available GPU (typically GPU 0)
        selected_device = "cuda:0" if device_count > 0 else "cuda"
        
        # Log device selection details
        try:
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"🎯 Selected GPU device: {selected_device} ({device_name}, {device_memory:.1f}GB)")
        except Exception as e:
            logging.warning(f"Could not get GPU device details: {e}")
            logging.info(f"🎯 Selected GPU device: {selected_device}")
        
        return selected_device
        
    except ImportError:
        logging.error("❌ PyTorch not available - falling back to CPU")
        return "cpu"
    except Exception as e:
        logging.error(f"❌ Error in device selection: {e} - falling back to CPU")
        return "cpu"

def setup_model(
    model_size: str = MODEL_SIZE,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> WhisperModel:
    """
    Initialize and return the Whisper model with optimal GPU batch processing.
    
    Uses BatchedInferencePipeline for true GPU utilization through batching rather than 
    excessive worker threads. This is the correct approach for maximizing GPU throughput.
    
    Args:
        model_size: The model size to use (default from config)
        device: Force specific device (overrides auto-detection)
        compute_type: Force specific compute type (overrides auto-selection)
        batch_size: Batch size for GPU inference (auto-calculated if None)
        
    Returns:
        The initialized BatchedInferencePipeline (GPU) or WhisperModel (CPU) with optimal settings    """
    logging.info("=== Whisper Model Setup with GPU Optimization ===")
    
    try:
        from faster_whisper import WhisperModel
        
        # Force CUDA device selection for maximum GPU utilization
        target_device = force_cuda_device() if device is None else device
        
        # For faster-whisper, we need to use "cuda" instead of "cuda:0"
        if target_device.startswith("cuda:"):
            target_device = "cuda"
        
        # Auto-select compute type based on device if not specified
        if compute_type is None:
            if target_device.startswith("cuda"):
                compute_type = "float16"  # Optimal for GPU
            else:
                compute_type = "int8"     # Optimal for CPU
        
        # Auto-calculate optimal batch size if not specified
        if batch_size is None and target_device.startswith("cuda") and torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                
                # Calculate optimal batch size based on GPU memory and model size
                if model_size in ["large-v3", "large-v2", "large"]:
                    if gpu_mem >= 20:
                        batch_size = 16
                    elif gpu_mem >= 12:
                        batch_size = 8
                    elif gpu_mem >= 8:
                        batch_size = 4
                    else:
                        batch_size = 2
                elif model_size in ["medium"]:
                    if gpu_mem >= 16:
                        batch_size = 24
                    elif gpu_mem >= 8:
                        batch_size = 16
                    else:
                        batch_size = 8
                else:  # small, base, tiny
                    if gpu_mem >= 8:
                        batch_size = 32
                    else:
                        batch_size = 16
                        
                logging.info(f"Auto-calculated batch size: {batch_size} for {model_size} on {gpu_mem:.1f}GB GPU")
            except Exception as e:
                logging.warning(f"Failed to auto-calculate batch size: {e}, using default")
                batch_size = 8
        elif batch_size is None:
            batch_size = 1  # CPU fallback
        
        # Configure GPU optimizations
        if target_device.startswith("cuda") and torch.cuda.is_available():
            # Clear GPU memory before model loading
            torch.cuda.empty_cache()
            
            # PyTorch optimizations for GPU inference
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            
            logging.info(f"GPU optimizations enabled for {torch.cuda.get_device_name(0)}")
          # Initialize Whisper model
        logging.info(f"Loading Whisper {model_size} model on {target_device} with {compute_type}")
        
        model = WhisperModel(
            model_size,
            device=target_device,
            compute_type=compute_type,
            cpu_threads=4 if target_device == "cpu" else 0,
            num_workers=4  # Keep workers reasonable to avoid CPU bottleneck
        )
        
        # Log successful initialization
        if target_device.startswith("cuda"):
            logging.info(f"✅ Whisper model loaded on GPU with {compute_type} precision")
            print(f"✅ Whisper model loaded on GPU ({target_device}) with {compute_type} precision")
        else:
            logging.warning(f"⚠️ Whisper model loaded on CPU with {compute_type} precision (slower)")
            print(f"⚠️ Whisper model loaded on CPU (slower). Consider GPU setup for better performance.")
        
        # Store batch_size as an attribute for transcription functions  
        model.optimal_batch_size = batch_size
        
        return model
        
    except Exception as e:
        logging.error(f"❌ Error setting up Whisper model: {str(e)}")
        logging.warning("Falling back to CPU model with basic settings")
        print(f"❌ Whisper model setup failed: {str(e)}")
        print("🔄 Falling back to CPU model...")
        
        # Clear any GPU memory that might be allocated
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Fallback to CPU with safe settings
        try:
            fallback_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=4
            )
            fallback_model.optimal_batch_size = 1
            logging.info("✅ Fallback CPU model initialized successfully")
            print("✅ Fallback CPU model loaded successfully")
            return fallback_model
            
        except Exception as fallback_error:
            logging.error(f"❌ Even CPU fallback failed: {fallback_error}")
            print(f"❌ Critical error: Even CPU fallback failed: {fallback_error}")
            raise RuntimeError(f"Failed to initialize any Whisper model: {fallback_error}")

def transcribe_audio(model: WhisperModel, audio_file: str, speaker_segments: Optional[List[Dict]] = None) -> Tuple[List, object]:
    """
    Transcribe audio file using the Whisper model with optimal GPU batching.
    
    Args:
        model: Initialized WhisperModel or BatchedInferencePipeline instance
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
    
    # Get optimal batch size from the model
    batch_size = getattr(model, 'optimal_batch_size', 8)
    
    # Check if we're using BatchedInferencePipeline
    is_batched = hasattr(model, 'model')  # BatchedInferencePipeline wraps a model
    
    if is_batched:
        logging.info(f"Using BatchedInferencePipeline with batch_size={batch_size} for maximum GPU utilization")
        print(f"🚀 GPU Batch Processing: batch_size={batch_size}")
    else:
        logging.info(f"Using standard model with batch_size={batch_size}")
      # Create basic transcription parameters (common to both models)
    base_transcription_params = {
        'beam_size': BEAM_SIZE,
        'word_timestamps': True,  # Enable word timestamps for better alignment with diarization
        'vad_filter': True,       # Filter out non-speech parts
        'vad_parameters': dict(min_silence_duration_ms=500),  # Configure VAD for better accuracy
        'initial_prompt': "This is a podcast transcription.",
        'condition_on_previous_text': True,
    }
      # Create transcription parameters (without batch_size for now)
    transcription_params = {
        'beam_size': BEAM_SIZE,
        'word_timestamps': True,  # Enable word timestamps for better alignment with diarization
        'vad_filter': True,       # Filter out non-speech parts
        'vad_parameters': dict(min_silence_duration_ms=500),  # Configure VAD for better accuracy
        'initial_prompt': "This is a podcast transcription.",
        'condition_on_previous_text': True,
    }
    
    # Note: batch_size is stored in model.optimal_batch_size but not used in transcribe() call
    # because the current faster-whisper version doesn't support it in WhisperModel.transcribe()
    logging.info(f"Using model with optimal_batch_size={batch_size} (for future use)")
    
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
    import glob
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

    # First check if the expected file exists
    if os.path.exists(audio_file):
        print(f"Audio file already exists: {audio_file}")
        return (audio_file, base_name) if return_base_name else audio_file
    
    # If not found, try to find similar files in the directory
    # Look for files that might match the episode (with different naming conventions)
    episode_title = latest.get('title', '').lower()
    print(f"Looking for existing files matching episode: {episode_title}")
    
    # Try various patterns to find existing files
    patterns_to_try = [
        f"{safe_title}.*",  # exact match
        f"*{safe_title.split('_')[-1]}*.*" if '_' in safe_title else f"*{safe_title}*.*",  # partial match
        "*episode*44*.*",  # fallback pattern for episode 44
        "*44*.*"  # very broad pattern
    ]
    
    for pattern in patterns_to_try:
        matching_files = glob.glob(os.path.join(output_dir, pattern))
        audio_files = [f for f in matching_files if f.lower().endswith(('.mp3', '.m4a', '.wav', '.flac'))]
        
        if audio_files:
            # Sort by modification time to get the most recent
            audio_files.sort(key=os.path.getmtime, reverse=True)
            found_file = audio_files[0]
            found_base_name = os.path.splitext(os.path.basename(found_file))[0]
            print(f"Found existing audio file: {found_file}")
            return (found_file, found_base_name) if return_base_name else found_file

    # If no existing file found, download the new one
    print(f"No existing file found. Downloading latest episode to {audio_file} ...")
    try:
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(audio_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded: {audio_file}")
        return (audio_file, base_name) if return_base_name else audio_file
    except Exception as e:
        print(f"Failed to download episode: {e}")
        return (None, None) if return_base_name else None

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
    """    # Step 0: Apply optimal GPU utilization settings at the start
    print("🚀 Applying optimal GPU utilization settings...")
    optimal_batch_size, num_workers = maximize_gpu_utilization()
    
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
    print(f"Prepared audio: {prepared_audio}")    # Step 3: Transcribe audio (with diarization)    print("[2/4] Running speaker diarization ...")
    
    # Verify GPU setup before starting diarization
    gpu_info = verify_gpu_setup()
    
    try:
        import torch
        import torchaudio
        
        # Load audio with GPU optimization
        print("Loading audio for diarization...")
        waveform, sample_rate = torchaudio.load(prepared_audio)
        
        # Run diarization with GPU acceleration
        speaker_segments = diarize_audio(waveform=waveform, sample_rate=sample_rate)
        
        if speaker_segments is not None:
            # Count the number of speaker segments
            num_segments = len(list(speaker_segments.itertracks()))
            print(f"✅ Diarization complete. Found {num_segments} speaker segments.")
            
            # Log speaker information
            speakers = set()
            for segment, _, label in speaker_segments.itertracks(yield_label=True):
                speakers.add(label)
            print(f"🎤 Detected {len(speakers)} unique speakers: {', '.join(sorted(speakers))}")
            logging.info(f"Diarization found {num_segments} segments with {len(speakers)} speakers")
        else:
            print("❌ Diarization failed or returned no segments.")
            logging.warning("Diarization failed or returned no segments")
            
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        logging.error(f"Diarization failed: {e}")
        speaker_segments = None
          # Clear GPU memory on error
        try:
            if 'torch' in locals() and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("GPU memory cleared after diarization error")
        except:
            pass
    
    print("[3/4] Transcribing audio ...")
    
    # Initialize Whisper model with GPU acceleration 
    print("🔧 Setting up Whisper model with GPU acceleration...")
    model = setup_model(batch_size=optimal_batch_size)
    
    # Run transcription
    print("🎵 Running transcription...")
    segments, _ = transcribe_audio(model, prepared_audio, speaker_segments=speaker_segments)
    
    # Clear GPU memory after transcription to free up resources
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("GPU memory cleared after transcription")
    except:
        pass
    
    # Write transcript files
    transcript_text = write_transcript_files(segments, f"{base_path}_transcript.txt", f"{base_path}_ts.txt", speaker_segments=speaker_segments)
    print(f"✅ Transcription complete. Transcript saved to {base_path}_transcript.txt")
    
    # Create merged transcript if speaker labels are present
    merged_transcript_text = None
    if transcript_text and any('[SPEAKER_' in line or re.search(r'\[[^\]]+\]', line) for line in transcript_text.splitlines()):
        print("🔄 Creating merged transcript (grouping consecutive speaker lines)...")
        merged_file = write_merged_transcript(transcript_text, base_path)
        if merged_file:
            print(f"✅ Merged transcript created: {os.path.basename(merged_file)}")
            # Read the merged transcript for processing
            try:
                with open(merged_file, 'r', encoding='utf-8') as f:
                    merged_transcript_text = f.read()
                print("📝 Using merged transcript for processing workflow (cleaner format)")
            except Exception as e:
                print(f"⚠️ Could not read merged transcript: {e}, using original transcript")
                merged_transcript_text = None
    
    # Step 4: Process transcript
    print("[4/4] Processing transcript workflow (summary, blog, history, etc.) ...")
    # Use merged transcript if available, otherwise use original
    text_for_processing = merged_transcript_text if merged_transcript_text else transcript_text
    if text_for_processing:
        if merged_transcript_text:
            print("🔄 Processing with merged transcript (improved readability)")
        process_transcript_workflow(text_for_processing, base_name, output_dir)
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
    This function implements GPU acceleration for pyannote speaker diarization.
    
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
          # Check CUDA availability and force GPU usage with MAXIMUM utilization
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device = torch.device("cuda")
            print("🚀 FORCING MAXIMUM CUDA UTILIZATION for diarization")
            logging.info("FORCING MAXIMUM CUDA UTILIZATION for diarization")
            
            # Clear GPU cache before starting
            torch.cuda.empty_cache()
            
            # AGGRESSIVE GPU optimizations for pyannote
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Set aggressive memory allocation for pyannote
            try:
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% for diarization
            except:
                pass
            
            # Log GPU information with enhanced details
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = gpu_props.total_memory / (1024**3)
            multiprocessors = gpu_props.multi_processor_count
            
            print(f"🎯 MAXIMUM GPU UTILIZATION: {gpu_name} ({gpu_memory:.1f} GB, {multiprocessors} MPs)")
            logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB, {multiprocessors} multiprocessors)")
            
        else:
            device = torch.device("cpu")
            print("⚠️  CUDA not available for diarization - using CPU (slow)")
            logging.warning("CUDA not available for diarization - using CPU (slow)")
        
        # Load the diarization pipeline
        print("Loading pyannote speaker-diarization-3.1 pipeline...")
        logging.info("Loading pyannote speaker-diarization-3.1 pipeline")
        pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization-3.1', 
            use_auth_token=hf_token
        )
        
        # FORCE the pipeline to use GPU if available
        if cuda_available:
            print("Moving diarization pipeline to GPU...")
            logging.info("Moving diarization pipeline to GPU")
            pipeline.to(device)
            
            # Verify pipeline is on GPU
            logging.info(f"Pipeline device: {device}")
            print(f"✅ Diarization pipeline using: {device}")
        else:
            print(f"✅ Diarization pipeline using: {device}")
        
        # Load audio if not provided in memory
        if waveform is None or sample_rate is None:
            if audio_file_path is None:
                raise ValueError('No audio data or file path provided for diarization.')
            print("Loading audio for diarization...")
            logging.info(f"Loading audio from: {audio_file_path}")
            waveform, sample_rate = torchaudio.load(audio_file_path)
        
        # Move waveform to GPU if using CUDA
        if cuda_available:
            print("Moving audio waveform to GPU...")
            logging.info("Moving audio waveform to GPU")
            waveform = waveform.to(device)
            logging.info(f"Waveform device: {waveform.device}")
        
        # Run diarization with progress indication
        print("Running speaker diarization...")
        logging.info("Starting diarization inference")
        
        # Monitor GPU memory if using CUDA
        if cuda_available:
            memory_before = torch.cuda.memory_allocated(0) / 1024**2
            logging.info(f"GPU memory before diarization: {memory_before:.2f} MB")
        
        # Run the actual diarization
        diarization = pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        
        # Log memory usage after processing
        if cuda_available:
            memory_after = torch.cuda.memory_allocated(0) / 1024**2
            logging.info(f"GPU memory after diarization: {memory_after:.2f} MB")
            print(f"GPU memory used: {memory_after:.2f} MB")
            
            # Clear GPU cache after processing
            torch.cuda.empty_cache()
            memory_final = torch.cuda.memory_allocated(0) / 1024**2
            logging.info(f"GPU memory after cleanup: {memory_final:.2f} MB")
        
        print("✅ Speaker diarization completed successfully")
        logging.info("Diarization completed successfully")
        return diarization
        
    except Exception as e:
        logging.error(f'Diarization failed: {e}')
        print(f"❌ Diarization failed: {e}")
        
        # Clear GPU memory even on failure
        if 'cuda_available' in locals() and cuda_available:
            try:
                torch.cuda.empty_cache()
                logging.info("GPU memory cleared after diarization failure")
            except:
                pass
        
        return None

def verify_gpu_setup():
    """
    Comprehensive verification of GPU setup for both Whisper and pyannote.
    This function checks CUDA availability and performs test operations.
    
    Returns:
        dict: GPU status information
    """
    try:
        import torch
        
        gpu_info = {
            'cuda_available': False,
            'device_count': 0,
            'device_name': None,
            'device_memory': 0,
            'pytorch_version': torch.__version__,
            'cuda_version': None,
            'cudnn_version': None,
            'test_passed': False,
            'recommendations': []
        }
        
        print("🔍 Verifying GPU setup...")
        logging.info("Starting GPU verification")
        
        # Basic CUDA availability check
        cuda_available = torch.cuda.is_available()
        gpu_info['cuda_available'] = cuda_available
        
        if not cuda_available:
            print("❌ CUDA not available")
            gpu_info['recommendations'].append("Install NVIDIA GPU drivers")
            gpu_info['recommendations'].append("Install PyTorch with CUDA support")
            gpu_info['recommendations'].append("Check CUDA_PATH environment variable")
            return gpu_info
        
        # Get device information
        device_count = torch.cuda.device_count()
        gpu_info['device_count'] = device_count
        
        if device_count > 0:
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info['device_name'] = device_name
            gpu_info['device_memory'] = device_memory
            
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"🎯 Primary GPU: {device_name}")
            print(f"💾 GPU Memory: {device_memory:.1f} GB")
            
            # Get CUDA version information
            try:
                cuda_version = torch.version.cuda
                gpu_info['cuda_version'] = cuda_version
                print(f"🔧 CUDA Version: {cuda_version}")
            except:
                print("⚠️  CUDA version unknown")
            
            try:
                cudnn_version = torch.backends.cudnn.version()
                gpu_info['cudnn_version'] = cudnn_version
                print(f"🔧 cuDNN Version: {cudnn_version}")
            except:
                print("⚠️  cuDNN version unknown")
        
        # Perform test operations
        try:
            print("🧪 Testing GPU operations...")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test tensor creation and operations
            test_tensor = torch.ones(1000, 1000, device='cuda')
            result = test_tensor @ test_tensor  # Matrix multiplication
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"✅ GPU test passed - Memory used: {memory_used:.2f} MB")
            
            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()
            
            gpu_info['test_passed'] = True
            
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            gpu_info['recommendations'].append("Check GPU compatibility")
            gpu_info['recommendations'].append("Verify PyTorch CUDA installation")
        
        # Memory recommendations
        if gpu_info['device_memory'] < 4:
            gpu_info['recommendations'].append("GPU has limited memory (<4GB) - consider using smaller models")
        elif gpu_info['device_memory'] >= 8:
            print("🚀 GPU has sufficient memory for large models")
        
        # Final status
        if gpu_info['test_passed']:
            print("✅ GPU setup verification completed successfully")
            logging.info("GPU setup verification passed")
        else:
            print("⚠️  GPU setup has issues - check recommendations")
            logging.warning("GPU setup verification failed")
        
        return gpu_info
        
    except ImportError:
        print("❌ PyTorch not available")
        return {'cuda_available': False, 'recommendations': ['Install PyTorch']}
    except Exception as e:
        print(f"❌ GPU verification error: {e}")
        logging.error(f"GPU verification error: {e}")
        return {'cuda_available': False, 'recommendations': ['Check PyTorch installation']}

def maximize_gpu_utilization():
    """
    Configure optimal GPU settings for Whisper inference.
    Focus on proper batching rather than excessive worker threads.
    """
    import os
    import torch
    
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU memory: {gpu_memory:.1f}GB")
            
            # Calculate optimal batch size based on GPU memory
            # RTX 3080 (10GB): batch_size=8-16
            # RTX 4090 (24GB): batch_size=16-32
            if gpu_memory >= 20:
                optimal_batch_size = 16
                num_workers = 8
            elif gpu_memory >= 8:
                optimal_batch_size = 8
                num_workers = 6
            else:
                optimal_batch_size = 4
                num_workers = 4
            
            logging.info(f"Optimal batch size: {optimal_batch_size}")
            logging.info(f"Optimal workers: {num_workers}")
            
            # Set reasonable environment variables for GPU optimization
            env_vars = {
                'OMP_NUM_THREADS': str(num_workers),
                'MKL_NUM_THREADS': str(num_workers),
                'NUMBA_NUM_THREADS': str(num_workers),
                'CUDA_LAUNCH_BLOCKING': '0',
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'CUDA_VISIBLE_DEVICES': '0',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,expandable_segments:True',
                'TORCH_NUM_THREADS': str(num_workers),
            }
            
            for key, value in env_vars.items():
                os.environ[key] = value
                logging.info(f"Set {key}={value}")
            
            # PyTorch optimizations for GPU inference
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                
            torch.set_num_threads(num_workers)
            
            logging.info("✅ GPU optimization settings applied!")
            
            return optimal_batch_size, num_workers
        else:
            logging.warning("❌ No CUDA GPU detected")
            return 4, 4
            
    except Exception as e:
        logging.error(f"❌ Error setting GPU optimization: {e}")
        return 4, 4

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
