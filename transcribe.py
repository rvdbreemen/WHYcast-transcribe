#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WHYcast Transcribe - v0.1.0

A tool for transcribing podcast episodes with optional speaker diarization,
summarization, and blog post generation.
"""

from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import os
import sys
import re
import time
import json
import glob
import logging
import logging as py_logging
import argparse
import warnings
import hashlib
import gc
from tqdm import tqdm

# Integrate content_generator for unified content processing
try:
    import content_generator
    from content_generator import process_transcript_workflow, write_workflow_outputs, process_directory_workflow, process_speaker_assignment_workflow
    CONTENT_GENERATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"content_generator module not found or failed to import: {e}")
    CONTENT_GENERATOR_AVAILABLE = False

import torch
import markdown
import requests
import feedparser

# Import content generator module for content processing
import content_generator

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
        OPENAI_SPEAKER_ASSIGNMENT_MODEL
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

# Import utility functions from utils modules
from utils.text_processing import (
    estimate_token_count, split_into_chunks, truncate_transcript,
    load_vocabulary_mappings, apply_vocabulary_corrections
)
from utils.file_helpers import read_prompt_file, check_file_size, write_transcript_files
from utils.format_converters import convert_markdown_to_html, convert_markdown_to_wiki
from utils.openai_processor import process_with_openai, choose_appropriate_model

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
            logging.error("Failed to process transcript workflow.")
            return False
        
        # Use the imported write_workflow_outputs from content_generator.py
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
        if base_name.endswith(suffix):
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
    
    if not cleaned_files:
        logging.warning(f"No cleaned transcript files found in directory: {directory}")
        return
    
    logging.info(f"Found {len(cleaned_files)} cleaned transcript files to process")
    processed_count = 0
    
    for cleaned_file in tqdm(cleaned_files, desc="Regenerating blogs from cleaned transcripts"):
        # Derive base name without the "_cleaned" suffix
        base_path = cleaned_file.replace("_cleaned.txt", "")
        summary_file = f"{base_path}_summary.txt"
        
        # Check if summary exists
        if os.path.exists(summary_file):
            logging.info(f"Regenerating blog from cleaned transcript: {cleaned_file}")
            if regenerate_blog_only(cleaned_file, summary_file):
                processed_count += 1
            # Force garbage collection to release memory
            import gc
            gc.collect()
        else:
            logging.warning(f"Skipping {cleaned_file}: No matching summary file found")
    
    logging.info(f"Successfully regenerated {processed_count} blog posts from cleaned transcripts")

def generate_history_extraction(transcript_file: str, force: bool = False) -> bool:
    """
    Generate history extraction from a cleaned transcript or generate cleaned transcript first if needed.
    
    Args:
        transcript_file: Path to the transcript file or cleaned transcript file
        force: Flag to force regeneration even if history already exists
        
    Returns:
        Success status (True/False)
    """
    if not os.path.exists(transcript_file):
        logging.error(f"Transcript file does not exist: {transcript_file}")
        return False
    
    try:
        # Get base filename without path and extension and without _cleaned suffix if present
        base_filename = os.path.splitext(os.path.basename(transcript_file))[0]
        if base_filename.endswith("_cleaned"):
            base_filename = base_filename[:-8]  # Remove "_cleaned" from the end
        
        # Create output directory path (same as transcript file directory)
        output_dir = os.path.dirname(transcript_file)
        
        # Check if history extraction already exists
        history_path = os.path.join(output_dir, f"{base_filename}_history.txt")
        if os.path.exists(history_path) and not force:
            logging.info(f"History extraction already exists: {history_path}. Use --force to regenerate.")
            return True
            
        # Check if this is already a cleaned transcript
        if "_cleaned.txt" in transcript_file:
            cleaned_transcript_file = transcript_file
        else:
            # Check if a cleaned transcript exists
            base = os.path.splitext(transcript_file)[0]
            cleaned_transcript_file = f"{base}_cleaned.txt"
            
            # If cleaned transcript doesn't exist, generate it
            if not os.path.exists(cleaned_transcript_file):
                logging.info(f"No cleaned transcript found: {cleaned_transcript_file}, generating one...")
                if not regenerate_cleaned_transcript(transcript_file):
                    logging.error("Failed to generate cleaned transcript, cannot proceed with history extraction.")
                    return False
        
        # Read the cleaned transcript
        with open(cleaned_transcript_file, 'r', encoding='utf-8') as f:
            cleaned_transcript = f.read()
        
        # Read the history extraction prompt
        history_extract_prompt = read_prompt_file(PROMPT_HISTORY_EXTRACT_FILE)
        if not history_extract_prompt:
            logging.error("History extraction prompt file not found or empty, cannot generate history extraction.")
            return False

        logging.info("Generating history lesson extraction...")
        
        # Use the specified history model
        history_extract = process_with_openai(
            cleaned_transcript, 
            history_extract_prompt, 
            OPENAI_HISTORY_MODEL, 
            max_tokens=MAX_TOKENS * 2
        )
        
        if not history_extract:
            logging.error("Failed to generate history extraction")
            return False
        
        # Save the history extraction to the correct path
        with open(history_path, 'w', encoding='utf-8') as f:
            f.write(history_extract)
        logging.info(f"History extraction saved to: {history_path}")
        
        # Generate and write HTML version
        html_content = convert_markdown_to_html(history_extract)
        html_path = os.path.join(output_dir, f"{base_filename}_history.html")
        with open(html_path, 'w', encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML history extraction saved to: {html_path}")
        
        # Generate and write Wiki version
        wiki_content = convert_markdown_to_wiki(history_extract)
        wiki_path = os.path.join(output_dir, f"{base_filename}_history.wiki")
        with open(wiki_path, 'w', encoding="utf-8") as f:
            f.write(wiki_content)
        logging.info(f"Wiki history extraction saved to: {wiki_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error generating history extraction: {str(e)}")
        return False

def regenerate_all_history_extractions(directory: str, force: bool = False) -> None:
    """
    Generate history extractions for all transcript files in the directory.
    Will use cleaned transcripts if available or generate them otherwise.
    
    Args:
        directory: Directory containing transcript files
        force: Flag to force regeneration even if history already exists
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
    
    # First check for cleaned transcript files
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
        
        # Skip if history exists and not forcing
        if os.path.exists(history_path) and not force:
            logging.info(f"Skipping {transcript_file} - history extraction already exists (use --force to override)")
            skipped_count += 1
            continue
        
        if generate_history_extraction(transcript_file, force=force):
            processed_count += 1
        
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
    Download the latest episode from the podcast feed if not already processed, unless force is True.
    
    Args:
        feed_url: URL of the RSS feed
        download_dir: Directory to save the downloaded file
        force: If True, always re-download and reprocess
    
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
    
    # If force is True, always re-download and overwrite the file
    filename = get_episode_filename(episode)
    full_path = os.path.join(download_dir, filename)
    if not force and episode_already_processed(episode, download_dir):
        logging.info(f"Episode '{episode['title']}' has already been processed, skipping")
        return None
    
    # If force and file exists, remove it before re-downloading
    if force and os.path.exists(full_path):
        try:
            os.remove(full_path)
            logging.info(f"Removed existing file before forced re-download: {full_path}")
        except Exception as e:
            logging.warning(f"Could not remove existing file: {full_path}: {str(e)}")
    
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

def perform_speaker_diarization(audio_file: str, min_speakers: int = DIARIZATION_MIN_SPEAKERS, 
                              max_speakers: int = DIARIZATION_MAX_SPEAKERS) -> Optional[List[Dict]]:
    """
    Perform speaker diarization on an audio file using pyannote.audio.
    
    Args:
        audio_file: Path to the audio file
        min_speakers: Minimum number of speakers to identify
        max_speakers: Maximum number of speakers to identify
        
    Returns:
        List of dictionaries containing speaker segments or None if failed
    """
    try:
        from pyannote.audio import Pipeline
        
        # Define cache directory for local models
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "diarization")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get HuggingFace token from .env - don't ask if not found since we first want to check .env
        huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')
        
        if not huggingface_token:
            logging.warning("HuggingFace token niet gevonden in .env bestand.")
            # Nu pas vragen om token als deze niet in .env staat
            huggingface_token = get_huggingface_token(ask_if_missing=True)
            
        if not huggingface_token:
            logging.warning("Geen HuggingFace token beschikbaar. Speaker diarization heeft een token nodig.")
            logging.warning("Je kunt deze later instellen via de HUGGINGFACE_TOKEN environment variabele of in het .env bestand")
            return None
        
        # First try to use a local cached model if available
        pipeline = None
        model_to_use = DIARIZATION_MODEL
        local_model_path = os.path.join(cache_dir, model_to_use.replace("/", "_"))
        
        # Try loading from local cache first
        if os.path.exists(local_model_path):
            try:
                logging.info(f"Proberen om diarization model te laden van lokale cache: {local_model_path}")
                # Probeer te laden met verschillende API-versies
                try:
                    # Nieuwste versie met token parameter
                    pipeline = Pipeline.from_pretrained(
                        local_model_path,
                        token=huggingface_token
                    )
                except (TypeError, ValueError):
                    try:
                        # Oudere versie met use_auth_token parameter
                        pipeline = Pipeline.from_pretrained(
                            local_model_path,
                            use_auth_token=huggingface_token
                        )
                    except (TypeError, ValueError):
                        # Probeer zonder token (indien model lokaal al volledig beschikbaar is)
                        pipeline = Pipeline.from_pretrained(local_model_path)
                        
                logging.info(f"Diarization model succesvol geladen van lokale cache")
            except Exception as e:
                logging.warning(f"Kon model niet laden van lokale cache: {str(e)}")
                pipeline = None
        
        # If local loading failed, try downloading the model and save it
        if pipeline is None:
            try:
                logging.info(f"Diarization model downloaden: {DIARIZATION_MODEL}")
                # Probeer beide parameter namen voor token authenticatie
                try:
                    pipeline = Pipeline.from_pretrained(
                        DIARIZATION_MODEL,
                        use_auth_token=huggingface_token,
                        cache_dir=cache_dir
                    )
                except TypeError:
                    # Probeer met 'token' parameter als 'use_auth_token' niet werkt (nieuwere versies)
                    logging.info("Proberen met 'token' parameter in plaats van 'use_auth_token'")
                    pipeline = Pipeline.from_pretrained(
                        DIARIZATION_MODEL,
                        token=huggingface_token,
                        cache_dir=cache_dir
                    )
                
                # Save the model locally for future use
                try:
                    logging.info(f"Model opslaan in lokale cache: {local_model_path}")
                    # Nieuwere versies gebruiken save_pretrained in plaats van to_disk
                    if hasattr(pipeline, 'to_disk'):
                        pipeline.to_disk(local_model_path)
                    elif hasattr(pipeline, 'save_pretrained'):
                        pipeline.save_pretrained(local_model_path)
                    else:
                        logging.warning("Kon model niet opslaan: geen geschikte opslagmethode gevonden")
                        # Sla het model niet op, maar ga verder met verwerking
                    logging.info(f"Model succesvol opgeslagen in: {local_model_path}")
                except Exception as save_error:
                    logging.warning(f"Kon model niet opslaan naar schijf: {str(save_error)}")
                    # Dit is niet kritiek, ga verder met verwerking
            except Exception as primary_error:
                error_message = str(primary_error)
                logging.warning(f"Kon primaire diarization model niet laden: {error_message}")
                
                # If primary model fails, try the alternative model
                alternative_model_path = os.path.join(cache_dir, DIARIZATION_ALTERNATIVE_MODEL.replace("/", "_"))
                
                # Try loading alternative from local cache first
                if (os.path.exists(alternative_model_path)):
                    try:
                        logging.info(f"Proberen om alternatief model te laden van lokale cache: {alternative_model_path}")
                        pipeline = Pipeline.from_pretrained(
                            alternative_model_path,
                            use_auth_token=huggingface_token
                        )
                        model_to_use = DIARIZATION_ALTERNATIVE_MODEL
                        logging.info(f"Alternatief model succesvol geladen van cache.")
                    except Exception as e:
                        logging.warning(f"Kon alternatief model niet laden van cache: {str(e)}")
                        pipeline = None
                
                # If local loading failed, try downloading the alternative model
                if pipeline is None:
                    try:
                        logging.info(f"Alternatief diarization model downloaden: {DIARIZATION_ALTERNATIVE_MODEL}")
                        # Probeer beide parameter namen voor token authenticatie
                        try:
                            pipeline = Pipeline.from_pretrained(
                                DIARIZATION_ALTERNATIVE_MODEL,
                                use_auth_token=huggingface_token,
                                cache_dir=cache_dir
                            )
                        except TypeError:
                            # Probeer met 'token' parameter als 'use_auth_token' niet werkt
                            pipeline = Pipeline.from_pretrained(
                                DIARIZATION_ALTERNATIVE_MODEL,
                                token=huggingface_token,
                                cache_dir=cache_dir
                            )
                            
                        model_to_use = DIARIZATION_ALTERNATIVE_MODEL
                        logging.info(f"Alternatief model succesvol geladen.")
                        
                        # Save the alternative model locally
                        try:
                            logging.info(f"Alternatief model opslaan in lokale cache: {alternative_model_path}")
                            # Nieuwere versies gebruiken save_pretrained in plaats van to_disk
                            if hasattr(pipeline, 'to_disk'):
                                pipeline.to_disk(alternative_model_path)
                            elif hasattr(pipeline, 'save_pretrained'):
                                pipeline.save_pretrained(alternative_model_path)
                            else:
                                logging.warning("Kon alternatief model niet opslaan: geen geschikte opslagmethode gevonden")
                                # Sla het model niet op, maar ga verder met verwerking
                            logging.info(f"Alternatief model succesvol opgeslagen")
                        except Exception as save_error:
                            logging.warning(f"Kon alternatief model niet opslaan naar schijf: {str(save_error)}")
                            # Dit is niet kritiek, ga verder met verwerking
                    except Exception as alt_error:
                        alt_error_message = str(alt_error)
                        logging.error(f"Kon alternatief model niet laden: {alt_error_message}")
                        
                        if "unauthorized" in error_message.lower() or "access token" in error_message.lower() or "gated" in error_message.lower():
                            logging.error(f"Toegangsfout voor diarization modellen. Controleer:")
                            logging.error(f"1. Of je HUGGINGFACE_TOKEN geldig is")
                            logging.error(f"2. Bezoek https://hf.co/{DIARIZATION_MODEL} en accepteer de gebruiksvoorwaarden")
                            logging.error(f"3. Bezoek ook https://hf.co/{DIARIZATION_ALTERNATIVE_MODEL} en accepteer die voorwaarden")
                            logging.error(f"4. Zorg dat je account toegang heeft tot deze modellen")
                        else:
                            logging.error(f"Fout bij laden van diarization modellen. Primaire fout: {error_message}")
                            logging.error(f"Alternatieve fout: {alt_error_message}")
                        return None
        
        if not pipeline:
            logging.error("Kon geen diarization pipeline initialiseren.")
            return None
            
        # Verplaats pipeline naar CUDA als beschikbaar
        try:
            import torch
            if torch.cuda.is_available():
                logging.info("CUDA beschikbaar, verplaatsen van diarization pipeline naar GPU")
                print("GPU versnelling inschakelen voor diarization...")
                pipeline.to(torch.device("cuda"))
                print(f"Diarization pipeline verplaatst naar: {torch.cuda.get_device_name(0)}")
            else:
                logging.info("CUDA niet beschikbaar, diarization wordt op CPU uitgevoerd")
                print("GPU niet beschikbaar, diarization wordt op CPU uitgevoerd")
        except Exception as e:
            logging.warning(f"Kon pipeline niet naar GPU verplaatsen: {str(e)}. Gebruik CPU.")
            print("Kon GPU versnelling niet inschakelen, diarization wordt op CPU uitgevoerd")
            
        # Run diarization with configured parameters
        logging.info(f"Diarization uitvoeren met model {model_to_use}, min_speakers={min_speakers}, max_speakers={max_speakers}")
        
        # Voeg extra logging toe voor beter voortgangsinzicht
        start_time = time.time()
        print(f"\nStart speaker diarization op {os.path.basename(audio_file)}...")
        print(f"Dit kan enkele minuten duren afhankelijk van de lengte van het audiobestand.\n")
        
        # Voorbereiden van audio om problemen met ongelijke tensorsegmenten te voorkomen
        try:
            import torchaudio  # ensure torchaudio is available for audio processing
            print("Audio voorbereiden voor diarization...")
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # Zorg dat we stereo naar mono converteren indien nodig
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Bereken totale lengte and zorg dat deze deelbaar is door 16000 (10ms window)
            # om ongelijke tensor-maten te voorkomen
            length = waveform.size(1)
            target_length = ((length // 16000) + 1) * 16000
            
            # Padding toevoegen als de lengte niet deelbaar is door 16000
            if length < target_length:
                padding = torch.zeros((1, target_length - length))
                waveform = torch.cat([waveform, padding], dim=1)
                
            # Sla tijdelijk op als wav voor betere compatibiliteit
            temp_audio_file = audio_file + ".temp.wav"
            torchaudio.save(temp_audio_file, waveform, sample_rate)
            
            # Gebruik het bewerkte audio bestand
            diarization_audio_file = temp_audio_file
            print(f"Audio voorbereid: originele lengte={length}, nieuwe lengte={waveform.size(1)}")
        except Exception as prep_error:
            logging.warning(f"Kon audio niet voorbereiden: {str(prep_error)}, gebruiken origineel bestand")
            diarization_audio_file = audio_file
            
        # Voer diarisatie uit
        try:
            diarization = pipeline(
                diarization_audio_file,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Verwijder tijdelijk bestand indien aangemaakt
            if diarization_audio_file != audio_file and os.path.exists(diarization_audio_file):
                try:
                    os.remove(diarization_audio_file)
                except:
                    pass
                    
        except Exception as diar_error:
            # Als er nog steeds een fout is, probeer fallback methode
            logging.warning(f"Diarization fout: {str(diar_error)}, probeer fallback methode")
            print("Eerste poging mislukt, probeer alternatieve methode...")
            
            # Als tijdelijk bestand bestaat, verwijder het
            if diarization_audio_file != audio_file and os.path.exists(diarization_audio_file):
                try:
                    os.remove(diarization_audio_file)
                except:
                    pass
                    
            # Probeer met een eenvoudigere configuratie
            diarization = pipeline(
                audio_file,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                segmentation_batch_size=1  # Kleinere batch size
            )
        
        # Bereken en toon verwerkingstijd
        elapsed_time = time.time() - start_time
        print(f"\nDiarisatie voltooid in {elapsed_time:.1f} seconden.")
        
        # Convert to list of segments
        print("Verwerken van diarisatie resultaten...")
        segments = []
        
        # Maak een lijst van alle tracks en toon voortgang met tqdm
        all_tracks = list(diarization.itertracks(yield_label=True))
        unique_speakers = set()
        
        for turn, _, speaker in tqdm(all_tracks, desc="Segmenten verwerken", unit="segment"):
            segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
            unique_speakers.add(speaker)
        
        logging.info(f"Diarization voltooid met {len(segments)} segmenten en {len(unique_speakers)} sprekers")
        print(f"Diarisatie geïdentificeerd: {len(unique_speakers)} unieke sprekers in {len(segments)} segmenten")
        return segments
        
    except Exception as e:
        logging.error(f"Fout bij uitvoeren speaker diarization: {str(e)}")
        return None

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
            check_file_size(output_file, MAX_FILE_SIZE_KB)
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
    else:
        main(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
            force=args.force, skip_vocabulary=args.skip_vocabulary,
            use_diarization=args.diarize if args.diarize else None if not args.no_diarize else False,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers,
            diarization_model=args.diarization_model)
