"""
Configuration settings for the transcription system.
"""
import os
import logging

# Version information
VERSION = "0.1.1"

# Model configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

# OpenAI configuration
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1")  # Default model for general tasks
OPENAI_LARGE_CONTEXT_MODEL = os.environ.get("OPENAI_LARGE_CONTEXT_MODEL", "gpt-4.1")  # Model for potentially long inputs (summary, blog)
OPENAI_HISTORY_MODEL = os.environ.get("OPENAI_HISTORY_MODEL", "gpt-4.1")  # Model specifically for history extraction
OPENAI_SPEAKER_ASSIGNMENT_MODEL = os.environ.get("OPENAI_SPEAKER_ASSIGNMENT_MODEL", "gpt-4.1")  # Model for speaker assignment
TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "8000"))  # Increased for longer summaries
# Max tokens to send to OpenAI (considering model's max context - completion tokens)
MAX_INPUT_TOKENS = int(os.environ.get("OPENAI_MAX_INPUT_TOKENS", "60000"))  # Increased to handle 200kB
# Number of tokens to use for estimating text length (OpenAI uses ~4 chars per token on average)
CHARS_PER_TOKEN = int(os.environ.get("OPENAI_CHARS_PER_TOKEN", "4"))
# Maximum file size to process without warning (in KB)
MAX_FILE_SIZE_KB = 500  # Maximum file size in KB to process without warning

# Advanced summarization settings
USE_RECURSIVE_SUMMARIZATION = os.environ.get("USE_RECURSIVE_SUMMARIZATION", "True").lower() in ("true", "1", "yes")
MAX_CHUNK_SIZE = int(os.environ.get("MAX_CHUNK_SIZE", "40000"))  # Maximum size of each chunk for recursive summarization
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "1000"))  # Overlap between chunks to maintain context

# Speaker diarization settings
USE_SPEAKER_DIARIZATION = os.environ.get("USE_SPEAKER_DIARIZATION", "True").lower() in ("true", "1", "yes")
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
# Alternative model if the primary one is not available or fails
DIARIZATION_ALTERNATIVE_MODEL = os.environ.get("DIARIZATION_ALTERNATIVE_MODEL", "pyannote/segmentation-3.0")
DIARIZATION_MIN_SPEAKERS = int(os.environ.get("DIARIZATION_MIN_SPEAKERS", "1"))
DIARIZATION_MAX_SPEAKERS = int(os.environ.get("DIARIZATION_MAX_SPEAKERS", "10"))

# Prompt file paths - using absolute paths to ensure they're found regardless of working directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Update paths to point to the prompts subdirectory with correct filenames
PROMPT_CLEANUP_FILE = os.path.join(base_dir, "prompts", "cleanup_prompt.txt")
PROMPT_SUMMARY_FILE = os.path.join(base_dir, "prompts", "summary_prompt.txt")
PROMPT_BLOG_FILE = os.path.join(base_dir, "prompts", "blog_prompt.txt")
PROMPT_BLOG_ALT1_FILE = os.path.join(base_dir, "prompts", "blog_alt1_prompt.txt")
PROMPT_HISTORY_EXTRACT_FILE = os.path.join(base_dir, "prompts", "history_extract_prompt.txt")
PROMPT_SPEAKER_ASSIGN_FILE = os.path.join(base_dir, "prompts", "speaker_assignment_prompt.txt")  # Correct prompt filename

# Custom vocabulary settings
USE_CUSTOM_VOCABULARY = True  # Set to False to disable
VOCABULARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.json")
