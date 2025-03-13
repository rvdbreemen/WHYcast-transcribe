"""
Configuration settings for the transcription system.
"""
import os

# Version information
VERSION = "0.0.6"

# Model configuration
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

# OpenAI configuration
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")  # Updated to current model name
OPENAI_LARGE_CONTEXT_MODEL = os.environ.get("OPENAI_LARGE_CONTEXT_MODEL", "gpt-4o-2024-05-13")  # Updated to latest version with 128K context
TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "4000"))  # Increased for longer summaries
# Max tokens to send to OpenAI (considering model's max context - completion tokens)
MAX_INPUT_TOKENS = int(os.environ.get("OPENAI_MAX_INPUT_TOKENS", "50000"))  # Increased to handle 200kB
# Number of tokens to use for estimating text length (OpenAI uses ~4 chars per token on average)
CHARS_PER_TOKEN = int(os.environ.get("OPENAI_CHARS_PER_TOKEN", "4"))
# Maximum file size to process without warning (in KB)
MAX_FILE_SIZE_KB = int(os.environ.get("MAX_FILE_SIZE_KB", "250"))  # Increased to handle 200kB+ files

# Advanced summarization settings
USE_RECURSIVE_SUMMARIZATION = os.environ.get("USE_RECURSIVE_SUMMARIZATION", "True").lower() in ("true", "1", "yes")
MAX_CHUNK_SIZE = int(os.environ.get("MAX_CHUNK_SIZE", "40000"))  # Maximum size of each chunk for recursive summarization
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "1000"))  # Overlap between chunks to maintain context

# Prompt file paths - using absolute paths to ensure they're found regardless of working directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Update paths to point to the prompts subdirectory with correct filenames
PROMPT_CLEANUP_FILE = os.path.join(base_dir, "prompts", "cleanup_prompt.txt")
PROMPT_SUMMARY_FILE = os.path.join(base_dir, "prompts", "summary_prompt.txt")
PROMPT_BLOG_FILE = os.path.join(base_dir, "prompts", "blog_prompt.txt")

# Custom vocabulary settings
USE_CUSTOM_VOCABULARY = os.environ.get("USE_CUSTOM_VOCABULARY", "True").lower() in ("true", "1", "yes")
VOCABULARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.json")
