"""
Configuration settings for the WHYcast Transcribe application.
"""

import os

# Version information
VERSION = "0.1.0"

# Model settings
MODEL_SIZE = "large-v3"  # Options: tiny, base, small, medium, large-v1, large-v2, large-v3
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'models--Systran--faster-whisper-large-v3')
DEVICE = "cuda"          # Options: cpu, cuda
COMPUTE_TYPE = "float16" # Options: float16, int8
BEAM_SIZE = 5            # Number of beams for beam search

# OpenAI API settings
OPENAI_MODEL = "gpt-4o"
OPENAI_LARGE_CONTEXT_MODEL = "gpt-4-turbo"
TEMPERATURE = 0.2
MAX_TOKENS = 4000
MAX_INPUT_TOKENS = 16000  # Maximum tokens for input to OpenAI
CHARS_PER_TOKEN = 4       # Rough estimate for token counting

# Text chunking settings
MAX_CHUNK_SIZE = 20000    # Characters per chunk for large text processing
CHUNK_OVERLAP = 1000      # Character overlap between chunks

# File settings
MAX_FILE_SIZE_KB = 500000  # Maximum recommended file size in KB

# Prompt file paths
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts')
PROMPT_CLEANUP_FILE = os.path.join(PROMPTS_DIR, 'cleanup_prompt.txt')
PROMPT_SUMMARY_FILE = os.path.join(PROMPTS_DIR, 'summary_prompt.txt')
PROMPT_BLOG_FILE = os.path.join(PROMPTS_DIR, 'blog_prompt.txt')
PROMPT_BLOG_ALT1_FILE = os.path.join(PROMPTS_DIR, 'blog_prompt_alt1.txt')

# Vocabulary settings
VOCABULARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocabulary.json')
USE_CUSTOM_VOCABULARY = True
