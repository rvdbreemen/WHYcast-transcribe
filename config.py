#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration for WHYcast Transcribe
"""

import os

# Version of the application
VERSION = "0.0.5"

# Whisper model configuration
MODEL_SIZE = "large-v3"  # Options: tiny, small, medium, large-v1, large-v2, large-v3
DEVICE = "cuda"          # Options: cuda, cpu, auto
COMPUTE_TYPE = "float16"  # Options: float16, float32, int8
BEAM_SIZE = 5

# Phi-4 model configuration
PHI_MODEL_NAME = "microsoft/Phi-4-multi"  # Phi-4 model from HuggingFace
PHI_MAX_NEW_TOKENS = 2048        # Maximum number of tokens to generate
PHI_TEMPERATURE = 0.7            # Temperature for generation

# Token management
MAX_INPUT_TOKENS = 16000         # Maximum input tokens for model
CHARS_PER_TOKEN = 4              # Approximate characters per token

# File settings
PROMPT_FILE = "summary_prompt_blog.txt"       # Path to the prompt file
MAX_FILE_SIZE_KB = 15000         # Maximum file size in KB for warning

# Chunking settings for large transcripts
USE_RECURSIVE_SUMMARIZATION = True
MAX_CHUNK_SIZE = 10000           # Characters per chunk
CHUNK_OVERLAP = 500              # Overlap between chunks

# Vocabulary settings
VOCABULARY_FILE = "vocabulary.json"
USE_CUSTOM_VOCABULARY = True     # Whether to use custom vocabulary corrections

# LMStudio API Configuration
USE_LMSTUDIO_API = True  # Set to False to always try loading Phi model directly
LMSTUDIO_URL = "http://localhost:1234/v1"  # Base URL for LMStudio API
LMSTUDIO_API_KEY = ""  # API key if required (often not needed for local server)
LMSTUDIO_MODEL = "phi-4"  # The model name to use with LMStudio API (often just "model" for locally loaded models)
