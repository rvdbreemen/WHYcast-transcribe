#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Content generation functionality for WHYcast-transcribe.

This consolidated module provides functions for generating various content
from podcast transcripts:
- Summaries (with support for large/recursive summarization)
- Blog posts (with multiple variants)
- History lesson extractions
- Speaker assignment

The module presents a unified interface while maintaining clear organization
of the different content generation capabilities.
"""

import os
import logging
import glob
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm

# Import configuration
from config import (
    OPENAI_MODEL,
    OPENAI_LARGE_CONTEXT_MODEL,
    OPENAI_HISTORY_MODEL,
    OPENAI_SPEAKER_MODEL,
    MAX_TOKENS,
    MAX_INPUT_TOKENS,
    USE_RECURSIVE_SUMMARIZATION,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP,
    PROMPT_CLEANUP_FILE,
    PROMPT_SUMMARY_FILE,
    PROMPT_BLOG_FILE,
    PROMPT_BLOG_ALT1_FILE,
    PROMPT_HISTORY_EXTRACT_FILE
)

# Import utilities from other modules
from transcribe import process_with_openai, choose_appropriate_model, convert_markdown_to_html, convert_markdown_to_wiki
from utils.text_processing import split_into_chunks, estimate_token_count
from utils.file_helpers import read_prompt_file

# Final cleanup: This module now only contains utility functions and process_speaker_assignment_workflow.
# All main workflow logic is in transcribe.py.

def process_speaker_assignment_workflow(transcript: str, base_path: str) -> None:
    """
    Process transcript with speaker assignment prompt using o3-mini, chunking if needed.
    Output: <base>_with_speakers.txt, .html, .md
    """
    from config import OPENAI_MODEL, MAX_INPUT_TOKENS
    from utils.text_processing import split_into_chunks
    from transcribe import process_with_openai
    from utils.file_helpers import read_prompt_file
    from transcribe import convert_markdown_to_html
    import os

    # Check for diarization tags in transcript
    if not any(f"SPEAKER_" in line for line in transcript.splitlines()):
        raise ValueError("Transcript does not contain diarization speaker tags. Speaker assignment prompt requires diarized transcript.")

    prompt = read_prompt_file(os.path.join("prompts", "speaker_assignment_prompt.txt"))
    if not prompt:
        raise RuntimeError("Speaker assignment prompt file not found.")

    # Chunk transcript if needed
    chunks = split_into_chunks(transcript, max_chunk_size=MAX_INPUT_TOKENS * 4)  # 4 chars/token
    results = []
    for chunk in chunks:
        result = process_with_openai(chunk, prompt, OPENAI_SPEAKER_MODEL, max_tokens=MAX_TOKENS)
        results.append(result)
    full_result = "\n\n".join(results)

    # Write .txt (markdown)
    txt_path = f"{base_path}_speaker_assignment.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_result)
    # Write .html
    html_path = f"{base_path}_speaker_assignment.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(convert_markdown_to_html(full_result))
    # Write .wiki
    wiki_path = f"{base_path}_speaker_assignment.wiki"
    with open(wiki_path, "w", encoding="utf-8") as f:
        f.write(convert_markdown_to_wiki(full_result))
    return txt_path, html_path, wiki_path