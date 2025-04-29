#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text processing utilities for WHYcast-transcribe.

This module provides functions for:
- Estimating token counts for LLM input
- Splitting and truncating transcripts for token limits
- Model selection based on transcript length
- Loading and applying custom vocabulary corrections

Intended for use in all modules that require text chunking, token estimation, or vocabulary handling.
"""

import os
import re
import logging
import json
from typing import List, Dict, Optional


def estimate_token_count(text: str, chars_per_token: int = 4) -> int:
    """
    Estimate the number of tokens in a text.
    
    Args:
        text: The text to estimate
        chars_per_token: Number of characters per token (default: 4)
        
    Returns:
        Estimated number of tokens
    """
    return len(text) // chars_per_token


def truncate_transcript(transcript: str, max_tokens: int, chars_per_token: int = 4) -> str:
    """
    Truncate a transcript to fit within token limits.
    
    Args:
        transcript: The transcript text
        max_tokens: Maximum token count allowed
        chars_per_token: Number of characters per token (default: 4)
        
    Returns:
        Truncated transcript
    """
    estimated_tokens = estimate_token_count(transcript, chars_per_token)
    if estimated_tokens <= max_tokens:
        return transcript
        
    # If transcript is too long, keep the first part and last part
    chars_to_keep = max_tokens * chars_per_token
    first_part_size = chars_to_keep // 2
    last_part_size = chars_to_keep - first_part_size - 100  # Leave room for ellipsis message
    
    first_part = transcript[:first_part_size]
    last_part = transcript[-last_part_size:]
    
    return first_part + "\n\n[...transcript truncated due to length...]\n\n" + last_part


def choose_appropriate_model(transcript: str, 
                             default_model: str = "gpt-4o", 
                             large_context_model: str = "gpt-4o-2024-05-13",
                             max_input_tokens: int = 50000,
                             chars_per_token: int = 4) -> str:
    """
    Choose the appropriate model based on the transcript length.
    
    Args:
        transcript: The transcript text
        default_model: Default model to use for shorter texts
        large_context_model: Model to use for very long texts
        max_input_tokens: Maximum input tokens for the default model
        chars_per_token: Number of characters per token (default: 4)
        
    Returns:
        Model name to use
    """
    estimated_tokens = estimate_token_count(transcript, chars_per_token)
    if estimated_tokens > max_input_tokens:
        logging.info(f"Transcript is very long (~{estimated_tokens} tokens), using large context model")
        return large_context_model
    return default_model


def split_into_chunks(text: str, max_chunk_size: int = 40000, overlap: int = 1000) -> List[str]:
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


def load_vocabulary_mappings(vocab_path: str = None) -> dict:
    """
    Load custom vocabulary mappings from a JSON file.
    Args:
        vocab_path: Path to the vocabulary JSON file (optional)
    Returns:
        Dictionary of vocabulary mappings
    """
    if vocab_path is None:
        vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vocabulary.json')
    if not os.path.exists(vocab_path):
        return {}
    try:
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab
    except Exception:
        return {}


def apply_vocabulary_corrections(text: str, vocabulary: dict) -> str:
    """
    Apply vocabulary corrections to a text string.
    Args:
        text: The text to correct
        vocabulary: Dictionary of vocabulary mappings
    Returns:
        Corrected text
    """
    if not vocabulary:
        return text
    for wrong, correct in vocabulary.items():
        text = text.replace(wrong, correct)
    return text