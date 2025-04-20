import re
import logging
from typing import List
from whycast_transcribe.config import MAX_INPUT_TOKENS, CHARS_PER_TOKEN


def estimate_token_count(text: str) -> int:
    """
    Estimate number of tokens in text based on average chars per token.
    """
    return len(text) // CHARS_PER_TOKEN


def split_into_chunks(text: str, max_chunk_size: int = MAX_INPUT_TOKENS, overlap: int = 1000) -> List[str]:
    """
    Split text into chunks of specified maximum size with overlap.
    """
    if len(text) <= max_chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        # Attempt to break at paragraph
        para_break = text.rfind('\n\n', start, end)
        if para_break != -1 and para_break > start + max_chunk_size // 2:
            end = para_break + 2
        chunks.append(text[start:end])
        start = max(start + 1, end - overlap)
    return chunks