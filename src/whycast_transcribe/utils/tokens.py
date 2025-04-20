import math
from typing import List
from whycast_transcribe.config import CHARS_PER_TOKEN


def estimate_token_count(text: str) -> int:
    """
    Estimate number of tokens in text based on average chars per token.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def split_into_chunks(text: str, max_tokens: int = None, max_char_per_chunk: int = None) -> List[str]:
    """
    Split text into chunks that do not exceed max_tokens or max_char_per_chunk.
    If max_tokens provided, derive max_char_per_chunk.
    """
    if max_tokens:
        max_chars = max_tokens * CHARS_PER_TOKEN
    elif max_char_per_chunk:
        max_chars = max_char_per_chunk
    else:
        return [text]

    paragraphs = text.split('\n\n')
    chunks: List[str] = []
    buffer = ''
    for para in paragraphs:
        para_text = para.strip()
        if not para_text:
            continue
        if len(buffer) + len(para_text) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{para_text}" if buffer else para_text
        else:
            if buffer:
                chunks.append(buffer)
            buffer = ''
            if len(para_text) <= max_chars:
                buffer = para_text
            else:
                # split large paragraph
                for i in range(0, len(para_text), max_chars):
                    chunks.append(para_text[i:i+max_chars])
    if buffer:
        chunks.append(buffer)
    return chunks