import json
import os
import logging
import re

from whycast_transcribe.config import BASE_DIR

# Path to vocabulary corrections file
VOCAB_FILE = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'vocabulary.json'))


def load_vocabulary():
    """
    Load vocabulary corrections from JSON file.
    """
    try:
        with open(VOCAB_FILE, 'r', encoding='utf-8') as vf:
            return json.load(vf)
    except Exception as e:
        logging.error(f"Failed to load vocabulary file: {e}")
        return {}


def apply_vocabulary_corrections(text: str, vocab: dict) -> str:
    """
    Replace occurrences of misheard words/phrases based on vocabulary mapping.
    Only replace full words or exact phrases, not partial matches.
    """
    if not vocab:
        return text
    # Sort keys by length descending to avoid partial overlaps
    for wrong in sorted(vocab, key=len, reverse=True):
        correct = vocab[wrong]
        # Use word boundaries for single words, or exact match for phrases
        if ' ' in wrong:
            # Phrase: match as is
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        else:
            # Single word: match as full word
            pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        text = pattern.sub(correct, text)
    return text