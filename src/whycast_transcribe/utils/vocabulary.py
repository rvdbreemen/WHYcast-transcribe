import json
import os
import logging

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


def apply_vocabulary_corrections(text: str) -> str:
    """
    Replace occurrences of misheard words based on vocabulary mapping.
    """
    vocab = load_vocabulary()
    for wrong, correct in vocab.items():
        text = text.replace(wrong, correct)
    return text