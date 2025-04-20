import math
from whycast_transcribe.utils.tokens import estimate_token_count, split_into_chunks


def test_estimate_token_count():
    text = 'x' * 9
    # CHARS_PER_TOKEN = 4 by default => ceil(9/4) = 3
    assert estimate_token_count(text) == math.ceil(9 / 4)


def test_split_into_chunks_by_max_tokens():
    text = 'a' * 9
    # max_tokens=2 => max_chars=2*4=8
    chunks = split_into_chunks(text, max_tokens=2)
    assert chunks == ['a' * 8, 'a']


def test_split_into_chunks_by_max_char_per_chunk():
    text = 'para1\n\npara2\n\npara3'
    # max_char_per_chunk=7 => splits on paragraphs larger than 7
    chunks = split_into_chunks(text, max_char_per_chunk=7)
    assert chunks == ['para1', 'para2', 'para3']
