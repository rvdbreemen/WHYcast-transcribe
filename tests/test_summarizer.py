import pytest

from whycast_transcribe.postprocess.summarizer import summarize_transcript

class Dummy:
    pass

@pytest.fixture(autouse=True)
def dummy_mocks(monkeypatch):
    # Default to non-recursive
    monkeypatch.setenv('USE_RECURSIVE_SUMMARIZATION', 'False')
    # Stub read_prompt_file
    monkeypatch.setattr('whycast_transcribe.postprocess.summarizer.read_prompt_file', lambda _: 'prompt')
    # Stub process_with_openai
    monkeypatch.setattr('whycast_transcribe.postprocess.summarizer.process_with_openai', lambda text, prompt, model, max_tokens=None: f'summary of {text}')
    # Stub split_into_chunks
    monkeypatch.setattr('whycast_transcribe.postprocess.summarizer.split_into_chunks', lambda text, max_chunk_size=None: [text[:5], text[5:]])
    return Dummy

def test_no_prompt(monkeypatch):
    monkeypatch.setattr('whycast_transcribe.postprocess.summarizer.read_prompt_file', lambda _: None)
    result = summarize_transcript('hello world')
    assert result is None

def test_single_pass(monkeypatch):
    # USE_RECURSIVE_SUMMARIZATION False by default
    result = summarize_transcript('hello world')
    assert result == 'summary of hello world'

def test_recursive(monkeypatch):
    monkeypatch.setenv('USE_RECURSIVE_SUMMARIZATION', 'True')
    # Summarize two chunks
    result = summarize_transcript('abcdefghij')
    # Expect summary of each chunk joined
    assert result == 'summary of abcde\n\nsummary of fghij'