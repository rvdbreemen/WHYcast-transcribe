import logging
import pytest

from whycast_transcribe.postprocess.cleanup import cleanup_transcript

class Dummy:
    pass

@pytest.fixture(autouse=True)
def dummy_mocks(monkeypatch):
    # Provide a dummy prompt
    monkeypatch.setattr('whycast_transcribe.postprocess.cleanup.read_prompt_file', lambda _: 'prompt')
    # Return transcript uppercased as "cleaned"
    monkeypatch.setattr('whycast_transcribe.postprocess.cleanup.process_with_openai', lambda text, prompt, model, max_tokens: text.upper())
    # Always choose base model
    monkeypatch.setattr('whycast_transcribe.postprocess.cleanup.choose_appropriate_model', lambda text: 'model')
    return Dummy

def test_cleanup_no_prompt(monkeypatch):
    monkeypatch.setattr('whycast_transcribe.postprocess.cleanup.read_prompt_file', lambda _: None)
    assert cleanup_transcript('abc') == 'abc'

def test_cleanup_success(monkeypatch):
    # Using default dummy_mocks, cleanup will uppercase
    result = cleanup_transcript('hello world')
    assert result == 'HELLO WORLD'

def test_cleanup_short_truncation(monkeypatch):
    # process returns very short
    monkeypatch.setattr('whycast_transcribe.postprocess.cleanup.process_with_openai', lambda t, prompt, model, max_tokens: 'x')
    result = cleanup_transcript('longtext')
    assert result == 'longtext'
