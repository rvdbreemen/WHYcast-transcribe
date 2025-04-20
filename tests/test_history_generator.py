import pytest

from whycast_transcribe.postprocess.history_generator import generate_history

class Dummy:
    pass

@pytest.fixture(autouse=True)
def dummy_mocks(monkeypatch):
    # Stub read_prompt_file
    monkeypatch.setattr('whycast_transcribe.postprocess.history_generator.read_prompt_file', lambda _: 'prompt')
    # Stub process_with_openai
    monkeypatch.setattr('whycast_transcribe.postprocess.history_generator.process_with_openai', lambda text, prompt, model, max_tokens=None: f'history: {text}')
    return Dummy

def test_generate_history_no_prompt(monkeypatch):
    monkeypatch.setattr('whycast_transcribe.postprocess.history_generator.read_prompt_file', lambda _: None)
    result = generate_history('cleaned')
    assert result is None

def test_generate_history_success():
    result = generate_history('cleaned text')
    assert result == 'history: cleaned text'