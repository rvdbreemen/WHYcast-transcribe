import pytest

from whycast_transcribe.postprocess.blog_generator import generate_blog, generate_blog_alt

class Dummy:
    pass

@pytest.fixture(autouse=True)
def dummy_mocks(monkeypatch):
    # Stub read_prompt_file for primary and alt prompts
    monkeypatch.setattr('whycast_transcribe.postprocess.blog_generator.read_prompt_file', lambda path: 'prompt' if 'blog' in path else None)
    # Stub process_with_openai
    monkeypatch.setattr('whycast_transcribe.postprocess.blog_generator.process_with_openai', lambda text, prompt, model, max_tokens=None: f'blog: {text}')
    return Dummy

def test_generate_blog_no_prompt(monkeypatch):
    monkeypatch.setattr('whycast_transcribe.postprocess.blog_generator.read_prompt_file', lambda _: None)
    result = generate_blog('clean', 'summary')
    assert result is None

def test_generate_blog_success():
    result = generate_blog('clean', 'summary')
    assert result == 'blog: Summary:\nsummary\nTranscript:\nclean'

def test_generate_blog_alt_no_prompt(monkeypatch):
    monkeypatch.setattr('whycast_transcribe.postprocess.blog_generator.read_prompt_file', lambda _: None)
    result = generate_blog_alt('clean', 'summary')
    assert result is None

def test_generate_blog_alt_success():
    result = generate_blog_alt('clean', 'summary')
    assert result == 'blog: Summary:\nsummary\nTranscript:\nclean'
