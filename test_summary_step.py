import pytest
import transcribe


def setup_dummy(monkeypatch, *, estimated_tokens=10):
    monkeypatch.setattr(transcribe, "read_prompt_file", lambda path: "PROMPT")
    monkeypatch.setattr(transcribe, "choose_appropriate_model", lambda text: "model")
    monkeypatch.setattr(transcribe, "estimate_token_count", lambda text: estimated_tokens)


def test_summary_small(monkeypatch):
    setup_dummy(monkeypatch, estimated_tokens=10)

    called = {}
    def fake_process(text, prompt, model, max_tokens=transcribe.MAX_TOKENS):
        called['args'] = (text, prompt, model)
        return 'summary'
    monkeypatch.setattr(transcribe, 'process_with_openai', fake_process)
    monkeypatch.setattr(transcribe, 'summarize_large_transcript', lambda t, p: 'large')

    result = transcribe.summary_step('cleaned')
    assert result == 'summary'
    assert called['args'][0] == 'cleaned'


def test_summary_recursive(monkeypatch):
    setup_dummy(monkeypatch, estimated_tokens=transcribe.MAX_INPUT_TOKENS + 1)
    monkeypatch.setattr(transcribe, 'USE_RECURSIVE_SUMMARIZATION', True)
    monkeypatch.setattr(transcribe, 'process_with_openai', lambda *a, **k: 'short')

    called = {}
    def fake_recursive(text, prompt):
        called['args'] = (text, prompt)
        return 'long'
    monkeypatch.setattr(transcribe, 'summarize_large_transcript', fake_recursive)

    result = transcribe.summary_step('cleaned')
    assert result == 'long'
    assert called['args'][0] == 'cleaned'

