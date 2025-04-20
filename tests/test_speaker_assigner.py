import pytest

from whycast_transcribe.postprocess.speaker_assigner import assign_speakers

class Dummy:
    pass

@pytest.fixture(autouse=True)
def dummy_mocks(monkeypatch):
    # Stub read_prompt_file
    monkeypatch.setattr('whycast_transcribe.postprocess.speaker_assigner.read_prompt_file', lambda _: 'prompt')
    # Stub split_into_chunks
    monkeypatch.setattr('whycast_transcribe.postprocess.speaker_assigner.split_into_chunks', lambda text, max_chunk_size: [text])
    # Stub process_with_openai
    monkeypatch.setattr('whycast_transcribe.postprocess.speaker_assigner.process_with_openai', lambda chunk, prompt, model, max_tokens=None: f'assigned {chunk}')
    return Dummy

def test_no_diarization_tags():
    # No speaker tags => skip
    result = assign_speakers('no tags here')
    assert result is None

def test_assign_speakers_success():
    transcript = 'SPEAKER_1 Hello'
    result = assign_speakers(transcript)
    assert 'assigned SPEAKER_1 Hello' in result