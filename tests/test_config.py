import os
import importlib
import pytest


def test_default_config(monkeypatch):
    # Ensure no environment overrides
    monkeypatch.delenv('VERSION', raising=False)
    monkeypatch.delenv('WHISPER_MODEL_SIZE', raising=False)
    import whycast_transcribe.config as config
    importlib.reload(config)
    assert config.VERSION == '0.1.1'
    assert config.MODEL_SIZE == 'large-v3'
    assert config.DEVICE == 'cuda'
    assert isinstance(config.BEAM_SIZE, int)


def test_env_overrides(monkeypatch):
    monkeypatch.setenv('VERSION', '2.0.0')
    monkeypatch.setenv('WHISPER_MODEL_SIZE', 'small')
    import whycast_transcribe.config as config
    importlib.reload(config)
    assert config.VERSION == '2.0.0'
    assert config.MODEL_SIZE == 'small'
