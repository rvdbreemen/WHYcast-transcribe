"""The diarization settings must reach the pyannote pipeline (ADR-002).

Five settings lived in ``config.py``, were listed in the web UI config viewer,
and were read by no production code:

    USE_SPEAKER_DIARIZATION, DIARIZATION_MODEL, DIARIZATION_ALTERNATIVE_MODEL,
    DIARIZATION_MIN_SPEAKERS, DIARIZATION_MAX_SPEAKERS

``diarize_audio`` hardcoded ``'pyannote/speaker-diarization-3.1'`` and called the
pipeline with no speaker bounds. So a user could point DIARIZATION_MODEL at
another model, watch the web UI confirm the new value, and change nothing at
all; and the bound that would have stopped episode 1 being split into six
labels for four people was never passed.

These tests use a fake Pipeline. What matters is the arguments that leave our
code, not what pyannote does with them.
"""

import sys
import types

import pytest

from whycast.pipeline import diarization


@pytest.fixture
def fake_pyannote(monkeypatch):
    """Intercept pyannote.audio.Pipeline and record how it is used."""
    calls = {"from_pretrained": [], "invocations": []}

    class FakePipeline:
        @staticmethod
        def from_pretrained(model, use_auth_token=None):
            calls["from_pretrained"].append(model)
            if getattr(FakePipeline, "fail_for", None) == model:
                raise RuntimeError(f"simulated load failure for {model}")
            if getattr(FakePipeline, "none_for", None) == model:
                # Not hypothetical: pyannote returns None rather than raising
                # when a repository is missing, private or gated
                # (pyannote/audio/core/pipeline.py:107-121).
                return None
            return FakePipeline()

        def to(self, device):
            return self

        def __call__(self, audio, **kwargs):
            calls["invocations"].append(kwargs)
            return _EmptyAnnotation()

    class _EmptyAnnotation:
        def itertracks(self, yield_label=False):
            return iter(())

    module = types.ModuleType("pyannote.audio")
    module.Pipeline = FakePipeline
    parent = types.ModuleType("pyannote")
    parent.audio = module
    monkeypatch.setitem(sys.modules, "pyannote", parent)
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)
    calls["cls"] = FakePipeline
    return calls


@pytest.fixture
def waveform():
    import torch

    return torch.zeros(1, 16000), 16000


class TestTheModelComesFromConfig:
    def test_the_configured_model_is_the_one_loaded(self, monkeypatch, fake_pyannote, waveform):
        monkeypatch.setattr(diarization, "DIARIZATION_MODEL", "someone/other-pipeline-1.0")
        monkeypatch.setattr(diarization, "USE_SPEAKER_DIARIZATION", True)
        diarization.diarize_audio(waveform=waveform[0], sample_rate=waveform[1], hf_token="x")
        assert fake_pyannote["from_pretrained"] == ["someone/other-pipeline-1.0"]

    def test_the_alternative_model_is_tried_when_the_primary_fails(
        self, monkeypatch, fake_pyannote, waveform
    ):
        """ADR-002 names this fallback as its mitigation for pyannote drift."""
        monkeypatch.setattr(diarization, "DIARIZATION_MODEL", "primary/model")
        monkeypatch.setattr(diarization, "DIARIZATION_ALTERNATIVE_MODEL", "backup/model")
        monkeypatch.setattr(diarization, "USE_SPEAKER_DIARIZATION", True)
        fake_pyannote["cls"].fail_for = "primary/model"
        try:
            diarization.diarize_audio(waveform=waveform[0], sample_rate=waveform[1], hf_token="x")
        finally:
            fake_pyannote["cls"].fail_for = None
        assert fake_pyannote["from_pretrained"] == ["primary/model", "backup/model"]

    def test_the_fallback_also_fires_when_pyannote_returns_none(
        self, monkeypatch, fake_pyannote, waveform
    ):
        """The failure ADR-002 actually names does not raise at all.

        Pipeline.from_pretrained catches RepositoryNotFoundError - and so
        GatedRepoError - prints a hint and returns None. A fallback that only
        caught exceptions would never fire for the gated case it exists for,
        which is exactly what the first version of this code did.
        """
        monkeypatch.setattr(diarization, "DIARIZATION_MODEL", "gated/model")
        monkeypatch.setattr(diarization, "DIARIZATION_ALTERNATIVE_MODEL", "backup/model")
        monkeypatch.setattr(diarization, "USE_SPEAKER_DIARIZATION", True)
        fake_pyannote["cls"].none_for = "gated/model"
        try:
            diarization.diarize_audio(waveform=waveform[0], sample_rate=waveform[1], hf_token="x")
        finally:
            fake_pyannote["cls"].none_for = None
        assert fake_pyannote["from_pretrained"] == ["gated/model", "backup/model"]

    def test_a_none_return_with_no_fallback_names_the_gated_repository(
        self, monkeypatch, fake_pyannote, waveform, caplog
    ):
        """`result is None` alone proves nothing: it is None either way.

        diarize_audio converts every failure to None for its callers, so the
        only thing that distinguishes the fix from the bug is WHAT gets logged.
        Without the _load helper the log reads "'NoneType' object has no
        attribute 'to'"; with it, the message names the repository and the URL
        where the conditions are accepted.
        """
        monkeypatch.setattr(diarization, "DIARIZATION_MODEL", "gated/model")
        monkeypatch.setattr(diarization, "DIARIZATION_ALTERNATIVE_MODEL", "")
        monkeypatch.setattr(diarization, "USE_SPEAKER_DIARIZATION", True)
        fake_pyannote["cls"].none_for = "gated/model"
        try:
            with caplog.at_level("ERROR"):
                result = diarization.diarize_audio(
                    waveform=waveform[0], sample_rate=waveform[1], hf_token="x"
                )
        finally:
            fake_pyannote["cls"].none_for = None

        assert result is None
        assert "gated/model" in caplog.text, (
            "the failure must name the repository that could not be loaded"
        )
        assert "NoneType" not in caplog.text, (
            "a NoneType error means the None return was not caught"
        )


class TestSpeakerBoundsReachThePipeline:
    def test_max_speakers_is_passed(self, monkeypatch, fake_pyannote, waveform):
        monkeypatch.setattr(diarization, "USE_SPEAKER_DIARIZATION", True)
        monkeypatch.setattr(diarization, "DIARIZATION_MIN_SPEAKERS", 1)
        monkeypatch.setattr(diarization, "DIARIZATION_MAX_SPEAKERS", 4)
        diarization.diarize_audio(waveform=waveform[0], sample_rate=waveform[1], hf_token="x")
        assert fake_pyannote["invocations"] == [{"max_speakers": 4}]

    def test_min_speakers_is_passed_when_it_constrains_anything(
        self, monkeypatch, fake_pyannote, waveform
    ):
        monkeypatch.setattr(diarization, "USE_SPEAKER_DIARIZATION", True)
        monkeypatch.setattr(diarization, "DIARIZATION_MIN_SPEAKERS", 3)
        monkeypatch.setattr(diarization, "DIARIZATION_MAX_SPEAKERS", 5)
        diarization.diarize_audio(waveform=waveform[0], sample_rate=waveform[1], hf_token="x")
        assert fake_pyannote["invocations"] == [{"min_speakers": 3, "max_speakers": 5}]

    def test_a_min_of_one_is_not_passed_because_it_constrains_nothing(
        self, monkeypatch, fake_pyannote, waveform
    ):
        monkeypatch.setattr(diarization, "USE_SPEAKER_DIARIZATION", True)
        monkeypatch.setattr(diarization, "DIARIZATION_MIN_SPEAKERS", 1)
        monkeypatch.setattr(diarization, "DIARIZATION_MAX_SPEAKERS", 0)
        diarization.diarize_audio(waveform=waveform[0], sample_rate=waveform[1], hf_token="x")
        assert fake_pyannote["invocations"] == [{}]


class TestTheSwitchActuallySwitches:
    def test_disabling_diarization_skips_it_entirely(self, monkeypatch, fake_pyannote, waveform):
        monkeypatch.setattr(diarization, "USE_SPEAKER_DIARIZATION", False)
        result = diarization.diarize_audio(
            waveform=waveform[0], sample_rate=waveform[1], hf_token="x"
        )
        assert result is None
        assert fake_pyannote["from_pretrained"] == [], "pyannote must not even be loaded"
