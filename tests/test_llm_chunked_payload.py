"""The chunked path must send the same per-model contract as the single-shot path.

``process_large_text_in_chunks`` builds its own payloads, twice: once per chunk
and once for the final consistency pass. Before this test it carried its own
copy of the o-series guess, so a fix on the single-shot path could drift away
from it silently.

These use a fake client rather than the API on purpose. What can go wrong here
is the shape of the request - ``max_tokens`` or a ``temperature`` reaching a
model that answers 400 to both - and that is visible in the payload without
spending anything. The live contract those assertions encode was measured
against the API on 2026-08-30; see tests/test_llm_model_params.py.
"""

import pytest

from whycast.pipeline import llm


class FakeCompletion:
    def __init__(self, content="processed chunk."):
        self.choices = [type("C", (), {
            "message": type("M", (), {"content": content})(),
            "finish_reason": "stop",
        })()]
        self.usage = None


class RecordingClient:
    """Captures every payload instead of sending it."""

    def __init__(self):
        self.payloads = []
        outer = self

        class _Completions:
            def create(self, **params):
                outer.payloads.append(params)
                return FakeCompletion()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


@pytest.fixture
def chunked_payloads(monkeypatch):
    """Run the chunked path over text long enough to produce more than one chunk.

    max_chunk_size is forced small so the split is cheap; the real
    ``split_into_chunks`` still does the work, so this exercises the production
    path and not a stand-in for it.
    """
    monkeypatch.setattr(llm, "MAX_CHUNK_SIZE", 400)
    client = RecordingClient()
    text = ("Dit is een zin over de podcast. " * 120)
    llm.process_large_text_in_chunks(text, "Clean this up.", "gpt-5.6-luna", client,
                                     reasoning_effort="high")
    assert len(client.payloads) > 1, "opzet mislukt: er is maar een chunk gemaakt"
    return client.payloads


class TestChunkedPathSendsTheModernContract:
    def test_never_sends_the_legacy_token_parameter(self, chunked_payloads):
        assert all("max_tokens" not in p for p in chunked_payloads)

    def test_always_sends_max_completion_tokens(self, chunked_payloads):
        assert all("max_completion_tokens" in p for p in chunked_payloads)

    def test_never_sends_a_temperature(self, chunked_payloads):
        """The new models reject any temperature other than 1, on every call."""
        assert all("temperature" not in p for p in chunked_payloads)

    def test_passes_the_reasoning_effort_through_to_every_call(self, chunked_payloads):
        assert all(p.get("reasoning_effort") == "high" for p in chunked_payloads)


class TestChunkedPathStillHonoursLegacyModels:
    def test_a_legacy_model_gets_max_tokens_and_no_reasoning_effort(self, monkeypatch):
        monkeypatch.setattr(llm, "MAX_CHUNK_SIZE", 400)
        client = RecordingClient()
        llm.process_large_text_in_chunks("Dit is een zin over de podcast. " * 120,
                                         "Clean this up.", "gpt-4.1", client,
                                         reasoning_effort="high")
        assert client.payloads
        for p in client.payloads:
            assert "max_tokens" in p
            assert "max_completion_tokens" not in p
            assert "reasoning_effort" not in p
            assert "temperature" not in p
