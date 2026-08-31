"""The per-model half of an OpenAI payload (ADR-003).

These tests exist because the previous spelling guessed the parameter contract
from the model name's first letter::

    is_o_series_model = model_name.startswith("o") and not model_name.startswith("gpt")

That put ``gpt-5.6-luna`` on the legacy side, so every call to it sent
``max_tokens`` and a ``temperature``, and the API answered 400 twice over. The
contract below was measured against the live API on 2026-08-30, not inferred:

    max_tokens            -> 400, "Use 'max_completion_tokens' instead"
    temperature=0.7       -> 400, "Only the default (1) value is supported"
    reasoning_effort      -> none | low | medium | high | xhigh
"""

import pytest

from whycast.pipeline.llm import model_params


class TestTokenParameterName:
    """Legacy chat models take ``max_tokens``; everything else does not."""

    @pytest.mark.parametrize("model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-3.5-turbo"])
    def test_legacy_chat_models_take_max_tokens(self, model):
        assert model_params(model, 100) == {"max_tokens": 100}

    @pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.2", "o4-mini", "o3"])
    def test_modern_models_take_max_completion_tokens(self, model):
        assert model_params(model, 100)["max_completion_tokens"] == 100
        assert "max_tokens" not in model_params(model, 100)

    def test_an_unknown_model_lands_on_the_modern_side(self):
        """The list names exceptions, so a model nobody has heard of yet is modern.

        This is the direction that survives: a wrong guess here is a 400 that
        names the parameter to use, where the old default silently produced the
        legacy spelling for every future model.
        """
        assert "max_completion_tokens" in model_params("gpt-9-whatever", 100)


class TestTemperatureIsNeverSent:
    """No model gets a temperature any more; the new ones reject anything but 1."""

    @pytest.mark.parametrize("model", ["gpt-4.1", "gpt-5.6-luna", "gpt-5.6-sol", "o4-mini"])
    def test_no_temperature_in_the_payload(self, model):
        assert "temperature" not in model_params(model, 100, "high")


class TestReasoningEffort:
    def test_passed_through_for_modern_models(self):
        assert model_params("gpt-5.6-sol", 100, "high")["reasoning_effort"] == "high"

    def test_dropped_for_legacy_models_that_would_reject_it(self):
        assert "reasoning_effort" not in model_params("gpt-4.1", 100, "high")

    @pytest.mark.parametrize("effort", [None, ""])
    def test_omitted_when_unset_so_the_api_default_applies(self, effort):
        assert "reasoning_effort" not in model_params("gpt-5.6-luna", 100, effort)


class TestConfiguredDefaultsMatchTheContract:
    """The shipped defaults must be callable with the shipped code."""

    def test_every_default_model_produces_a_payload_the_api_accepts(self):
        from whycast import config

        for model in (
            config.OPENAI_MODEL,
            config.OPENAI_LARGE_CONTEXT_MODEL,
            config.OPENAI_HISTORY_MODEL,
            config.OPENAI_SPEAKER_MODEL,
        ):
            params = model_params(model, 1000, config.OPENAI_REASONING_EFFORT)
            assert "temperature" not in params
            assert ("max_tokens" in params) ^ ("max_completion_tokens" in params)

    def test_reasoning_effort_defaults_are_values_the_api_accepts(self):
        from whycast import config

        accepted = {"none", "low", "medium", "high", "xhigh"}
        assert config.OPENAI_REASONING_EFFORT in accepted
        assert config.OPENAI_SPEAKER_REASONING_EFFORT in accepted
