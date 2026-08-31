"""Missing optional dependencies must fail with a sentence, not a shim's ghost.

ADR-008 says library code raises and the worker translates. Two call sites used
an optional shim without checking its flag, so a missing package surfaced as
``'NoneType' object is not callable`` or ``'NoneType' object has no attribute
'parse'`` - neither of which names the package or the fix.

The first attempt at this guard was worse than the bug. It raised inside
``process_with_openai``'s outer ``try``, whose handler formats ``call_id``, a
local bound several statements later. The call died with
``UnboundLocalError: cannot access local variable 'call_id'``, and even with
that fixed the handler ends in ``return None``, so the carefully worded message
would have been logged and swallowed. Both are covered below.
"""

import pytest

from whycast.errors import ConfigurationError
from whycast.pipeline import feed, llm


class TestTheOpenAIGuard:
    def test_a_missing_openai_raises_configuration_error(self, monkeypatch):
        monkeypatch.setattr(llm, "openai_available", False)
        with pytest.raises(ConfigurationError) as excinfo:
            llm.process_with_openai("some text", "some prompt", "gpt-5.6-luna")
        assert "openai" in str(excinfo.value).lower()

    def test_the_message_names_the_install_command(self, monkeypatch):
        monkeypatch.setattr(llm, "openai_available", False)
        with pytest.raises(ConfigurationError) as excinfo:
            llm.process_with_openai("t", "p", "gpt-5.6-luna")
        assert "pip install openai" in str(excinfo.value)

    def test_it_is_not_swallowed_into_a_none_return(self, monkeypatch):
        """The outer try ends in `return None`; the guard must sit outside it."""
        monkeypatch.setattr(llm, "openai_available", False)
        with pytest.raises(ConfigurationError):
            llm.process_with_openai("t", "p", "gpt-5.6-luna")

    def test_it_does_not_die_on_an_unbound_call_id(self, monkeypatch):
        """The regression: the handler formatted call_id before it was bound."""
        monkeypatch.setattr(llm, "openai_available", False)
        with pytest.raises(Exception) as excinfo:
            llm.process_with_openai("t", "p", "gpt-5.6-luna")
        assert not isinstance(excinfo.value, UnboundLocalError), (
            "an early failure must not crash the error handler itself"
        )

    def test_an_early_failure_after_the_guard_still_reports_itself(self, monkeypatch):
        """call_id is bound before the try, so ensure_api_key failing is readable.

        This path was broken before the guard existed too: no OPENAI_API_KEY
        raised ValueError, whose handler then died on the same unbound name.
        """
        monkeypatch.setattr(llm, "openai_available", True)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(Exception) as excinfo:
            llm.process_with_openai("t", "p", "gpt-5.6-luna")
        assert not isinstance(excinfo.value, UnboundLocalError)


class TestTheFeedparserGuard:
    def test_a_missing_feedparser_raises_configuration_error(self, monkeypatch):
        monkeypatch.setattr(feed, "feedparser_available", False)
        with pytest.raises(ConfigurationError) as excinfo:
            feed.podcast_fetching_workflow("https://example.com/feed.xml", "out")
        assert "feedparser" in str(excinfo.value).lower()

    def test_the_bulk_download_path_is_guarded_too(self, monkeypatch):
        """Two call sites reach feedparser.parse; both need the check."""
        monkeypatch.setattr(feed, "feedparser_available", False)
        target = getattr(feed, "download_all_episodes", None)
        if target is None:
            pytest.skip("bulk download entry point not present under that name")
        with pytest.raises(ConfigurationError):
            target("https://example.com/feed.xml", "out")

    def test_the_message_names_the_install_command(self, monkeypatch):
        monkeypatch.setattr(feed, "feedparser_available", False)
        with pytest.raises(ConfigurationError) as excinfo:
            feed.podcast_fetching_workflow("https://example.com/feed.xml", "out")
        assert "pip install feedparser" in str(excinfo.value)


class TestTheErrorReachesTheCli:
    def test_configuration_error_is_not_a_pipeline_error(self):
        """Which is why transcribe.py needs its own except arm for it.

        A sibling, not a subclass: ``except PipelineError`` does not catch it,
        so before the fix the CLI answered a missing package with a traceback.
        """
        from whycast.errors import PipelineError, WhycastError

        assert issubclass(ConfigurationError, WhycastError)
        assert not issubclass(ConfigurationError, PipelineError)

    def test_the_cli_handles_it(self):
        import re
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "transcribe.py").read_text(
            encoding="utf-8"
        )
        assert re.search(r"except\s+ConfigurationError", source), (
            "transcribe.py must translate ConfigurationError into an exit code "
            "instead of letting it surface as a traceback"
        )
