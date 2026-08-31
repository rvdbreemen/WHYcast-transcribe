"""Reading the speaker mapping out of the analysis model's prose (ADR-004).

ADR-004 splits judgement from application: the model decides who SPEAKER_02 is,
code applies that decision. This parser is the seam between the two, and it was
the weakest part of it - tuned to one model's formatting, with a last-resort
regex that scanned the whole document.

Both fixtures in tests/data/speaker_analysis/ are real answers, captured on
2026-08-30 from the models named in the filenames, for the same transcript
(podcasts/episode_1_merged.txt). They disagree about who is who; that is not
this parser's problem and not what these tests check. What they check is that
whatever the model decided arrives intact.

Two failures motivated this file, both measured:

* ``gpt-5.6-sol`` returned all seven labels in a fenced block under a markdown
  heading ``## FINAL MAPPING FOR TRANSCRIPT``. The parser returned six: the
  heading had no colon so the section match missed, and the fallback regex
  required ``SPEAKER_\\d+``, which SPEAKER_UNKNOWN is not.
* On another run of the same input the fallback regex picked a sentence out of
  the surrounding commentary and handed it back as a speaker's name.
"""

from pathlib import Path

import pytest

from whycast.pipeline.speakers import parse_speaker_mapping_from_analysis as parse

FIXTURES = Path(__file__).parent / "data" / "speaker_analysis"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestTheOldModelsFormatKeepsWorking:
    """o4-mini writes per-speaker sections ending in '- Final Label:'."""

    @pytest.fixture
    def mapping(self):
        return parse(fixture("o4mini_episode_1.txt"))

    def test_all_seven_labels_survive(self, mapping):
        assert set(mapping) == {
            "[SPEAKER_00]", "[SPEAKER_01]", "[SPEAKER_02]", "[SPEAKER_03]",
            "[SPEAKER_04]", "[SPEAKER_05]", "[SPEAKER_UNKNOWN]",
        }

    def test_the_names_are_what_the_model_said(self, mapping):
        assert mapping["[SPEAKER_01]"] == "Ad"
        assert mapping["[SPEAKER_02]"] == "Dani"
        assert mapping["[SPEAKER_03]"] == "Odd"
        assert mapping["[SPEAKER_05]"] == "Dave"


class TestTheNewModelsFormatIsReadCorrectly:
    """gpt-5.6-sol writes a fenced block under a markdown heading, no colon."""

    @pytest.fixture
    def mapping(self):
        return parse(fixture("gpt56sol_episode_1.txt"))

    def test_speaker_unknown_is_not_dropped(self, mapping):
        """The regression: 45 of 155 segments in this episode carry that label."""
        assert "[SPEAKER_UNKNOWN]" in mapping

    def test_all_seven_labels_survive(self, mapping):
        assert set(mapping) == {
            "[SPEAKER_00]", "[SPEAKER_01]", "[SPEAKER_02]", "[SPEAKER_03]",
            "[SPEAKER_04]", "[SPEAKER_05]", "[SPEAKER_UNKNOWN]",
        }

    def test_the_names_are_what_the_model_said(self, mapping):
        assert mapping["[SPEAKER_00]"] == "Ad"
        assert mapping["[SPEAKER_01]"] == "Nancy"
        assert mapping["[SPEAKER_02]"] == "Dani"
        assert mapping["[SPEAKER_05]"] == "Dave"

    def test_no_label_is_a_sentence(self, mapping):
        """Every value must be short enough to be a name or a role."""
        for label, name in mapping.items():
            assert len(name) <= 40, f"{label} kreeg geen naam maar proza: {name!r}"


class TestImplausibleLabelsAreRejected:
    """Commentary around the mapping must never become someone's name."""

    def test_a_sentence_is_not_accepted_as_a_name(self):
        text = (
            "## FINAL MAPPING FOR TRANSCRIPT\n"
            "```text\n"
            "SPEAKER_00 -> Nancy\n"
            "SPEAKER_01 -> clause I'm Ad should be manually changed to Ad, and "
            "SPEAKER_UNKNOWN should ideally be replaced locally.\n"
            "```\n"
        )
        mapping = parse(text)
        assert mapping.get("[SPEAKER_00]") == "Nancy"
        assert "[SPEAKER_01]" not in mapping

    def test_prose_outside_the_mapping_block_is_ignored(self):
        text = (
            "In the discussion SPEAKER_03 -> sounds like it could be almost anyone here.\n"
            "\n"
            "## FINAL MAPPING FOR TRANSCRIPT\n"
            "SPEAKER_00 -> Nancy\n"
            "SPEAKER_03 -> Ad\n"
        )
        mapping = parse(text)
        assert mapping["[SPEAKER_03]"] == "Ad"

    def test_markdown_emphasis_is_stripped_from_a_name(self):
        text = "## FINAL MAPPING FOR TRANSCRIPT\nSPEAKER_00 -> **Nancy**\n"
        assert parse(text)["[SPEAKER_00]"] == "Nancy"


class TestBothArrowSpellings:
    @pytest.mark.parametrize("arrow", ["->", "→"])
    def test_ascii_and_unicode_arrows_both_parse(self, arrow):
        text = f"## FINAL MAPPING FOR TRANSCRIPT\nSPEAKER_00 {arrow} Nancy\nSPEAKER_UNKNOWN {arrow} Host\n"
        mapping = parse(text)
        assert mapping["[SPEAKER_00]"] == "Nancy"
        assert mapping["[SPEAKER_UNKNOWN]"] == "Host"


class TestNothingToParse:
    def test_empty_input_gives_an_empty_mapping(self):
        assert parse("") == {}

    def test_text_without_a_mapping_gives_an_empty_mapping(self):
        assert parse("The audio was too noisy to tell the speakers apart.") == {}
