"""Splitting a transcript into chunks (ADR-005).

Every chunk this function returns becomes one paid OpenAI call, in both
``summarize_large_transcript`` and ``process_large_text_in_chunks``. The count
is therefore part of the contract, not an implementation detail.

It used to end the loop with::

    start = max(start + 1, end - overlap)

and nothing else. Once ``end`` reached the end of the text, ``start`` jumped to
``len(text) - overlap``, which is still short of the end, so the loop kept
going: ``end`` stayed put, ``end - overlap`` stayed below ``start``, and
``start + 1`` won every round. That appended exactly ``overlap`` more chunks,
1000 down to 1, each one a real API call. A 100 KB transcript cost 1002 calls
where it needed 2.
"""

import pytest

from whycast.pipeline.llm import split_into_chunks


class TestChunkCount:
    """The number of chunks is the number of API calls. Guard it."""

    def test_the_regression_that_cost_a_thousand_calls(self):
        text = "y" * 100_000
        chunks = split_into_chunks(text, max_chunk_size=80_000, overlap=1_000)
        assert len(chunks) == 2, f"verwacht 2 chunks, kreeg er {len(chunks)}"

    def test_no_degenerate_tail_of_shrinking_chunks(self):
        """The old bug's fingerprint: a run of chunks each one character shorter."""
        chunks = split_into_chunks("y" * 100_000, max_chunk_size=80_000, overlap=1_000)
        lengths = [len(c) for c in chunks]
        assert lengths == sorted(set(lengths), reverse=True) or len(lengths) <= 2
        assert not any(a - b == 1 for a, b in zip(lengths, lengths[1:])), lengths[:20]

    @pytest.mark.parametrize("size,expected", [
        (5_000, 1),      # ruim onder de chunkgrootte
        (10_000, 1),     # precies de chunkgrootte
        (10_001, 2),     # een teken erboven
        (25_000, 3),
    ])
    def test_counts_across_the_boundary(self, size, expected):
        chunks = split_into_chunks("y" * size, max_chunk_size=10_000, overlap=1_000)
        assert len(chunks) == expected


class TestChunksCoverTheWholeText:
    def test_concatenating_without_the_overlap_reproduces_the_text(self):
        text = "y" * 45_000
        chunks = split_into_chunks(text, max_chunk_size=10_000, overlap=1_000)
        rebuilt = chunks[0] + "".join(c[1_000:] for c in chunks[1:])
        assert rebuilt == text

    def test_consecutive_chunks_overlap(self):
        text = "y" * 45_000
        chunks = split_into_chunks(text, max_chunk_size=10_000, overlap=1_000)
        assert len(chunks) > 1
        for a, b in zip(chunks, chunks[1:]):
            assert a[-1_000:] == b[:1_000]


class TestNoChunkIsAScrap:
    """A chunk shorter than the overlap carries no new text worth a call."""

    @pytest.mark.parametrize("size", [10_001, 12_000, 19_999, 20_000, 20_001, 100_000])
    def test_every_chunk_is_longer_than_the_overlap(self, size):
        chunks = split_into_chunks("y" * size, max_chunk_size=10_000, overlap=1_000)
        assert all(len(c) > 1_000 for c in chunks), [len(c) for c in chunks]


class TestBreakPointsStillWork:
    """The split prefers paragraph and sentence boundaries; that must survive."""

    def test_splits_on_a_paragraph_break(self):
        first = "a" * 7_000
        second = "b" * 7_000
        chunks = split_into_chunks(f"{first}\n\n{second}", max_chunk_size=10_000, overlap=100)
        assert chunks[0].endswith("\n\n")
        assert chunks[0].startswith("a") and chunks[0].rstrip().endswith("a")

    def test_splits_on_a_sentence_break(self):
        text = ("a" * 6_000) + ". " + ("b" * 7_000)
        chunks = split_into_chunks(text, max_chunk_size=10_000, overlap=100)
        assert chunks[0].endswith(". ")
