"""
Tests for :mod:`whycast.episodes` - the tolerant podcast directory scanner
(ADR-008 / TASK-002 phase 1).

Two kinds of test live here, and the split is deliberate:

* **Synthetic** (most of the file). A ``tmp_path`` directory is built from
  scratch that reproduces every trap the real ``podcasts/`` holds: the
  ``episode_1`` / ``episode_10`` prefix collision, a base name with a space,
  chained suffixes, two spellings of "transcript", an episode whose audio was
  deleted, and junk that must stay junk. Exact counts and exact kinds are
  asserted here, because this directory cannot change under the test.

* **Real** (:class:`TestRealPodcastDirectory`). One test scans the operator's
  actual ``podcasts/`` and asserts only invariants that must hold for *any*
  directory. No count, no name, nothing that a new episode would break.

A note on one requested real-directory invariant, "no episode has a duplicate
``(kind, fmt)`` pair": that is not an invariant of this scanner, it is
explicitly *not* the design. ``Episode.artifact``'s docstring names the live
case - ``ep29_speaker_assignment.txt`` next to ``ep29_ts_speaker_assignment.txt``
- and the real directory holds six such pairs across ``ep29`` and
``episode_10``. The scanner is supposed to keep both files and let
``Episode.artifact`` pick the newest.

What *is* an invariant, and what :mod:`webui.db` actually depends on, is that no
**path** is ever indexed twice: ``artifacts.path`` is ``TEXT PRIMARY KEY``, so a
repeated path would make ``rescan()`` raise ``IntegrityError``. That is the
invariant asserted below, together with an explicit test pinning the intended
duplicate behaviour.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from whycast.episodes import (  # noqa: E402
    ARTIFACT_FORMATS,
    ARTIFACT_KINDS,
    Episode,
    ScanResult,
    parse_episode_number,
    scan_podcasts,
)
from whycast.errors import ConfigurationError  # noqa: E402

REAL_PODCAST_DIR = REPO_ROOT / "podcasts"


# ---------------------------------------------------------------------------
# Synthetic directory
# ---------------------------------------------------------------------------

#: Every file in the synthetic directory, with the content written into it.
#: Content is only ever used to make a failure readable; the scanner reads
#: nothing but directory metadata.
#:
#: The set is chosen so that each entry is a trap the real directory contains:
#:
#:   episode_1 / episode_10        prefix collision (both audio, both artifacts)
#:   "Episode 28"                  a base name with a space in it
#:   episode_13_ts_speaker_...     a chained suffix; the outermost names the kind
#:   episode_7.txt + _transcript   two spellings of the same kind
#:   episode_52*                   audio-less: the mp3 was deleted, artifacts stayed
#:   Whycast33 / ep29 / whycast-*  the four number-parsing styles
#:   E99 Thumbnail.jpg, promo.mp4  junk that must never become an artifact
#:   notes.dat, LICENSE            unknown extension, and no extension at all
#:   episode_1.foo_summary.txt     dot-separated rest: belongs to episode_1, but
#:                                 is not a recognised artifact of it
#:   episode_13.mp3.ffmpeg...      the real directory's ffmpeg leftover shape
#:   extra_recordings/             a subdirectory; the scan is shallow
SYNTHETIC_FILES = {
    # -- episode_1 vs episode_10: the collision the whole design turns on ----
    "episode_1.mp3": "audio-1",
    "episode_1.txt": "bare transcript of one",
    "episode_1_summary.txt": "summary of one",
    "episode_10.mp3": "audio-10",
    "episode_10_blog.txt": "blog of ten",
    "episode_10_blog.html": "<p>blog of ten</p>",
    "episode_10_ts.txt": "timestamped ten",
    # -- a base name with a space -------------------------------------------
    "Episode 28.mp3": "audio-28",
    "Episode 28_summary.txt": "summary of twenty-eight",
    # -- chained suffix, plus the same (kind, fmt) slot filled twice ---------
    "episode_13.mp3": "audio-13",
    "episode_13_speaker_assignment.txt": "old assignment",
    "episode_13_ts_speaker_assignment.txt": "new assignment",
    "episode_13.mp3.ffmpeg.16k_diarization.txt": "ffmpeg leftover",
    # -- two spellings of transcript ----------------------------------------
    "episode_7.mp3": "audio-7",
    "episode_7.txt": "bare transcript of seven",
    "episode_7_transcript.txt": "suffixed transcript of seven",
    # -- number-parsing styles ----------------------------------------------
    "ep29.mp3": "audio-29",
    "ep29_cleaned.txt": "cleaned twenty-nine",
    "Whycast33.mp3": "audio-33",
    "Whycast33_history.wiki": "history thirty-three",
    "whycast-episode-13.mp3": "audio-13-alt",
    "whycast-episode-13_blog_alt1.html": "<p>alt blog</p>",
    # -- audio-less episode: artifacts survived, the mp3 did not ------------
    "episode_52.txt": "bare transcript of fifty-two",
    "episode_52_ts.txt": "timestamped fifty-two",
    "episode_52_summary.html": "<p>summary of fifty-two</p>",
    # -- junk ---------------------------------------------------------------
    "E99 Thumbnail.jpg": "not an artifact",
    "promo.mp4": "not an artifact",
    "notes.dat": "not an artifact",
    "LICENSE": "not an artifact",
    # -- belongs to episode_1, but is not a recognised artifact of it -------
    "episode_1.foo_summary.txt": "unrecognised",
}

#: Subdirectories created in the synthetic directory. The scan is shallow, so
#: these must be reported rather than descended into.
SYNTHETIC_DIRS = ("extra_recordings",)

#: Expected episode base names, in the order the scanner promises: by number,
#: then by lowercase base name. Two episodes carry number 13, which is the
#: point - `number` is not a key, `base_name` is.
EXPECTED_BASE_NAMES = [
    "episode_1",
    "episode_7",
    "episode_10",
    "episode_13",
    "whycast-episode-13",
    "Episode 28",
    "ep29",
    "Whycast33",
    "episode_52",
]

#: Files that belong to no episode. Names, not paths, for readability.
EXPECTED_UNMATCHED_NAMES = {
    "E99 Thumbnail.jpg",
    "LICENSE",
    "episode_1.foo_summary.txt",
    "episode_13.mp3.ffmpeg.16k_diarization.txt",
    "extra_recordings",
    "notes.dat",
    "promo.mp4",
}


@pytest.fixture
def synthetic_dir(tmp_path):
    """A podcast directory holding every naming trap, and nothing else.

    Returns the directory path. Mtimes are set explicitly for the two files
    that share a ``(kind, fmt)`` slot, so the "newest wins" tie-break is tested
    against a known answer rather than against whatever order the filesystem
    happened to stamp.
    """
    root = tmp_path / "podcasts"
    root.mkdir()
    for name, content in SYNTHETIC_FILES.items():
        (root / name).write_text(content, encoding="utf-8")
    for name in SYNTHETIC_DIRS:
        (root / name).mkdir()

    # The superseded file is older; the _ts_ one is what the current pipeline
    # writes. Mirrors ep29 on the real disk, where the gap is 36 days.
    _set_mtime(root / "episode_13_speaker_assignment.txt", 1_700_000_000)
    _set_mtime(root / "episode_13_ts_speaker_assignment.txt", 1_700_086_400)
    return root


def _set_mtime(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def _episode(result: ScanResult, base_name: str) -> Episode:
    """The one episode with this base name, or a readable failure."""
    matches = [e for e in result.episodes if e.base_name == base_name]
    assert len(matches) == 1, (
        f"expected exactly one episode named {base_name!r}, got "
        f"{[e.base_name for e in result.episodes]}"
    )
    return matches[0]


def _names(paths) -> set:
    return {os.path.basename(p) for p in paths}


def _artifact_names(episode: Episode, kind: str) -> set:
    return {os.path.basename(a.path) for a in episode.artifacts if a.kind == kind}


# ---------------------------------------------------------------------------
# The collision that motivates the whole matching rule
# ---------------------------------------------------------------------------


class TestPrefixCollision:
    """``episode_1`` and ``episode_10`` both exist; neither may swallow the other."""

    def test_episode_10_blog_does_not_attach_to_episode_1(self, synthetic_dir):
        result = scan_podcasts(str(synthetic_dir))
        one = _episode(result, "episode_1")
        ten = _episode(result, "episode_10")

        assert _artifact_names(ten, "blog") == {
            "episode_10_blog.txt",
            "episode_10_blog.html",
        }
        assert not one.has("blog"), (
            "episode_10_blog.* attached to episode_1: the longest-base rule is broken"
        )
        assert "episode_10_ts.txt" in _artifact_names(ten, "ts")
        assert not one.has("ts")

    def test_episode_1_keeps_its_own_artifacts(self, synthetic_dir):
        result = scan_podcasts(str(synthetic_dir))
        one = _episode(result, "episode_1")
        assert _artifact_names(one, "transcript") == {"episode_1.txt"}
        assert _artifact_names(one, "summary") == {"episode_1_summary.txt"}

    def test_no_phantom_episode_from_a_dot_separated_rest(self, synthetic_dir):
        """``episode_1.foo_summary.txt`` belongs to episode_1 and mints nothing.

        Without ``.`` counting as a separator this file would fall through to
        the audio-less branch, which peels ``_summary`` and would mint a
        phantom episode ``episode_1.foo`` next to the real one.
        """
        result = scan_podcasts(str(synthetic_dir))
        assert "episode_1.foo" not in {e.base_name for e in result.episodes}
        assert "episode_1.foo_summary.txt" in _names(result.unmatched)


# ---------------------------------------------------------------------------
# Naming shapes
# ---------------------------------------------------------------------------


class TestNamingShapes:
    def test_base_name_with_a_space(self, synthetic_dir):
        result = scan_podcasts(str(synthetic_dir))
        episode = _episode(result, "Episode 28")
        assert os.path.basename(episode.audio_path) == "Episode 28.mp3"
        assert _artifact_names(episode, "summary") == {"Episode 28_summary.txt"}

    def test_chained_suffix_resolves_to_the_outermost_kind(self, synthetic_dir):
        """``<base>_ts_speaker_assignment`` is the assignment, not a ts artifact."""
        result = scan_podcasts(str(synthetic_dir))
        episode = _episode(result, "episode_13")

        assignments = _artifact_names(episode, "speaker_assignment")
        assert "episode_13_ts_speaker_assignment.txt" in assignments
        # It must NOT also be filed as a "ts" artifact, and it must NOT mint
        # an episode "episode_13_ts".
        assert not episode.has("ts")
        assert "episode_13_ts" not in {e.base_name for e in result.episodes}

    def test_both_transcript_spellings_are_transcripts(self, synthetic_dir):
        """``<base>.txt`` and ``<base>_transcript.txt`` are both kind=transcript."""
        result = scan_podcasts(str(synthetic_dir))
        episode = _episode(result, "episode_7")
        assert _artifact_names(episode, "transcript") == {
            "episode_7.txt",
            "episode_7_transcript.txt",
        }
        assert all(a.fmt == "txt" for a in episode.artifacts if a.kind == "transcript")

    def test_duplicate_kind_and_format_keeps_both_files_and_prefers_the_newest(
        self, synthetic_dir
    ):
        """Two files in one ``(kind, fmt)`` slot: both indexed, newest served.

        This is the case the "no duplicate (kind, fmt)" invariant would have
        forbidden. It is real (``ep29``, ``episode_10``) and intended: the
        older file is a superseded pipeline generation, and dropping it at scan
        time would hide it from the unmatched list too.
        """
        result = scan_podcasts(str(synthetic_dir))
        episode = _episode(result, "episode_13")

        txt_assignments = [
            a
            for a in episode.artifacts
            if a.kind == "speaker_assignment" and a.fmt == "txt"
        ]
        assert len(txt_assignments) == 2, "both files must survive the scan"
        assert len({a.path for a in txt_assignments}) == 2

        best = episode.artifact("speaker_assignment")
        assert best is not None
        assert os.path.basename(best.path) == "episode_13_ts_speaker_assignment.txt", (
            "the newest file must win; ordering by path alone would serve the "
            "superseded _speaker_assignment file forever"
        )

    def test_audio_less_episode_is_minted_from_its_artifacts(self, synthetic_dir):
        """The mp3 is gone; the artifacts still form one episode, not three strays."""
        result = scan_podcasts(str(synthetic_dir))
        episode = _episode(result, "episode_52")

        assert episode.audio_path is None
        assert episode.audio_size is None
        assert episode.number == 52
        assert _names([a.path for a in episode.artifacts]) == {
            "episode_52.txt",
            "episode_52_ts.txt",
            "episode_52_summary.html",
        }
        # The bare "<base>.txt" is visited before "<base>_ts.txt" (0x2E sorts
        # before 0x5F) and can never mint an episode on its own. It must still
        # have landed on the episode the _ts file minted.
        assert episode.has("transcript")
        assert episode.has("ts")
        assert episode.artifact("summary").fmt == "html"

    @pytest.mark.parametrize(
        "name",
        ["E99 Thumbnail.jpg", "promo.mp4", "notes.dat", "LICENSE"],
    )
    def test_junk_is_reported_unmatched_not_invented_into_an_artifact(
        self, synthetic_dir, name
    ):
        result = scan_podcasts(str(synthetic_dir))
        assert name in _names(result.unmatched)
        all_artifacts = [a.path for e in result.episodes for a in e.artifacts]
        assert name not in _names(all_artifacts)

    def test_subdirectory_is_reported_not_descended_into(self, synthetic_dir):
        (synthetic_dir / "extra_recordings" / "episode_99.mp3").write_text("x")
        result = scan_podcasts(str(synthetic_dir))
        assert "extra_recordings" in _names(result.unmatched)
        assert "episode_99" not in {e.base_name for e in result.episodes}

    def test_unrecognised_suffix_on_a_known_base_is_unmatched(self, synthetic_dir):
        """The ffmpeg leftover belongs to episode_13 but names no kind."""
        result = scan_podcasts(str(synthetic_dir))
        assert "episode_13.mp3.ffmpeg.16k_diarization.txt" in _names(result.unmatched)
        episode = _episode(result, "episode_13")
        assert "episode_13.mp3.ffmpeg.16k_diarization.txt" not in _names(
            [a.path for a in episode.artifacts]
        )


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------


class TestEpisodeNumbers:
    @pytest.mark.parametrize(
        "base_name, expected",
        [
            ("episode_13", 13),
            ("ep29", 29),
            ("Episode 28", 28),
            ("Whycast33", 33),
            ("whycast-episode-13", 13),
            ("whycast_episode_4_-_fire_", 4),
            ("episode_0", 0),
            ("episode_100", 100),
            ("Other recordings", None),
            ("", None),
        ],
    )
    def test_parse_episode_number(self, base_name, expected):
        assert parse_episode_number(base_name) == expected

    def test_numbers_are_parsed_for_every_naming_style_in_the_directory(
        self, synthetic_dir
    ):
        result = scan_podcasts(str(synthetic_dir))
        numbers = {e.base_name: e.number for e in result.episodes}
        assert numbers == {
            "episode_1": 1,
            "episode_7": 7,
            "episode_10": 10,
            "episode_13": 13,
            "whycast-episode-13": 13,
            "Episode 28": 28,
            "ep29": 29,
            "Whycast33": 33,
            "episode_52": 52,
        }

    def test_two_episodes_may_share_a_number(self, synthetic_dir):
        """``number`` is not a key. ``base_name`` is."""
        result = scan_podcasts(str(synthetic_dir))
        thirteens = [e.base_name for e in result.episodes if e.number == 13]
        assert sorted(thirteens) == ["episode_13", "whycast-episode-13"]

    def test_episodes_are_ordered_by_number_then_name(self, synthetic_dir):
        result = scan_podcasts(str(synthetic_dir))
        assert [e.base_name for e in result.episodes] == EXPECTED_BASE_NAMES


# ---------------------------------------------------------------------------
# Whole-directory guarantees
# ---------------------------------------------------------------------------


class TestScanResultGuarantees:
    def test_expected_episodes_and_unmatched(self, synthetic_dir):
        result = scan_podcasts(str(synthetic_dir))
        assert [e.base_name for e in result.episodes] == EXPECTED_BASE_NAMES
        assert _names(result.unmatched) == EXPECTED_UNMATCHED_NAMES

    def test_every_entry_is_accounted_for_exactly_once(self, synthetic_dir):
        """The module's headline guarantee: nothing in the directory vanishes."""
        result = scan_podcasts(str(synthetic_dir))
        accounted = []
        for episode in result.episodes:
            if episode.audio_path:
                accounted.append(episode.audio_path)
            accounted.extend(a.path for a in episode.artifacts)
        accounted.extend(result.unmatched)

        normalised = [os.path.normcase(os.path.abspath(p)) for p in accounted]
        assert len(normalised) == len(set(normalised)), "an entry was counted twice"

        on_disk = {
            os.path.normcase(os.path.abspath(str(synthetic_dir / name)))
            for name in os.listdir(synthetic_dir)
        }
        assert set(normalised) == on_disk

    def test_kinds_and_formats_stay_inside_the_vocabulary(self, synthetic_dir):
        result = scan_podcasts(str(synthetic_dir))
        for episode in result.episodes:
            for artifact in episode.artifacts:
                assert artifact.kind in ARTIFACT_KINDS
                assert artifact.fmt in ARTIFACT_FORMATS

    def test_episode_mtime_is_the_newest_of_audio_and_artifacts(self, synthetic_dir):
        result = scan_podcasts(str(synthetic_dir))
        for episode in result.episodes:
            newest = max(
                [os.path.getmtime(episode.audio_path)] if episode.audio_path else [0.0]
            )
            for artifact in episode.artifacts:
                newest = max(newest, artifact.mtime)
            assert episode.mtime == pytest.approx(newest)

    def test_two_scans_produce_equal_results(self, synthetic_dir):
        first = scan_podcasts(str(synthetic_dir))
        second = scan_podcasts(str(synthetic_dir))
        assert first == second
        assert [e.base_name for e in first.episodes] == [
            e.base_name for e in second.episodes
        ]
        assert first.unmatched == second.unmatched
        for a, b in zip(first.episodes, second.episodes):
            assert [x.path for x in a.artifacts] == [x.path for x in b.artifacts]

    def test_paths_are_absolute(self, synthetic_dir):
        result = scan_podcasts(str(synthetic_dir))
        for episode in result.episodes:
            if episode.audio_path:
                assert os.path.isabs(episode.audio_path)
            for artifact in episode.artifacts:
                assert os.path.isabs(artifact.path)
        for path in result.unmatched:
            assert os.path.isabs(path)

    def test_empty_directory_scans_clean(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = scan_podcasts(str(empty))
        assert result.episodes == []
        assert result.unmatched == []

    def test_missing_directory_raises_rather_than_exits(self, tmp_path):
        with pytest.raises(ConfigurationError):
            scan_podcasts(str(tmp_path / "does-not-exist"))

    def test_scan_writes_nothing(self, synthetic_dir):
        """The scanner reads directory metadata and nothing else (ADR-008)."""
        before = {
            name: os.stat(synthetic_dir / name).st_mtime
            for name in os.listdir(synthetic_dir)
        }
        scan_podcasts(str(synthetic_dir))
        after = {
            name: os.stat(synthetic_dir / name).st_mtime
            for name in os.listdir(synthetic_dir)
        }
        assert before == after


# ---------------------------------------------------------------------------
# The real directory: invariants only
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_PODCAST_DIR.is_dir(), reason="no podcasts/ directory in this checkout"
)
class TestRealPodcastDirectory:
    """Scans the operator's actual ``podcasts/``.

    Every assertion here must survive a new episode being added tomorrow, so
    nothing counts, names, or numbers anything. What is asserted is what the
    module docstring promises and what :mod:`webui.db` relies on.
    """

    @pytest.fixture(scope="class")
    def real(self):
        return scan_podcasts(str(REAL_PODCAST_DIR))

    def test_scan_succeeds_and_finds_something(self, real):
        assert real.episodes, "the real podcasts/ should hold at least one episode"

    def test_no_artifact_path_appears_under_two_episodes(self, real):
        owner = {}
        for episode in real.episodes:
            for artifact in episode.artifacts:
                key = os.path.normcase(artifact.path)
                assert key not in owner, (
                    f"{artifact.path} is claimed by both {owner.get(key)!r} and "
                    f"{episode.base_name!r}"
                )
                owner[key] = episode.base_name

    def test_no_artifact_path_is_indexed_twice(self, real):
        """``webui.db.artifacts.path`` is a PRIMARY KEY; a repeat breaks rescan.

        This replaces the "no duplicate (kind, fmt) per episode" invariant,
        which is false by design - see this module's docstring.
        """
        paths = [
            os.path.normcase(a.path) for e in real.episodes for a in e.artifacts
        ]
        assert len(paths) == len(set(paths))

    def test_repeated_kind_and_format_slots_are_always_distinct_files(self, real):
        """Where a ``(kind, fmt)`` slot repeats, it is two real files, resolvable."""
        for episode in real.episodes:
            buckets = {}
            for artifact in episode.artifacts:
                buckets.setdefault((artifact.kind, artifact.fmt), []).append(artifact)
            for (kind, fmt), artifacts in buckets.items():
                assert len({a.path for a in artifacts}) == len(artifacts)
                # Whatever the count, resolution is unambiguous: the newest
                # file wins, so a superseded generation is never served.
                chosen = episode.artifact(kind, fmt)
                assert chosen is not None
                assert chosen.mtime == max(a.mtime for a in artifacts)

    def test_every_unmatched_path_exists(self, real):
        for path in real.unmatched:
            assert os.path.exists(path), f"unmatched entry does not exist: {path}"

    def test_every_unmatched_path_is_inside_the_podcast_directory(self, real):
        root = os.path.normcase(os.path.abspath(str(REAL_PODCAST_DIR))) + os.sep
        for path in real.unmatched:
            assert os.path.normcase(os.path.abspath(path)).startswith(root)

    def test_nothing_is_accounted_for_twice(self, real):
        accounted = []
        for episode in real.episodes:
            if episode.audio_path:
                accounted.append(episode.audio_path)
            accounted.extend(a.path for a in episode.artifacts)
        accounted.extend(real.unmatched)
        normalised = [os.path.normcase(os.path.abspath(p)) for p in accounted]
        assert len(normalised) == len(set(normalised))

    def test_base_names_are_unique_case_insensitively(self, real):
        """``webui.db`` keys on ``base_name.lower()`` and refuses collisions."""
        keys = [e.base_name.lower() for e in real.episodes]
        assert len(keys) == len(set(keys))

    def test_vocabulary_is_respected(self, real):
        for episode in real.episodes:
            for artifact in episode.artifacts:
                assert artifact.kind in ARTIFACT_KINDS
                assert artifact.fmt in ARTIFACT_FORMATS
                assert artifact.size >= 0

    def test_ordering_is_the_documented_one(self, real):
        keys = [
            (e.number is None, e.number if e.number is not None else 0, e.base_name.lower())
            for e in real.episodes
        ]
        assert keys == sorted(keys)

    def test_a_second_scan_agrees_with_the_first(self, real):
        again = scan_podcasts(str(REAL_PODCAST_DIR))
        assert [e.base_name for e in again.episodes] == [
            e.base_name for e in real.episodes
        ]
        assert again.unmatched == real.unmatched


# ---------------------------------------------------------------------------
# Case, unicode and empty files
# ---------------------------------------------------------------------------


class TestCaseInsensitivity:
    """NTFS holds ``Episode_28.mp3`` next to ``episode_28_summary.txt``.

    Treating those as two episodes would mint a phantom audio-less episode
    beside the real one, which is exactly what the module docstring warns
    about. ``base_name`` keeps the casing of the audio file that won.
    """

    def test_mixed_case_artifact_joins_the_audio_episode(self, tmp_path):
        root = tmp_path / "podcasts"
        root.mkdir()
        (root / "Episode_60.mp3").write_bytes(b"audio")
        (root / "episode_60_summary.txt").write_bytes(b"summary")
        (root / "EPISODE_60_blog.html").write_bytes(b"blog")

        result = scan_podcasts(str(root))
        assert len(result.episodes) == 1
        episode = result.episodes[0]
        assert episode.base_name == "Episode_60", "the audio file's casing wins"
        assert episode.number == 60
        assert {a.kind for a in episode.artifacts} == {"summary", "blog"}
        assert result.unmatched == []

    def test_mixed_case_audio_less_episode_is_one_episode(self, tmp_path):
        root = tmp_path / "podcasts"
        root.mkdir()
        (root / "Episode_61_summary.txt").write_bytes(b"summary")
        (root / "episode_61_ts.txt").write_bytes(b"ts")

        result = scan_podcasts(str(root))
        assert len(result.episodes) == 1
        assert result.episodes[0].base_name.lower() == "episode_61"
        assert {a.kind for a in result.episodes[0].artifacts} == {"summary", "ts"}

    def test_uppercase_extension_is_still_an_artifact(self, tmp_path):
        root = tmp_path / "podcasts"
        root.mkdir()
        (root / "episode_62.MP3").write_bytes(b"audio")
        (root / "episode_62_summary.TXT").write_bytes(b"summary")

        result = scan_podcasts(str(root))
        assert len(result.episodes) == 1
        episode = result.episodes[0]
        assert episode.audio_path is not None
        assert [a.fmt for a in episode.artifacts] == ["txt"]


class TestUnusualFiles:
    def test_non_ascii_base_name(self, tmp_path):
        root = tmp_path / "podcasts"
        root.mkdir()
        (root / "aflevering_63_café.mp3").write_bytes(b"audio")
        (root / "aflevering_63_café_summary.txt").write_bytes(b"summary")

        result = scan_podcasts(str(root))
        assert len(result.episodes) == 1
        episode = result.episodes[0]
        assert episode.base_name == "aflevering_63_café"
        assert episode.number == 63
        assert episode.has("summary")

    def test_zero_byte_artifact_is_a_result_not_a_missing_file(self, tmp_path):
        """The pipeline can legitimately write an empty file."""
        root = tmp_path / "podcasts"
        root.mkdir()
        (root / "episode_64.mp3").write_bytes(b"audio")
        (root / "episode_64_summary.txt").write_bytes(b"")

        result = scan_podcasts(str(root))
        episode = result.episodes[0]
        assert episode.has("summary")
        assert episode.artifact("summary").size == 0

    def test_multiple_audio_files_for_one_base_pick_one_and_report_the_rest(
        self, tmp_path
    ):
        """``episode32.m4a`` next to ``episode32.mp3`` is live on disk."""
        root = tmp_path / "podcasts"
        root.mkdir()
        (root / "episode_65.mp3").write_bytes(b"mp3")
        (root / "episode_65.m4a").write_bytes(b"m4a")
        (root / "episode_65.wav").write_bytes(b"wav")

        result = scan_podcasts(str(root))
        assert len(result.episodes) == 1
        episode = result.episodes[0]
        assert os.path.basename(episode.audio_path) == "episode_65.mp3", (
            "mp3 is preferred over m4a over wav"
        )
        # Nothing disappears: the losers are reported, not dropped.
        assert _names(result.unmatched) == {"episode_65.m4a", "episode_65.wav"}
