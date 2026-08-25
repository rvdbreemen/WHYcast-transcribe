"""
Episode scanner for the WHYcast web UI (ADR-008).

The filesystem is the source of truth. This module reads a podcast directory
and reconstructs which files belong to which episode, so the web UI (and the
SQLite index it rebuilds from disk) never has to guess.

Naming in ``podcasts/`` is inconsistent after years of runs: ``episode_13``,
``ep29``, ``Episode 28``, ``Whycast33``, ``whycast-episode-13``,
``whycast_episode_4_-_fire_``. Matching therefore works in two passes:

1. Every audio file (``.mp3`` / ``.m4a`` / ``.wav``) defines a canonical base:
   its own stem. This is the only trustworthy source of episode identity.
2. A file whose stem carries a known artifact suffix but matches no base yet
   mints an audio-less episode - the mp3 may have been deleted or moved.
3. Every remaining file is attached to the *longest* base it matches, where a
   match means ``stem == base + rest`` and ``rest`` is empty or starts with
   ``_`` or ``.``.

Both halves of rule 3 matter. ``episode_1`` and ``episode_10`` both exist:
without longest-base-first, ``episode_10_blog.txt`` would attach to
``episode_1``; without the boundary check, ``episode_100_blog.txt`` would too.

Rule 2 is a separate pass rather than part of rule 3 because a bare
``<base>.<fmt>`` transcript can join an episode but can never mint one, and
``.`` sorts before ``_``: in one pass ``episode_52.txt`` is decided before
``episode_52_ts.txt`` has minted ``episode_52``.

Matching is case-insensitive. The directory is NTFS and holds ``Episode_28.mp3``
next to ``episode_28_summary.txt``; treating those as different episodes would
mint a phantom audio-less episode 28 beside the real one. ``base_name`` keeps
the casing of the audio file that won.

Guarantees relied on by callers:

* Pure and side-effect free: it reads directory metadata and nothing else.
  No writes, no network, no GPU, no model loading. Importable at any time.
* Every entry in the directory ends up exactly once in either an episode
  (as audio or artifact) or in ``ScanResult.unmatched``. Nothing vanishes.
* Deterministic ordering everywhere, so the index and the tests are stable.
* ``base_name`` is the unique key of an episode; ``number`` is **not**. Fifteen
  numbers on this disk are carried by two episodes each (``ep29`` next to
  ``Episode_29``, ``episode_33`` next to ``Whycast33``), which is naming
  history, not a bug. Anything that indexes episodes - a route, a database
  key, a dict - must key on ``base_name``.
* ``ScanResult.unmatched`` holds directory paths as well as file paths: the
  scan is shallow, and a subdirectory is reported rather than descended into.
  Callers that stat these entries must expect a directory.

Errors are raised, never terminated on (ADR-008 Decision Contract); progress
goes through :mod:`whycast.events`, never through the console.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from whycast.errors import ConfigurationError
from whycast.events import emit

__all__ = [
    "ARTIFACT_KINDS",
    "ARTIFACT_FORMATS",
    "AUDIO_EXTENSIONS",
    "Artifact",
    "Episode",
    "ScanResult",
    "belongs_to_base",
    "scan_podcasts",
]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Artifact kinds the pipeline produces, in display order.
ARTIFACT_KINDS = [
    "transcript",
    "ts",
    "cleaned",
    "summary",
    "blog",
    "blog_alt1",
    "history",
    "speaker_assignment",
    "analysis",
    "merged",
]

#: Artifact file formats, in preference order (best first).
ARTIFACT_FORMATS = ("txt", "html", "wiki", "md")

#: Extensions that make a file the audio of an episode.
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav")

_KIND_ORDER: Dict[str, int] = {k: i for i, k in enumerate(ARTIFACT_KINDS)}
_FMT_ORDER: Dict[str, int] = {f: i for i, f in enumerate(ARTIFACT_FORMATS)}
_AUDIO_ORDER: Dict[str, int] = {e: i for i, e in enumerate(AUDIO_EXTENSIONS)}

# Filename suffix -> artifact kind. Several suffixes are aliases: different
# pipeline generations wrote the same artifact under different names.
#   _assignment / _speaker_assignment  -> speaker_assignment
#   _analysis   / _speaker_analysis    -> analysis  (whycast/pipeline/speakers.py:174)
# A file that is exactly "<base>.<fmt>" (no suffix at all) is a transcript too;
# that case is handled separately because it has no suffix to look up.
_SUFFIX_KINDS: Dict[str, str] = {
    "_transcript": "transcript",
    "_ts": "ts",
    "_cleaned": "cleaned",
    "_summary": "summary",
    "_blog": "blog",
    "_blog_alt1": "blog_alt1",
    "_history": "history",
    "_assignment": "speaker_assignment",
    "_speaker_assignment": "speaker_assignment",
    "_analysis": "analysis",
    "_speaker_analysis": "analysis",
    "_merged": "merged",
}

# Longest first, so "_blog_alt1" wins over "_blog" and "_speaker_assignment"
# over "_assignment". Ties broken alphabetically for determinism.
_SUFFIXES_BY_LENGTH: Tuple[str, ...] = tuple(
    sorted(_SUFFIX_KINDS, key=lambda s: (-len(s), s))
)

_FIRST_NUMBER = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """One generated file belonging to an episode."""

    kind: str    #: one of :data:`ARTIFACT_KINDS`
    fmt: str     #: "txt" | "html" | "wiki" | "md"
    path: str    #: absolute path on disk
    size: int    #: bytes
    mtime: float  #: epoch seconds


@dataclass
class Episode:
    """One episode: its audio (if any) and everything generated from it."""

    base_name: str                  #: canonical stem, e.g. "episode_13"
    audio_path: Optional[str] = None  #: absolute path to the audio, None if audio-less
    audio_size: Optional[int] = None
    number: Optional[int] = None    #: parsed episode number for sorting
    artifacts: List[Artifact] = field(default_factory=list)
    mtime: float = 0.0              #: newest mtime across audio + artifacts

    def artifact(self, kind: str, fmt: Optional[str] = None) -> Optional[Artifact]:
        """Return the best artifact of ``kind``, or None.

        Without ``fmt`` the format preference is txt, then html, then wiki,
        then md. With ``fmt`` only that format is considered.

        Two files can occupy the same (kind, fmt) slot, because pipeline
        generations named the same artifact differently: ``ep29`` has both
        ``ep29_speaker_assignment.txt`` and ``ep29_ts_speaker_assignment.txt``.
        The newest file wins - it is the current output of the pipeline, and
        picking by path alone would return the older one forever (``_speaker``
        sorts before ``_ts_speaker``). Path breaks an exact mtime tie, so the
        choice stays deterministic.
        """
        candidates = [
            a for a in self.artifacts
            if a.kind == kind and (fmt is None or a.fmt == fmt)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda a: (_fmt_rank(a.fmt), -a.mtime, a.path))

    def has(self, kind: str) -> bool:
        """True if the episode has at least one artifact of ``kind``."""
        return any(a.kind == kind for a in self.artifacts)

    @property
    def kinds(self) -> List[str]:
        """Distinct artifact kinds present, in :data:`ARTIFACT_KINDS` order."""
        present = {a.kind for a in self.artifacts}
        return [k for k in ARTIFACT_KINDS if k in present]


@dataclass
class ScanResult:
    """Everything one directory scan found.

    ``unmatched`` holds paths that belong to no episode. Most are files, but a
    subdirectory of the podcast directory lands here too (the scan is shallow),
    so a consumer that calls :func:`open` or reads an extension on these must
    cope with a directory. ``os.stat`` works on both.
    """

    episodes: List[Episode] = field(default_factory=list)
    unmatched: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_rank(fmt: str) -> int:
    return _FMT_ORDER.get(fmt, len(_FMT_ORDER))


def _kind_rank(kind: str) -> int:
    return _KIND_ORDER.get(kind, len(_KIND_ORDER))


def parse_episode_number(base_name: str) -> Optional[int]:
    """Extract the episode number from a base name.

    The first run of digits wins, which covers every naming style on disk:
    "episode_13" -> 13, "ep29" -> 29, "Episode 28" -> 28, "Whycast33" -> 33,
    "whycast-episode-13" -> 13, "whycast_episode_4_-_fire_" -> 4.
    Returns None when the name carries no digits at all.
    """
    match = _FIRST_NUMBER.search(base_name)
    if match is None:
        return None
    try:
        return int(match.group())
    except ValueError:  # pragma: no cover - regex guarantees digits
        return None


def _peel_suffix(text: str) -> Tuple[str, Optional[str]]:
    """Strip one known artifact suffix from the right of ``text``.

    Returns ``(remainder, kind)``; ``kind`` is None when nothing matched.
    Longest suffix wins, so "_blog_alt1" is never read as "_blog".
    """
    lowered = text.lower()
    for suffix in _SUFFIXES_BY_LENGTH:
        if lowered.endswith(suffix):
            return text[: len(text) - len(suffix)], _SUFFIX_KINDS[suffix]
    return text, None


def _kind_of_rest(rest: str) -> Optional[str]:
    """Resolve the kind of an artifact from the part after its base name.

    ``rest`` is what follows the base in the filename stem: "" for a bare
    "<base>.txt" transcript, "_summary", or a chain like
    "_ts_speaker_assignment". The chain is peeled from the right and the
    outermost (rightmost) suffix names the kind - "<base>_ts_speaker_assignment"
    is the speaker assignment for <base>, not for "<base>_ts".

    The whole rest must decompose into known suffixes. Anything left over means
    this is not a recognised artifact ("<base>.mp3.ffmpeg.16k_diarization"),
    and inventing a kind for it would be worse than reporting it unmatched.
    """
    if rest == "":
        return "transcript"

    remainder, kind = _peel_suffix(rest)
    if kind is None:
        return None
    outermost = kind
    while remainder:
        remainder, inner = _peel_suffix(remainder)
        if inner is None:
            return None
    return outermost


def _derive_base(stem: str) -> Tuple[Optional[str], Optional[str]]:
    """Derive a base name for a file that matches no audio base.

    Peels known suffixes off the right until none are left; the outermost one
    names the kind. Returns ``(base, kind)``, or ``(None, None)`` when the file
    carries no known suffix - a bare "<name>.txt" is deliberately *not* treated
    as a transcript here, because without audio to anchor it any stray text
    file would mint an episode.
    """
    remainder, kind = _peel_suffix(stem)
    if kind is None or not remainder:
        return None, None
    outermost = kind
    while True:
        shorter, inner = _peel_suffix(remainder)
        if inner is None or not shorter:
            break
        remainder = shorter
    return remainder, outermost


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


class _Entry:
    """One directory entry with its stat data read exactly once."""

    __slots__ = ("name", "path", "stem", "ext", "size", "mtime")

    def __init__(self, name: str, path: str, size: int, mtime: float) -> None:
        self.name = name
        self.path = path
        stem, ext = os.path.splitext(name)
        self.stem = stem
        self.ext = ext.lower()
        self.size = size
        self.mtime = mtime


def scan_podcasts(podcast_dir: str) -> ScanResult:
    """Scan ``podcast_dir`` and group its files into episodes.

    The scan is shallow: a subdirectory is reported in ``unmatched`` rather
    than descended into.

    That is a real trade-off, not a free win, and one of the two
    subdirectories here loses it. ``Other recordings/`` genuinely holds
    unrelated material (stakeholder interviews) and belongs outside the index.
    ``episode31/`` does not: it holds the same episode 31 as the root copy
    (identical audio size, transcripts opening on the same sentence) plus nine
    artifacts the root copy lacks, all of which stay invisible. Descending is
    left for later on purpose - it needs a rule for which copy wins when both
    levels hold an artifact of the same kind, and inventing one silently is
    worse than showing the directory in the unmatched list where it can be seen.

    Raises:
        ConfigurationError: if ``podcast_dir`` is missing or is not a directory.
    """
    root = os.path.abspath(podcast_dir)
    if not os.path.isdir(root):
        raise ConfigurationError(f"Podcast directory not found: {podcast_dir}")

    files: List[_Entry] = []
    unmatched: List[str] = []

    with os.scandir(root) as it:
        for entry in it:
            try:
                if entry.is_dir():
                    unmatched.append(os.path.join(root, entry.name))
                    continue
                stat = entry.stat()
            except OSError as exc:
                emit(
                    "scan",
                    f"Skipping unreadable entry {entry.name}: {exc}",
                    level="warning",
                    path=os.path.join(root, entry.name),
                )
                continue
            files.append(
                _Entry(entry.name, os.path.join(root, entry.name),
                       stat.st_size, stat.st_mtime)
            )

    # Directory order is not guaranteed; sort so a scan of the same directory
    # always produces the same result.
    files.sort(key=lambda e: (e.name.lower(), e.name))

    episodes = _pass1_audio(files, unmatched)
    _pass2_artifacts(files, episodes, unmatched)

    for episode in episodes.values():
        episode.artifacts.sort(
            key=lambda a: (_kind_rank(a.kind), _fmt_rank(a.fmt), a.path)
        )
        # episode.mtime currently holds the audio mtime (0.0 when audio-less).
        episode.mtime = max([episode.mtime] + [a.mtime for a in episode.artifacts])

    ordered = sorted(
        episodes.values(),
        key=lambda e: (
            e.number is None,
            e.number if e.number is not None else 0,
            e.base_name.lower(),
            e.base_name,
        ),
    )
    unmatched.sort(key=lambda p: (os.path.basename(p).lower(), p))

    audio_less = sum(1 for e in ordered if e.audio_path is None)
    emit(
        "scan",
        f"Scanned {root}: {len(ordered)} episodes "
        f"({audio_less} without audio), {len(unmatched)} unmatched entries",
        episodes=len(ordered),
        audio_less=audio_less,
        unmatched=len(unmatched),
        directory=root,
    )
    return ScanResult(episodes=ordered, unmatched=unmatched)


def _pass1_audio(files: List[_Entry], unmatched: List[str]) -> Dict[str, Episode]:
    """Create one episode per audio base. Returns episodes keyed by lowercase base."""
    by_base: Dict[str, List[_Entry]] = {}
    for entry in files:
        if entry.ext in _AUDIO_ORDER:
            by_base.setdefault(entry.stem.lower(), []).append(entry)

    episodes: Dict[str, Episode] = {}
    for key, candidates in by_base.items():
        # One episode has one audio slot. Prefer mp3 over m4a over wav, then
        # sort by name; the rest are reported so no file disappears silently.
        candidates.sort(key=lambda e: (_AUDIO_ORDER[e.ext], e.name))
        primary = candidates[0]
        episodes[key] = Episode(
            base_name=primary.stem,
            audio_path=primary.path,
            audio_size=primary.size,
            number=parse_episode_number(primary.stem),
            artifacts=[],
            mtime=primary.mtime,
        )
        for extra in candidates[1:]:
            emit(
                "scan",
                f"Multiple audio files for base '{primary.stem}': "
                f"using {primary.name}, listing {extra.name} as unmatched",
                level="warning",
                base_name=primary.stem,
                chosen=primary.path,
                extra=extra.path,
            )
            unmatched.append(extra.path)
    return episodes


def _sorted_bases(episodes: Dict[str, Episode]) -> Tuple[str, ...]:
    """Known bases, longest first: "episode_10" must beat "episode_1"."""
    return tuple(sorted(episodes, key=lambda b: (-len(b), b)))


def belongs_to_base(stem: str, base: str) -> bool:
    """True when the filename stem ``stem`` belongs to episode base ``base``.

    The scanner's rule 3, as a predicate, so that everything which has to
    decide "is this file part of that episode?" decides it the same way. The
    stem is the filename without its extension; the match is case-insensitive,
    because the directory is NTFS and holds ``Episode_28.mp3`` next to
    ``episode_28_summary.txt``.

    A base matches when the stem is that base followed by a ``rest`` that is
    either empty or starts with a separator (``_`` or ``.``). The boundary
    check is the whole point: ``episode_1`` and ``episode_10`` both exist here,
    and a plain prefix test would claim ``episode_10_blog`` for ``episode_1``.

    Deleting an episode's files (``whycast.pipeline.feed.delete_episode_files``)
    is the second caller, and it must agree with the scanner exactly: a file the
    scanner attributes to ``episode_10`` must never be deleted by a re-run of
    ``episode_1``.
    """
    stem_lower = stem.lower()
    base_lower = base.lower()
    if not stem_lower.startswith(base_lower):
        return False
    rest = stem_lower[len(base_lower):]
    return not rest or rest[0] in "_."


def _matching_base(stem_lower: str, bases: Tuple[str, ...]) -> Optional[str]:
    """The longest known base ``stem_lower`` belongs to, or None.

    ``bases`` arrives longest-first, and that ordering matters as much as
    :func:`belongs_to_base`'s boundary check: without it ``episode_10_blog``
    would land on ``episode_1``.

    ``.`` is a separator alongside ``_`` even though no suffix in
    :data:`_SUFFIX_KINDS` begins with one, so a dot-form rest never resolves to
    a kind. That is precisely its job: it makes ``episode_1.foo_summary.txt``
    *belong* to ``episode_1`` and be reported unmatched there, instead of
    falling through to :func:`_derive_base`, which would peel ``_summary`` off
    and mint a phantom episode ``episode_1.foo`` beside the real one.
    """
    for base in bases:
        if belongs_to_base(stem_lower, base):
            return base
    return None


def _pass2_artifacts(
    files: List[_Entry],
    episodes: Dict[str, Episode],
    unmatched: List[str],
) -> None:
    """Attach every non-audio file to an episode, or report it unmatched.

    Two sub-passes, because attaching and minting cannot share one loop. A
    bare ``<base>.<fmt>`` is a transcript (module docstring, rule 2) but it
    carries no suffix, so :func:`_derive_base` cannot mint an episode from it -
    it can only ever *join* one. In a single pass over name-sorted files that
    is fatal for an audio-less episode: ``.`` (0x2E) sorts before ``_`` (0x5F),
    so ``episode_52.txt`` is visited before ``episode_52_ts.txt`` and is
    discarded as unmatched before the episode it belongs to exists.

    So: sub-pass A mints every audio-less episode a suffixed file identifies,
    then sub-pass B attaches all files to the now-complete set of bases. The
    realistic trigger is an operator deleting mp3s to reclaim space - 62
    episodes here are about 5 GB of audio - which must not silently detach
    every transcript.

    A bare file still mints nothing on its own (see :func:`_derive_base`): an
    episode whose files are *all* bare stays unmatched, deliberately, because
    any stray ``notes.txt`` would otherwise become an episode.
    """
    # Split off the files that can never be an artifact first, so both
    # sub-passes below iterate over the same, already-filtered list.
    artifact_files: List[Tuple[_Entry, str]] = []
    for entry in files:
        if entry.ext in _AUDIO_ORDER:
            continue
        fmt = entry.ext[1:] if entry.ext.startswith(".") else entry.ext
        if fmt not in _FMT_ORDER:
            # Thumbnails, videos, anything unknown: never an artifact.
            unmatched.append(entry.path)
            continue
        artifact_files.append((entry, fmt))

    bases = _sorted_bases(episodes)

    # -- Sub-pass A: mint the episodes that have no audio ------------------
    # Only files that match no existing base may mint one. Checking that first
    # is what stops "episode_1_frobnicated_summary.txt" - an unrecognised
    # artifact of a real episode - from minting "episode_1_frobnicated".
    for entry, _fmt in artifact_files:
        if _matching_base(entry.stem.lower(), bases) is not None:
            continue
        base_name, kind = _derive_base(entry.stem)
        if base_name is None or kind is None:
            continue
        key = base_name.lower()
        if key in episodes:
            continue
        episodes[key] = Episode(
            base_name=base_name,
            audio_path=None,
            audio_size=None,
            number=parse_episode_number(base_name),
            artifacts=[],
            mtime=0.0,
        )
        bases = _sorted_bases(episodes)

    # -- Sub-pass B: attach every file to its episode -----------------------
    for entry, fmt in artifact_files:
        stem_lower = entry.stem.lower()
        base = _matching_base(stem_lower, bases)
        if base is None:
            unmatched.append(entry.path)
            continue
        kind = _kind_of_rest(stem_lower[len(base):])
        if kind is None:
            # Right episode, unrecognised artifact: reported, never invented.
            unmatched.append(entry.path)
            continue
        episodes[base].artifacts.append(
            Artifact(kind=kind, fmt=fmt, path=entry.path,
                     size=entry.size, mtime=entry.mtime)
        )
