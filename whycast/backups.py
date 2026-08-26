"""
Versioned artifact backups: nothing a run overwrites is ever lost.

The rule (decided by the owner on 2026-08-26, ADR-011): before a run replaces an
artifact, the existing file is **moved** into a backup directory, and a backup
is never overwritten.

    backups/<base_name>/<run stamp>/<filename>

The run stamp is fixed once per process, so everything a single run displaces
lands together and can be read as one previous version rather than a pile of
loose files. Two runs can never collide, and if the clock ever hands out the
same second twice, :func:`_unique` adds a counter rather than clobbering.

Why moved rather than copied
----------------------------
A copy leaves the old file in ``podcasts/`` until the run overwrites it, so a
run that dies halfway leaves a directory holding a mix of old and new output
with nothing to say which is which. Moving makes the state unambiguous: what is
in ``podcasts/`` is from this run, and what the run displaced is in the backup.

Why not ``.bak``
----------------
ADR-009 removed per-artifact ``.bak`` files because they doubled the file count
in ``podcasts/`` and buried the corpus in rows nobody read. That objection was
to the *location*, never to keeping history. A separate tree keeps ``podcasts/``
exactly as clean while making every previous version recoverable, and unlike a
single ``.bak`` it keeps more than one.

What is never backed up
-----------------------
Only files a run overwrites. The pipeline's inputs - the transcript a
post-processing run reads, ``<base>_speakers.json``, ``vocabulary.json``, the
prompt files - are read, not written, so they stay where they are. Moving those
would leave the job with nothing to work from.
"""

import logging
import os
import re
import shutil
import time
from typing import Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    "backup_root",
    "run_stamp",
    "backup_dir_for",
    "move_to_backup",
    "move_artifacts",
]

PathLike = Union[str, "os.PathLike[str]"]

#: Directory holding every version a run displaced. Overridable so tests and
#: alternative layouts do not have to write next to the real corpus.
_ENV_BACKUP_DIR = "WHYCAST_BACKUP_DIR"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Set once per process; see the module docstring.
_RUN_STAMP: Optional[str] = None

#: Artifact filenames are ``<base>_<suffix>.<ext>``; the base is what groups a
#: backup. Anything unparseable is filed under this name rather than dropped.
_UNGROUPED = "_other"


def backup_root() -> str:
    """Root of the backup tree. Not created by this call."""
    configured = os.environ.get(_ENV_BACKUP_DIR)
    if configured:
        return os.path.abspath(configured)
    return os.path.join(_REPO_ROOT, "backups")


def run_stamp() -> str:
    """Timestamp identifying this process's run, computed once.

    One stamp per process is what makes a backup directory readable as "the
    state before this run" instead of a scatter of per-file timestamps.
    """
    global _RUN_STAMP
    if _RUN_STAMP is None:
        _RUN_STAMP = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
    return _RUN_STAMP


def backup_dir_for(path: PathLike, base_name: Optional[str] = None) -> str:
    """Directory this run's copy of ``path`` belongs in. Not created here."""
    if base_name is None:
        base_name = _base_from_filename(os.path.basename(os.fspath(path)))
    return os.path.join(backup_root(), base_name, run_stamp())


def move_to_backup(path: PathLike, base_name: Optional[str] = None) -> Optional[str]:
    """Move ``path`` into this run's backup directory.

    Returns the file's new location, or None when there was nothing to move.
    Never overwrites: a name already taken in the backup gets a counter.

    Failure is reported to the caller by raising, because losing the previous
    version silently is exactly what this module exists to prevent. Callers that
    would rather continue than fail must say so explicitly.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        return None

    directory = backup_dir_for(path, base_name)
    os.makedirs(directory, exist_ok=True)
    destination = _unique(os.path.join(directory, os.path.basename(path)))

    try:
        os.replace(path, destination)
    except OSError:
        # Different volume: os.replace cannot cross one, shutil.move can.
        shutil.move(path, destination)
    logger.info("Backed up %s -> %s", path, destination)
    return destination


def copy_to_backup(path: PathLike, base_name: Optional[str] = None) -> Optional[str]:
    """Preserve ``path`` in this run's backup, leaving the original in place.

    For the files a job both reads and writes. ``<base>_merged.txt`` is the
    case that forced this: speaker assignment produces it, and post-processing
    then prefers it over the raw transcript as its input. Moving it away before
    the run - the right thing for a pure output - left the job with nothing to
    read, so it failed before writing anything.

    The previous version still has to survive, because the run will overwrite
    it. Copying satisfies both: the job finds its input where it expects it,
    and the version it replaces is already safe.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        return None

    directory = backup_dir_for(path, base_name)
    os.makedirs(directory, exist_ok=True)
    destination = _unique(os.path.join(directory, os.path.basename(path)))
    shutil.copy2(path, destination)
    logger.info("Preserved %s -> %s", path, destination)
    return destination


def move_artifacts(paths, base_name: Optional[str] = None) -> list:
    """Clear every path in ``paths`` out of the way, into this run's backup.

    Called once, before a run starts, rather than file by file as each write
    happens. That ordering is the point: afterwards ``podcasts/`` holds only
    what this run produced, so a run that dies halfway leaves an unambiguous
    state instead of a mix of old and new output with nothing to tell them
    apart.

    Returns ``[(original, backup), ...]`` for what was moved. Missing paths are
    skipped silently - there is nothing to preserve.
    """
    moved = []
    for path in paths:
        destination = move_to_backup(path, base_name)
        if destination:
            moved.append((os.fspath(path), destination))
    return moved


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _unique(destination: str) -> str:
    """A path that does not exist yet, so a backup can never be overwritten."""
    if not os.path.exists(destination):
        return destination
    stem, extension = os.path.splitext(destination)
    counter = 2
    while os.path.exists(f"{stem}.{counter}{extension}"):
        counter += 1
    return f"{stem}.{counter}{extension}"


def _base_from_filename(filename: str) -> str:
    """Group artifacts of one episode together from the filename alone.

    ``episode_13_summary.txt`` and ``episode_13.txt`` both belong to
    ``episode_13``. The suffix list mirrors the scanner's vocabulary; anything
    that matches none of it keeps its own stem, which keeps unrelated files from
    piling into one directory.
    """
    stem = os.path.splitext(filename)[0]
    for suffix in (
        "_speaker_assignment",
        "_ts_speaker_assignment",
        "_blog_alt1",
        "_speaker_analysis",
        "_transcript",
        "_assignment",
        "_analysis",
        "_summary",
        "_cleaned",
        "_history",
        "_merged",
        "_speakers",
        "_blog",
        "_ts",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] or _UNGROUPED
    return stem or _UNGROUPED
