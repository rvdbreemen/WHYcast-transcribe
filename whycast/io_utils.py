"""
Atomic artifact writes for the WHYcast pipeline (ADR-008).

Why this module exists
----------------------
The web UI can cancel a running job, which kills the child process that runs
the pipeline. A CUDA out-of-memory kill does the same thing, uninvited. With a
plain ``open(path, "w")`` the target file is truncated the moment the handle is
opened, so a kill halfway through leaves a short, syntactically fine-looking
transcript on disk. The filesystem is the source of truth (ADR-008), so the
scanner would index that stump as a finished artifact.

The fix is the classic write-temp-then-rename dance:

1. write the new content to a temp file in the *same* directory,
2. ``flush()`` + ``os.fsync()`` so the bytes are with the OS, not in Python's
   buffer,
3. copy the existing target to ``<path>.bak`` - through a temp file of its own,
   so the backup is published by a rename too (optional safety net),
4. ``os.replace(tmp, path)`` - atomic on both Windows (``MoveFileEx`` with
   ``MOVEFILE_REPLACE_EXISTING``) and POSIX (``rename(2)``).

Same directory matters: ``os.replace`` is only atomic within one filesystem.
A temp file in ``%TEMP%`` on another volume degrades to copy-then-delete, which
is exactly the non-atomic behaviour we are removing.

At every instant the target path holds either the complete old file or the
complete new file. Never half of either. The same now holds for ``<path>.bak``:
it is a rename of a fully written, fsynced copy, so a process killed mid-backup
leaves the *previous* ``.bak`` in place rather than a full-length file of NUL
bytes that looks like a valid restore point.

Encoding and newlines
---------------------
``atomic_write_text`` mirrors the defaults of the builtin ``open()``:
``newline=None`` means universal-newline translation, so on Windows every
``\\n`` still becomes ``\\r\\n`` on disk. This is deliberate. The golden-master
tests pin the exact bytes of pipeline artifacts against snapshots taken from
the pre-refactor monolith; passing the usual atomic-write idiom ``newline=""``
here would silently rewrite every artifact to LF and break them.

Scope
-----
This is for whole-file writes - transcripts, summaries, blogs, analyses, and
the downloaded episode audio they are made from. Log files and caches keep
writing the way they always did: appending to a log is not a whole-file
replacement.

Audio used to be excluded here, on the grounds that buffering a 100 MB mp3 in
RAM to hand it to ``atomic_write_bytes`` would trade a real problem for a worse
one. That objection was to the *buffering*, never to the atomicity, and it is
answered by :func:`atomic_writer`: the caller streams chunk by chunk into the
temp file and the rename still publishes the result in one step. So a download
that dies halfway - a dropped connection, a cancelled job - now leaves no file
at the target path at all, rather than a truncated mp3 the scanner would index
as a finished episode and the pipeline would happily transcribe.

Stdlib only, no imports from the rest of the package: every pipeline module
writes files, so any dependency here becomes a circular-import risk.
"""

import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from typing import IO, Iterator, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    "atomic_write_text",
    "atomic_write_bytes",
    "atomic_writer",
    "sweep_stale_temps",
    "BACKUP_SUFFIX",
    "TEMP_SUFFIX",
]

#: Suffix of the copy kept of the previous version of a target file.
BACKUP_SUFFIX = ".bak"

#: Suffix of the in-progress temp file. Deliberately not one of
#: ``whycast.episodes.ARTIFACT_FORMATS`` (txt/html/wiki/md), so a temp file that
#: somehow survives is reported as unmatched by the scanner and never indexed
#: as a finished artifact. The same holds for ``BACKUP_SUFFIX``.
TEMP_SUFFIX = ".tmp"

#: Longest slice of the target filename reused in the temp filename. Bounded so
#: a long artifact name plus the random part cannot push the temp path over the
#: Windows MAX_PATH limit that the real path stayed under.
_TEMP_NAME_LIMIT = 60

PathLike = Union[str, "os.PathLike[str]"]


def atomic_write_text(
    path: PathLike,
    text: str,
    encoding: str = "utf-8",
    backup: bool = True,
    newline: Optional[str] = None,
    errors: Optional[str] = None,
) -> str:
    """Write ``text`` to ``path`` atomically.

    Args:
        path: Final path of the file. Its directory must already exist - the
            same requirement plain ``open(path, "w")`` has.
        text: Full contents to write.
        encoding: Text encoding, as for ``open()``.
        backup: When True and ``path`` already exists, copy it to
            ``path + ".bak"`` (overwriting a previous ``.bak``) before the
            replace.
        newline: Newline handling, as for ``open()``. ``None`` - the default -
            keeps the platform translation the builtin does, so artifacts stay
            byte-identical to what the pre-refactor code wrote.
        errors: Encoding error policy, as for ``open()``.

    Returns:
        The final path, as passed in.

    Raises:
        OSError: If the directory is missing or the write, fsync or replace
            fails. In that case the temp file is removed and any pre-existing
            file at ``path`` is left untouched.
    """
    path = os.fspath(path)
    fd, tmp_path = _make_temp(path)
    try:
        try:
            handle = os.fdopen(
                fd, "w", encoding=encoding, newline=newline, errors=errors
            )
        except BaseException:
            # fdopen did not take ownership of the descriptor; close it here or
            # it leaks for the lifetime of the process.
            os.close(fd)
            raise
        with handle as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _commit(tmp_path, path, backup)
    except BaseException:
        _discard(tmp_path)
        raise
    return path


def atomic_write_bytes(path: PathLike, data: bytes, backup: bool = True) -> str:
    """Write ``data`` to ``path`` atomically.

    The binary counterpart of :func:`atomic_write_text`; see it for the
    arguments, the return value and the failure behaviour.
    """
    path = os.fspath(path)
    fd, tmp_path = _make_temp(path)
    try:
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with handle as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _commit(tmp_path, path, backup)
    except BaseException:
        _discard(tmp_path)
        raise
    return path


@contextmanager
def atomic_writer(
    path: PathLike,
    mode: str = "wb",
    encoding: Optional[str] = None,
    newline: Optional[str] = None,
    backup: bool = True,
) -> Iterator[IO]:
    """Stream to ``path`` atomically; yields the handle to write to.

    The streaming counterpart of :func:`atomic_write_bytes`, for content that
    must not be held in memory as one object. An episode mp3 is 20-100 MB; the
    downloader receives it in 8 KB chunks and writes each one as it arrives, so
    it never needs the whole file at once. Everything else is the same: the
    handle is the temp file made by :func:`_make_temp` next to the target, and
    the target only ever appears via the same :func:`_commit` rename the other
    writers use.

    Leaving the ``with`` block normally flushes, fsyncs, closes and commits.
    Leaving it by *any* exception - ``OSError`` from a dropped connection,
    ``KeyboardInterrupt`` from Ctrl+C, ``GeneratorExit`` when the block is
    abandoned - discards the temp file and re-raises, so an aborted write is
    not published: the target keeps its previous content, or stays absent if it
    never existed. That is the point for downloads. A half-received mp3 at the
    real path is worse than no file, because the scanner (ADR-008) reads the
    filesystem as the source of truth and would index the stump as a finished
    episode.

    Args:
        path: Final path of the file. Its directory must already exist.
        mode: Write mode for the handle, as for ``open()``. Binary ``"wb"`` by
            default, since that is what streaming callers want; ``"w"`` gives a
            text handle and then ``encoding`` and ``newline`` apply.
        encoding: Text encoding, as for ``open()``. Must stay ``None`` in a
            binary mode - ``open()`` rejects the combination.
        newline: Newline handling, as for ``open()``. Same restriction.
        backup: When True and ``path`` already exists, copy it to
            ``path + ".bak"`` before the replace. Pass False when the target has
            no previous version worth keeping - audio downloads are the source
            recording, and a second 20-100 MB copy per download is pure waste.

    Yields:
        The open handle of the temp file. Do not close it inside the block; the
        context manager closes it before committing.

    Raises:
        OSError: If the directory is missing or the write, fsync or replace
            fails. Whatever the block raises propagates unchanged.

    A *killed* process runs no handler, so the temp file survives - the same
    hole :func:`sweep_stale_temps` exists to clean up, and by the same design
    it is inert until then: ``TEMP_SUFFIX`` is not an audio extension, so the
    scanner reports the leftover as unmatched instead of as an episode.
    """
    path = os.fspath(path)
    fd, tmp_path = _make_temp(path)
    try:
        try:
            handle = os.fdopen(fd, mode, encoding=encoding, newline=newline)
        except BaseException:
            # fdopen did not take ownership of the descriptor; close it here or
            # it leaks for the lifetime of the process.
            os.close(fd)
            raise
        with handle as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        _commit(tmp_path, path, backup)
    except BaseException:
        # The `with` above has already closed the handle by the time this runs,
        # which Windows insists on before the temp file can be unlinked.
        _discard(tmp_path)
        raise


def sweep_stale_temps(directory: PathLike, min_age_seconds: float = 60.0) -> List[str]:
    """Remove temp files left behind by writes that were killed mid-flight.

    A normal failed write cleans up after itself: ``atomic_write_*`` discards
    its temp file in an ``except BaseException`` handler. A process that is
    *killed* never runs that handler, so its temp file survives - measured on
    Windows: cancelling a job leaves ``.episode_99_summary.txt.tkl7p_c7.tmp``
    next to the artifacts. The scanner correctly reports it as unmatched, and
    without a sweep they accumulate one per cancelled write forever.

    Call this only when no write can be in flight in ``directory`` - after
    killing a job's process tree, from the worker. Jobs run strictly serially
    (ADR-008), so at that moment nothing else is writing. Calling it during a
    write would race a live writer's temp file, which is why the atomic write
    path itself never calls it.

    ``min_age_seconds`` is a second safety net: a temp file younger than this is
    left alone even if the caller got the timing wrong.

    Returns the paths that were removed.
    """
    directory = os.fspath(directory)
    removed: List[str] = []
    if not os.path.isdir(directory):
        return removed

    cutoff = time.time() - min_age_seconds
    for name in os.listdir(directory):
        if not (name.startswith(".") and name.endswith(TEMP_SUFFIX)):
            continue
        candidate = os.path.join(directory, name)
        try:
            if not os.path.isfile(candidate) or os.path.getmtime(candidate) > cutoff:
                continue
            os.unlink(candidate)
        except OSError as e:
            logger.warning("Could not remove stale temp file %s: %s", candidate, e)
            continue
        removed.append(candidate)

    if removed:
        logger.info("Removed %d stale temp file(s) from %s", len(removed), directory)
    return removed


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _make_temp(path: str):
    """Create the temp file next to ``path``; return ``(fd, tmp_path)``.

    ``mkstemp`` creates the file with mode 0600 and O_EXCL, so two concurrent
    writers of the same artifact each get their own temp file and the last
    ``os.replace`` wins cleanly - no interleaved bytes.
    """
    directory = os.path.dirname(os.path.abspath(path))
    prefix = "." + os.path.basename(path)[:_TEMP_NAME_LIMIT] + "."
    return tempfile.mkstemp(dir=directory, prefix=prefix, suffix=TEMP_SUFFIX)


def _commit(tmp_path: str, path: str, backup: bool) -> None:
    """Back up the current file if asked, then move the temp file into place."""
    if backup:
        _make_backup(path)
    os.replace(tmp_path, path)


def _make_backup(path: str) -> None:
    """Best-effort *atomic* copy of an existing ``path`` to ``path + ".bak"``.

    Best-effort on purpose. By the time this runs the new content is already
    written and fsynced; refusing to publish it because the safety copy failed
    would throw away the result of a job that can take tens of minutes on the
    GPU. A failed backup is logged and the write proceeds.

    "Best-effort" used to include a failure mode nothing could see. This wrote
    ``shutil.copy2(path, path + ".bak")`` straight onto the backup, and on
    Windows ``copy2`` goes through ``CopyFile2``, which pre-sizes the
    destination and then fills it. Kill the process mid-copy - a cancel, a CUDA
    OOM kill, ``taskkill /F`` - and the ``.bak`` is left at exactly the right
    size, with the right mtime, and mostly NUL bytes. No ``OSError`` is raised
    because the process is gone, so nothing is logged and no cheap check
    (size, mtime) can tell it from a good backup. A recovery that restores from
    it restores zeros.

    So the backup is published the same way the artifact is: copy into a temp
    file next to it, fsync, ``os.replace``. A kill at any point leaves either
    the previous ``.bak`` or the new one, plus at worst a ``.tmp`` file the
    scanner already quarantines. ``copystat`` after the copy keeps the mtime
    and mode that ``copy2`` used to carry over, and the rename carries them to
    the backup.
    """
    if not os.path.exists(path):
        return
    backup = path + BACKUP_SUFFIX
    try:
        fd, tmp_path = _make_temp(backup)
    except OSError as e:
        logger.warning("Could not back up %s before overwriting it: %s", path, e)
        return
    try:
        with open(path, "rb") as source, os.fdopen(fd, "wb") as target:
            shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        shutil.copystat(path, tmp_path)
        os.replace(tmp_path, backup)
    except OSError as e:
        _discard(tmp_path)
        logger.warning("Could not back up %s before overwriting it: %s", path, e)


def _discard(tmp_path: str) -> None:
    """Remove a temp file whose write did not make it to the target.

    Without this a failed or cancelled write leaves ``.tmp`` litter in the
    episode directories that the scanner walks on every rebuild.
    """
    try:
        os.unlink(tmp_path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not remove temp file %s: %s", tmp_path, e)
