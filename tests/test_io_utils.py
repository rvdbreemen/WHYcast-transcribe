"""
Tests for whycast.io_utils - atomic artifact writes (ADR-008).

Why these exist as a test file and not just a one-off script:
``transcribe.py`` is already a thin shim that re-exports the whycast
functions, so ``test_golden_extraction.py`` compares the new write path
against *itself*. Nothing else in the suite would catch a future "cleanup"
that changes ``newline=None`` to the usual atomic-write idiom ``newline=""``
in :mod:`whycast.io_utils` - which would silently rewrite every artifact
from CRLF to LF on Windows and break the byte-for-byte contract the golden
snapshots exist to protect.

So the first test here is the important one: whatever ``open(path, "w",
encoding="utf-8")`` puts on disk, ``atomic_write_text`` must put there too.
"""

import os
import threading

import pytest

from whycast import io_utils
from whycast.episodes import ARTIFACT_FORMATS
from whycast.io_utils import (
    BACKUP_SUFFIX,
    TEMP_SUFFIX,
    atomic_write_bytes,
    atomic_write_text,
)

# Multi-line on purpose: a newline-free string would pass with any newline
# setting and prove nothing. Non-ASCII on purpose: it pins the encoding too.
SAMPLE = (
    "# Episode 42 — The Answer\n"
    "\n"
    "Een regel met accenten: één café, über.\n"
    "Speaker arrow: SPEAKER_00 → Robert\n"
    "Laatste regel zonder trailing newline"
)


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def leftovers(directory):
    return sorted(n for n in os.listdir(directory) if n.endswith(TEMP_SUFFIX))


# ---------------------------------------------------------------------------
# The byte-for-byte contract
# ---------------------------------------------------------------------------

def test_text_bytes_match_a_plain_write(tmp_path):
    """The whole point: same bytes as open(..., 'w', encoding='utf-8')."""
    plain = tmp_path / "plain.txt"
    atomic = tmp_path / "atomic.txt"

    with open(plain, "w", encoding="utf-8") as f:
        f.write(SAMPLE)
    atomic_write_text(atomic, SAMPLE)

    assert read_bytes(atomic) == read_bytes(plain)


def test_platform_newline_translation_is_preserved(tmp_path):
    """On Windows that means CRLF; the test states the rule, not the platform."""
    plain = tmp_path / "plain.txt"
    atomic = tmp_path / "atomic.txt"

    with open(plain, "w", encoding="utf-8") as f:
        f.write(SAMPLE)
    atomic_write_text(atomic, SAMPLE)

    assert (b"\r\n" in read_bytes(atomic)) == (b"\r\n" in read_bytes(plain))
    if os.linesep == "\r\n":
        assert b"\r\n" in read_bytes(atomic), "CRLF translation was lost"


def test_explicit_newline_is_honoured(tmp_path):
    path = tmp_path / "lf.txt"
    atomic_write_text(path, SAMPLE, newline="")
    assert b"\r\n" not in read_bytes(path)


def test_bytes_match_a_plain_binary_write(tmp_path):
    blob = bytes(range(256)) * 40
    plain = tmp_path / "plain.bin"
    atomic = tmp_path / "atomic.bin"

    with open(plain, "wb") as f:
        f.write(blob)
    atomic_write_bytes(atomic, blob)

    assert read_bytes(atomic) == read_bytes(plain) == blob


def test_returns_the_final_path(tmp_path):
    path = tmp_path / "out.txt"
    assert atomic_write_text(path, "x") == str(path)
    assert atomic_write_bytes(path, b"x") == str(path)


def test_no_temp_file_is_left_after_a_successful_write(tmp_path):
    atomic_write_text(tmp_path / "out.txt", SAMPLE)
    assert leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# Backups
# ---------------------------------------------------------------------------

def test_backup_appears_only_when_overwriting(tmp_path):
    target = tmp_path / "ep042_summary.txt"
    backup = tmp_path / ("ep042_summary.txt" + BACKUP_SUFFIX)

    atomic_write_text(target, "version one\n")
    assert not backup.exists(), "a first write has nothing to back up"

    atomic_write_text(target, "version two\n")
    assert backup.read_text(encoding="utf-8") == "version one\n"
    assert target.read_text(encoding="utf-8") == "version two\n"


def test_backup_is_replaced_not_accumulated(tmp_path):
    target = tmp_path / "ep042_summary.txt"
    for text in ("one\n", "two\n", "three\n"):
        atomic_write_text(target, text)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "ep042_summary.txt", "ep042_summary.txt" + BACKUP_SUFFIX,
    ]
    assert (tmp_path / ("ep042_summary.txt" + BACKUP_SUFFIX)).read_text(
        encoding="utf-8") == "two\n"


def test_backup_false_writes_nothing_extra(tmp_path):
    target = tmp_path / "nobak.txt"
    atomic_write_text(target, "one\n", backup=False)
    atomic_write_text(target, "two\n", backup=False)
    assert [p.name for p in tmp_path.iterdir()] == ["nobak.txt"]


# ---------------------------------------------------------------------------
# Failure: the reason this module exists
# ---------------------------------------------------------------------------

def test_a_failed_replace_leaves_the_original_intact(tmp_path, monkeypatch):
    """A crash at the commit must not cost the reader the old file.

    This is the cancel-a-running-job case in miniature: the pipeline dies
    partway through writing an artifact that already exists on disk.
    """
    target = tmp_path / "ep042_transcript.txt"
    original = "the complete, trustworthy original transcript\n" * 50
    with open(target, "w", encoding="utf-8") as f:
        f.write(original)
    original_bytes = read_bytes(target)

    def exploding_replace(src, dst, *args, **kwargs):
        raise OSError(42, "simulated crash right at the rename")

    monkeypatch.setattr(os, "replace", exploding_replace)

    with pytest.raises(OSError):
        atomic_write_text(target, "HALF WRITTEN GARBAGE THAT MUST NEVER LAND\n")

    assert read_bytes(target) == original_bytes
    assert leftovers(tmp_path) == [], "a failed write must clean up its temp file"


def test_a_failed_write_leaves_no_file_at_a_fresh_target(tmp_path, monkeypatch):
    target = tmp_path / "brand_new.txt"

    def exploding_replace(src, dst, *args, **kwargs):
        raise OSError(42, "simulated crash right at the rename")

    monkeypatch.setattr(os, "replace", exploding_replace)

    with pytest.raises(OSError):
        atomic_write_text(target, SAMPLE)

    assert not target.exists(), "a partial artifact must never appear"
    assert leftovers(tmp_path) == []


def test_a_failure_during_the_write_leaves_the_original_intact(tmp_path, monkeypatch):
    """The other half of the failure surface: it breaks *before* the rename.

    ``test_a_failed_replace_...`` kills the commit, when the temp file is
    already complete. This kills the write itself, so the temp file on disk
    holds half an artifact - which is exactly the thing that must never reach
    the target path or survive as litter.
    """
    target = tmp_path / "ep042_transcript.txt"
    original = "the complete, trustworthy original transcript\n" * 50
    with open(target, "w", encoding="utf-8") as f:
        f.write(original)
    original_bytes = read_bytes(target)

    real_fsync = os.fsync

    def exploding_fsync(fd):
        raise OSError(28, "simulated disk full halfway through the write")

    monkeypatch.setattr(os, "fsync", exploding_fsync)

    with pytest.raises(OSError):
        atomic_write_text(target, "HALF WRITTEN GARBAGE THAT MUST NEVER LAND\n" * 50)

    monkeypatch.setattr(os, "fsync", real_fsync)
    assert read_bytes(target) == original_bytes
    assert leftovers(tmp_path) == [], "a half-written temp file must not survive"
    assert not (tmp_path / ("ep042_transcript.txt" + BACKUP_SUFFIX)).exists(), (
        "nothing was published, so nothing should have been backed up"
    )


def test_a_missing_directory_fails_like_a_plain_open(tmp_path):
    """Same precondition as open(): io_utils does not create directories."""
    missing = tmp_path / "no_such_dir" / "out.txt"
    with pytest.raises(OSError):
        atomic_write_text(missing, SAMPLE)
    with pytest.raises(OSError):
        atomic_write_bytes(missing, b"x")


# ---------------------------------------------------------------------------
# Unicode
#
# Transcripts are Dutch and English with speaker arrows, and summaries come
# back from an LLM that is fond of emoji and typographic dashes. The encoding
# is therefore part of the contract, not an implementation detail.
# ---------------------------------------------------------------------------

UNICODE_SAMPLE = (
    "Emoji: 🎙️ 🇳🇱 ✅ 42\n"
    "CJK: 播客转录 - ポッドキャスト\n"
    "Combining: é vs é (must stay two different strings)\n"
    "Math and dashes: — – … ≠ ∞\n"
    "RTL: مرحبا بالعالم\n"
    "Zero width joiner: 👨‍👩‍👧‍👦\n"
)


def test_unicode_round_trips_exactly(tmp_path):
    path = tmp_path / "ep042_summary.txt"
    atomic_write_text(path, UNICODE_SAMPLE)

    with open(path, "r", encoding="utf-8") as f:
        assert f.read() == UNICODE_SAMPLE
    assert read_bytes(path) == UNICODE_SAMPLE.encode("utf-8").replace(
        b"\n", os.linesep.encode("ascii")
    )


def test_unicode_bytes_match_a_plain_write(tmp_path):
    plain = tmp_path / "plain.txt"
    atomic = tmp_path / "atomic.txt"

    with open(plain, "w", encoding="utf-8") as f:
        f.write(UNICODE_SAMPLE)
    atomic_write_text(atomic, UNICODE_SAMPLE)

    assert read_bytes(atomic) == read_bytes(plain)


def test_unicode_survives_the_backup_copy(tmp_path):
    target = tmp_path / "ep042_summary.txt"
    atomic_write_text(target, UNICODE_SAMPLE)
    atomic_write_text(target, "revised\n")

    backup = tmp_path / ("ep042_summary.txt" + BACKUP_SUFFIX)
    with open(backup, "r", encoding="utf-8") as f:
        assert f.read() == UNICODE_SAMPLE


def test_an_unencodable_character_fails_without_touching_the_target(tmp_path):
    """A wrong encoding must cost the new content, never the old file."""
    target = tmp_path / "ep042_summary.txt"
    atomic_write_text(target, "the original\n")
    original_bytes = read_bytes(target)

    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(target, "emoji do not fit in latin-1: 🎙️", encoding="latin-1")

    assert read_bytes(target) == original_bytes
    assert leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# Concurrent writers
#
# Two jobs writing the same artifact is not a normal state - the worker is
# serial (ADR-008) - but a runner orphaned by a dead worker and a fresh one can
# overlap for a few seconds, which is exactly when a torn file would be written
# and then indexed as finished. mkstemp gives each writer its own O_EXCL temp
# file and os.replace publishes it in one step, so the target must never hold a
# mixture of two writers' bytes.
#
# Two things are deliberately *not* asserted here, because both are Windows
# telling the truth rather than the module misbehaving:
#
# * A concurrent write may fail with ``PermissionError`` (ERROR_ACCESS_DENIED).
#   Windows refuses ``MoveFileEx`` onto a destination another rename is busy
#   replacing, so of ~100 racing writes about a fifth lose. That is the
#   documented contract of these functions - raise, and leave the file that is
#   there intact - and it is loud. What must never happen is a *quiet* mixture,
#   which is what this test actually watches for.
# * There is no concurrent *reader*: a plain ``open()`` on Windows takes a
#   handle without FILE_SHARE_DELETE, so a reader makes the writer's replace
#   fail. That measures readers, not interleaving. ``os.stat`` does share
#   delete, which is why the sampler below uses sizes instead of contents.
# ---------------------------------------------------------------------------

def _writer_contents():
    """Distinct contents of distinct lengths, so any mixture is detectable."""
    return {
        name: f"[{name}]\n" + (f"{name} line with unicode é 42\n" * length)
        for name, length in (("alpha", 400), ("beta", 900), ("gamma", 150), ("delta", 600))
    }


def _on_disk(text):
    """The bytes ``atomic_write_text`` puts on disk for ``text`` on this platform."""
    return text.encode("utf-8").replace(b"\n", os.linesep.encode("ascii"))


def _run_writers(target, contents, rounds, backup):
    """Race every content at ``target`` while sampling its size. Returns a report.

    The sampler is the real interleaving detector: the four contents have four
    different lengths, so a size that is not one of them means a reader could
    have seen half of one writer and half of another. It samples through
    ``os.stat``, which on Windows shares delete access and therefore does not
    make the writers' ``os.replace`` fail.
    """
    sizes = set()
    stop = threading.Event()
    failures = []
    successes = []
    lock = threading.Lock()
    barrier = threading.Barrier(len(contents))

    def sample():
        while not stop.is_set():
            try:
                sizes.add(os.path.getsize(target))
            except OSError:
                # A stat inside the rename's delete-pending window. Not a
                # reading of the file, so nothing to record.
                pass

    def write(text):
        barrier.wait(timeout=60)
        for _ in range(rounds):
            try:
                atomic_write_text(target, text, backup=backup)
            except OSError as exc:
                with lock:
                    failures.append(exc)
            else:
                with lock:
                    successes.append(text)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    threads = [threading.Thread(target=write, args=(text,)) for text in contents.values()]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
            assert not thread.is_alive(), "a writer thread hung"
    finally:
        stop.set()
        sampler.join(timeout=30)
    return sizes, successes, failures


def test_concurrent_writers_never_interleave(tmp_path):
    target = tmp_path / "ep042_transcript.txt"
    contents = _writer_contents()
    rounds = 25
    valid_bytes = {_on_disk(text) for text in contents.values()}
    valid_sizes = {len(blob) for blob in valid_bytes}
    assert len(valid_sizes) == len(contents), "the fixture must give each writer a size"

    temp_paths = []
    temp_lock = threading.Lock()
    real_make_temp = io_utils._make_temp

    def recording_make_temp(path):
        fd, tmp = real_make_temp(path)
        with temp_lock:
            temp_paths.append(tmp)
        return fd, tmp

    monkeyed = False
    try:
        io_utils._make_temp = recording_make_temp
        monkeyed = True
        sizes, successes, failures = _run_writers(target, contents, rounds, backup=False)
    finally:
        if monkeyed:
            io_utils._make_temp = real_make_temp

    assert sizes - valid_sizes == set(), (
        f"the target held {sorted(sizes - valid_sizes)} bytes at some point, which "
        f"is no writer's content: two writers interleaved"
    )
    assert successes, "every single concurrent write failed; that is not contention"
    assert all(isinstance(exc, OSError) for exc in failures), (
        "a concurrent write failed with something other than the documented OSError"
    )
    assert read_bytes(target) in valid_bytes, "the target holds a mixture of two writers"
    assert leftovers(tmp_path) == [], "every writer must clean up after itself"
    assert len(temp_paths) == len(set(temp_paths)), (
        "two writers shared a temp file, which is how bytes get interleaved"
    )
    assert len(temp_paths) == len(contents) * rounds


def test_concurrent_writers_leave_a_whole_backup_too(tmp_path):
    """The ``.bak`` is published by a rename as well, so it cannot be torn either."""
    target = tmp_path / "ep042_summary.txt"
    contents = _writer_contents()
    valid_bytes = {_on_disk(text) for text in contents.values()}
    atomic_write_text(target, next(iter(contents.values())), backup=False)

    _sizes, successes, failures = _run_writers(target, contents, rounds=15, backup=True)

    assert successes
    assert all(isinstance(exc, OSError) for exc in failures)
    backup = tmp_path / ("ep042_summary.txt" + BACKUP_SUFFIX)
    assert read_bytes(target) in valid_bytes
    assert read_bytes(backup) in valid_bytes, "the backup holds a mixture of two writers"
    assert leftovers(tmp_path) == []


# ---------------------------------------------------------------------------
# Interaction with the episode scanner
# ---------------------------------------------------------------------------

def test_temp_and_backup_files_can_never_be_mistaken_for_artifacts():
    """A hard kill cannot run cleanup, so a stray .tmp will exist eventually.

    The filesystem is the source of truth (ADR-008), so the names chosen here
    must be invisible to whycast.episodes: neither suffix may be an artifact
    format, or a half-written file would be indexed as a finished one.
    """
    for suffix in (TEMP_SUFFIX, BACKUP_SUFFIX):
        assert suffix.lstrip(".") not in ARTIFACT_FORMATS


def test_a_stray_temp_file_does_not_become_an_artifact(tmp_path):
    from whycast.episodes import scan_podcasts

    (tmp_path / "episode_42.mp3").write_bytes(b"fake audio")
    atomic_write_text(tmp_path / "episode_42_summary.txt", "current\n")
    atomic_write_text(tmp_path / "episode_42_summary.txt", "revised\n")  # makes a .bak
    (tmp_path / (".episode_42_summary.txt.abcd1234" + TEMP_SUFFIX)).write_text("half")

    result = scan_podcasts(str(tmp_path))

    assert len(result.episodes) == 1, "no phantom episode from .bak/.tmp names"
    episode = result.episodes[0]
    assert episode.base_name == "episode_42"
    assert [(a.kind, a.fmt) for a in episode.artifacts] == [("summary", "txt")]
    assert sorted(os.path.basename(p) for p in result.unmatched) == [
        ".episode_42_summary.txt.abcd1234" + TEMP_SUFFIX,
        "episode_42_summary.txt" + BACKUP_SUFFIX,
    ]


# ---------------------------------------------------------------------------
# sweep_stale_temps - cleanup after a killed write
# ---------------------------------------------------------------------------


def test_sweep_removes_stale_temp_files(tmp_path):
    """A temp file a killed process could not clean up is swept away."""
    from whycast.io_utils import sweep_stale_temps

    stale = tmp_path / (".episode_42_summary.txt.abcd1234" + TEMP_SUFFIX)
    stale.write_text("half written")

    removed = sweep_stale_temps(tmp_path, min_age_seconds=0.0)

    assert [os.path.basename(p) for p in removed] == [stale.name]
    assert not stale.exists()


def test_sweep_leaves_artifacts_and_backups_alone(tmp_path):
    """Only temp files go; real artifacts and their backups must survive."""
    from whycast.io_utils import sweep_stale_temps

    artifact = tmp_path / "episode_42_summary.txt"
    artifact.write_text("real content")
    backup = tmp_path / ("episode_42_summary.txt" + BACKUP_SUFFIX)
    backup.write_text("previous content")
    audio = tmp_path / "episode_42.mp3"
    audio.write_bytes(b"fake audio")

    assert sweep_stale_temps(tmp_path, min_age_seconds=0.0) == []
    assert artifact.read_text() == "real content"
    assert backup.read_text() == "previous content"
    assert audio.exists()


def test_sweep_spares_young_temp_files(tmp_path):
    """The age guard protects a temp file from a writer that may still be live."""
    from whycast.io_utils import sweep_stale_temps

    fresh = tmp_path / (".episode_42_summary.txt.efgh5678" + TEMP_SUFFIX)
    fresh.write_text("in flight")

    assert sweep_stale_temps(tmp_path, min_age_seconds=3600) == []
    assert fresh.exists()


def test_sweep_on_a_missing_directory_is_a_no_op(tmp_path):
    from whycast.io_utils import sweep_stale_temps

    assert sweep_stale_temps(tmp_path / "does-not-exist") == []


def test_sweep_with_zero_age_removes_a_just_written_temp(tmp_path):
    """min_age_seconds=0 means no age guard, whatever the clock granularity.

    Regression: comparing mtime against a cutoff of "now" made this depend on
    filesystem timestamp resolution, so the sweep the worker runs right after a
    kill spared the very file it was called to remove - intermittently, which
    is the worst kind.
    """
    from whycast.io_utils import sweep_stale_temps

    just_written = tmp_path / (".episode_42_summary.txt.fresh999" + TEMP_SUFFIX)
    just_written.write_text("killed mid-write")

    removed = sweep_stale_temps(tmp_path, min_age_seconds=0.0)

    assert [os.path.basename(p) for p in removed] == [just_written.name]
    assert not just_written.exists()
