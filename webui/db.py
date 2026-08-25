"""
Rebuildable SQLite index over ``podcasts/`` for the WHYcast web UI (ADR-008).

The filesystem is the source of truth. This database is a **cache**: delete the
file, call :func:`init_db` and :func:`rescan`, and every row comes back. Nothing
here is authoritative and nothing here writes to ``podcasts/``.

Design notes worth knowing before changing this module:

* **Natural keys, no surrogate ids.** Episodes are keyed by ``base_key``
  (``base_name.lower()``, exactly the key :mod:`whycast.episodes` groups on),
  artifacts and unmatched entries by their absolute path. A rebuild therefore
  produces byte-identical rows instead of merely equivalent ones with shifted
  autoincrement ids.
* **``position`` columns preserve scanner order.** ``ORDER BY position``
  reproduces the exact ordering :func:`whycast.episodes.scan_podcasts`
  promises, so the UI never has to re-derive the sort and never disagrees
  with the scanner.
* **Autocommit plus explicit transactions.** The connection is opened with
  ``isolation_level=None``; :func:`rescan` wraps its replace-all in one
  ``BEGIN IMMEDIATE`` / ``COMMIT``. A crash mid-scan rolls back to the previous
  index; it can never leave half an index behind.
* **One module-level lock.** ``check_same_thread=False`` lets FastAPI share one
  connection across its threadpool, but a read issued on that same connection
  while :func:`rescan` is mid-transaction sits *inside* that transaction and
  would see the half-deleted state. Every public function therefore takes
  ``_LOCK``. One user, ~48 episodes: contention is irrelevant, correctness is
  not.
* **The schema version rebuilds only the tables this module owns.** The phase-2
  job queue lives in the same database and is *not* rebuildable from disk, so a
  version bump must never drop it.

No ``print()``, no ``exit()``: progress goes through :mod:`whycast.events`,
failures are raised (ADR-008 Decision Contract).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from whycast.episodes import (
    ARTIFACT_FORMATS,
    ARTIFACT_KINDS,
    ScanResult,
    scan_podcasts,
)
from whycast.errors import WhycastError
from whycast.events import emit

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_DB_PATH",
    "DEFAULT_PODCAST_DIR",
    "IndexConsistencyError",
    "init_db",
    "rescan",
    "list_episodes",
    "get_episode",
    "list_unmatched",
    "get_meta",
]

#: Bumped whenever the table layout below changes. On mismatch the index tables
#: are dropped and recreated empty; the next rescan refills them from disk.
SCHEMA_VERSION = 1

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)

#: Default location of the index database (inside the webui package).
DEFAULT_DB_PATH = os.path.join(_PACKAGE_DIR, "whycast_webui.db")

#: Default podcast directory: the repository's ``podcasts/``.
DEFAULT_PODCAST_DIR = os.path.join(_REPO_ROOT, "podcasts")

#: Tables this module owns. A schema bump drops exactly these, in this order
#: (children before parents, so the foreign key stays satisfied).
_INDEX_TABLES = ("artifacts", "episodes", "unmatched")

#: ``meta`` keys this module owns. Other subsystems may add their own keys to
#: the same table; a schema rebuild must leave those alone.
_MANAGED_META_KEYS = ("schema_version", "last_scan", "podcast_dir")

# See the module docstring: this guards the shared connection, readers included.
_LOCK = threading.RLock()

_KIND_ORDER = {kind: i for i, kind in enumerate(ARTIFACT_KINDS)}
#: Format preference, mirroring whycast.episodes: txt, html, wiki, md.
_FMT_ORDER = {fmt: i for i, fmt in enumerate(ARTIFACT_FORMATS)}

PathLike = Union[str, "os.PathLike[str]", None]


class IndexConsistencyError(WhycastError):
    """Raised when a scan result cannot be indexed unambiguously.

    In practice: two episodes whose ``base_name`` differs only in case. The
    scanner cannot produce that today (it groups on the lowercase name), so
    this exists to make a future scanner change fail loudly instead of
    silently dropping an episode.
    """


# Individual statements, not one executescript() blob: sqlite3.executescript()
# implicitly COMMITs any pending transaction before it runs, which would silently
# break the BEGIN IMMEDIATE that wraps a schema rebuild.
_META_STATEMENT = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
)"""

_SCHEMA_STATEMENTS = (
    """
CREATE TABLE IF NOT EXISTS episodes (
    base_key   TEXT    PRIMARY KEY,  -- base_name.lower(); the scanner's own key
    base_name  TEXT    NOT NULL,     -- casing of the audio file that won
    audio_path TEXT,                 -- NULL for audio-less episodes
    audio_size INTEGER,
    number     INTEGER,              -- NULL when the name carries no digits
    mtime      REAL    NOT NULL,     -- newest mtime across audio + artifacts
    position   INTEGER NOT NULL      -- index in ScanResult.episodes
)""",
    """
CREATE TABLE IF NOT EXISTS artifacts (
    path      TEXT    PRIMARY KEY,   -- absolute path; one file, one row
    base_key  TEXT    NOT NULL REFERENCES episodes(base_key) ON DELETE CASCADE,
    kind      TEXT    NOT NULL,      -- one of whycast.episodes.ARTIFACT_KINDS
    fmt       TEXT    NOT NULL,      -- txt | html | wiki | md
    size      INTEGER NOT NULL,
    mtime     REAL    NOT NULL,
    position  INTEGER NOT NULL       -- index in Episode.artifacts
)""",
    """
CREATE TABLE IF NOT EXISTS unmatched (
    path     TEXT    PRIMARY KEY,
    name     TEXT    NOT NULL,       -- basename, for display
    ext      TEXT    NOT NULL,       -- lowercase extension incl. dot, "" if none
    size     INTEGER,                -- best effort; NULL when the stat failed
    mtime    REAL,                   -- best effort; NULL when the stat failed
    position INTEGER NOT NULL        -- index in ScanResult.unmatched
)""",
    "CREATE INDEX IF NOT EXISTS idx_episodes_position ON episodes(position)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_number ON episodes(number)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_episode ON artifacts(base_key, position)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind)",
    "CREATE INDEX IF NOT EXISTS idx_unmatched_position ON unmatched(position)",
)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def init_db(path: PathLike = None) -> sqlite3.Connection:
    """Open (creating if needed) the index database and return the connection.

    Args:
        path: database file, or ``":memory:"``. Defaults to
            :data:`DEFAULT_DB_PATH`. Parent directories are created.

    Returns:
        A connection with ``row_factory = sqlite3.Row``, autocommit mode
        (``isolation_level=None``) and ``check_same_thread=False`` so FastAPI
        can share it across its threadpool. All access must go through this
        module's functions, which serialise on ``_LOCK``.

    Raises:
        sqlite3.Error: if the database cannot be opened or the schema created.
    """
    db_path = DEFAULT_DB_PATH if path is None else os.fspath(path)
    in_memory = db_path == ":memory:" or db_path.startswith("file::memory:")

    if not in_memory:
        db_path = os.path.abspath(db_path)
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute("PRAGMA foreign_keys = ON")
        if not in_memory:
            # WAL keeps readers from blocking on the rescan writer. It must run
            # outside a transaction, which autocommit mode guarantees.
            conn.execute("PRAGMA journal_mode = WAL")
            # This database is a cache: losing the last transaction to a power
            # cut costs one rescan, so full fsync-per-commit is not worth it.
            conn.execute("PRAGMA synchronous = NORMAL")
        with _LOCK:
            _ensure_schema(conn, db_path)
    except BaseException:
        conn.close()
        raise
    return conn


def _ensure_schema(conn: sqlite3.Connection, db_path: str) -> None:
    """Create the schema, rebuilding the index tables on a version mismatch."""
    conn.execute(_META_STATEMENT)
    stored = _read_meta_value(conn, "schema_version")

    conn.execute("BEGIN IMMEDIATE")
    try:
        if stored is not None and stored != str(SCHEMA_VERSION):
            # Drop only what this module owns. The phase-2 job queue shares this
            # database and is NOT rebuildable from disk (ADR-008), so `meta` and
            # any table not listed here must survive untouched.
            for table in _INDEX_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.executemany(
                "DELETE FROM meta WHERE key = ?",
                [(key,) for key in _MANAGED_META_KEYS],
            )
            emit(
                "index",
                f"Index schema {stored} != {SCHEMA_VERSION}; rebuilding "
                f"episode index in {db_path} (a rescan will refill it)",
                level="warning",
                database=db_path,
                old_schema_version=stored,
                new_schema_version=SCHEMA_VERSION,
            )
        for statement in _SCHEMA_STATEMENTS:
            conn.execute(statement)
        _write_meta_value(conn, "schema_version", SCHEMA_VERSION)
        conn.execute("COMMIT")
    except BaseException:
        # Guarded: if the COMMIT itself failed, SQLite may already have ended
        # the transaction, and an unguarded ROLLBACK would raise "cannot
        # rollback - no transaction is active" *over* the real error.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# Rescan
# ---------------------------------------------------------------------------


def rescan(conn: sqlite3.Connection, podcast_dir: PathLike = None) -> Dict[str, Any]:
    """Rebuild the whole index from ``podcast_dir`` and return a summary.

    Replace-all inside a single ``BEGIN IMMEDIATE`` transaction: idempotent
    (two runs over an unchanged directory produce identical rows) and atomic
    (a crash mid-scan leaves the previous index intact, never half of a new
    one). Nothing is written to ``podcast_dir``.

    Args:
        conn: connection from :func:`init_db`.
        podcast_dir: directory to scan; defaults to :data:`DEFAULT_PODCAST_DIR`.

    Returns:
        Counts and timing: ``episodes``, ``episodes_with_audio``,
        ``episodes_without_audio``, ``artifacts``, ``unmatched``,
        ``podcast_dir``, ``last_scan``, ``duration_seconds``.

    Raises:
        ConfigurationError: if ``podcast_dir`` is missing or not a directory.
        IndexConsistencyError: if two episodes share a lowercase base name.
    """
    directory = DEFAULT_PODCAST_DIR if podcast_dir is None else os.fspath(podcast_dir)
    started = time.time()

    with _LOCK:
        # Scanning inside the lock keeps two concurrent rescans from committing
        # out of order. It reads directory metadata only and costs milliseconds.
        result = scan_podcasts(directory)
        root = os.path.abspath(directory)

        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM artifacts")
            conn.execute("DELETE FROM episodes")
            conn.execute("DELETE FROM unmatched")
            artifact_count = _insert_episodes(conn, result)
            _insert_unmatched(conn, result)
            _write_meta_value(conn, "schema_version", SCHEMA_VERSION)
            _write_meta_value(conn, "podcast_dir", root)
            _write_meta_value(conn, "last_scan", repr(started))
            conn.execute("COMMIT")
        except BaseException:
            # Guarded for the same reason as in _ensure_schema: a failed COMMIT
            # may already have ended the transaction, and the ROLLBACK would
            # then bury the error that actually matters.
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    with_audio = sum(1 for e in result.episodes if e.audio_path is not None)
    summary: Dict[str, Any] = {
        "episodes": len(result.episodes),
        "episodes_with_audio": with_audio,
        "episodes_without_audio": len(result.episodes) - with_audio,
        "artifacts": artifact_count,
        "unmatched": len(result.unmatched),
        "podcast_dir": root,
        "last_scan": started,
        "duration_seconds": time.time() - started,
    }
    emit(
        "index",
        f"Indexed {summary['episodes']} episodes, {summary['artifacts']} artifacts, "
        f"{summary['unmatched']} unmatched files from {root}",
        **summary,
    )
    return summary


def _insert_episodes(conn: sqlite3.Connection, result: ScanResult) -> int:
    """Insert every episode and its artifacts. Returns the artifact count."""
    episode_rows = []
    artifact_rows = []
    seen: Dict[str, str] = {}

    for position, episode in enumerate(result.episodes):
        base_key = episode.base_name.lower()
        previous = seen.get(base_key)
        if previous is not None:
            raise IndexConsistencyError(
                f"Two episodes share the base key '{base_key}': "
                f"'{previous}' and '{episode.base_name}'. The index keys "
                f"episodes on the lowercase base name; refusing to drop one."
            )
        seen[base_key] = episode.base_name

        episode_rows.append(
            (
                base_key,
                episode.base_name,
                episode.audio_path,
                episode.audio_size,
                episode.number,
                float(episode.mtime),
                position,
            )
        )
        for index, artifact in enumerate(episode.artifacts):
            artifact_rows.append(
                (
                    artifact.path,
                    base_key,
                    artifact.kind,
                    artifact.fmt,
                    int(artifact.size),
                    float(artifact.mtime),
                    index,
                )
            )

    conn.executemany(
        "INSERT INTO episodes "
        "(base_key, base_name, audio_path, audio_size, number, mtime, position) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        episode_rows,
    )
    conn.executemany(
        "INSERT INTO artifacts "
        "(path, base_key, kind, fmt, size, mtime, position) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        artifact_rows,
    )
    return len(artifact_rows)


def _insert_unmatched(conn: sqlite3.Connection, result: ScanResult) -> None:
    """Insert the unmatched paths, with a best-effort size and mtime.

    The scanner reports unmatched entries as bare paths, so size and mtime are
    stat'ed here purely so the UI can show "this 4 MB stray mp4 is ignored".
    A file that vanished between scan and stat stores NULLs rather than
    failing the rescan.
    """
    rows = []
    for position, path in enumerate(result.unmatched):
        try:
            stat = os.stat(path)
            size: Optional[int] = stat.st_size
            mtime: Optional[float] = stat.st_mtime
        except OSError:
            size = None
            mtime = None
        name = os.path.basename(path)
        rows.append((path, name, os.path.splitext(name)[1].lower(), size, mtime, position))

    conn.executemany(
        "INSERT INTO unmatched (path, name, ext, size, mtime, position) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_episodes(
    conn: sqlite3.Connection, search: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Return every indexed episode, in scan order, with its artifact summary.

    Args:
        conn: connection from :func:`init_db`.
        search: optional filter. An episode matches when the term is a
            case-insensitive substring of ``base_name`` **or** equals its
            number exactly. The number half is exact, so "13" matches episode
            13 and not episode 130 - but the name half is still a substring
            search, so "13" also matches ``whycast-episode-130`` by name.
            Substring search is the useful behaviour here; it is documented
            rather than narrowed.

    Returns:
        One dict per episode: identity, audio, ``kinds`` (present kinds in
        :data:`whycast.episodes.ARTIFACT_KINDS` order) and ``formats``
        (kind -> formats on disk), which is what the status matrix renders.
        Artifact rows themselves are not nested here - use :func:`get_episode`
        for those.
    """
    sql = (
        "SELECT base_key, base_name, audio_path, audio_size, number, mtime, position "
        "FROM episodes"
    )
    params: List[Any] = []
    term = (search or "").strip()
    if term:
        sql += " WHERE lower(base_name) LIKE ? ESCAPE '\\' OR CAST(number AS TEXT) = ?"
        params = [f"%{_escape_like(term.lower())}%", term]
    sql += " ORDER BY position"

    with _LOCK:
        rows = conn.execute(sql, params).fetchall()
        artifacts_by_episode = _artifacts_by_episode(conn)

    return [
        _episode_dict(row, artifacts_by_episode.get(row["base_key"], []))
        for row in rows
    ]


def get_episode(conn: sqlite3.Connection, base_name: str) -> Optional[Dict[str, Any]]:
    """Return one episode with its artifacts nested, or None if unknown.

    Lookup is case-insensitive, matching how the scanner groups files: asking
    for ``episode_28`` finds the episode whose audio is ``Episode_28.mp3``.
    """
    key = (base_name or "").strip().lower()
    if not key:
        return None

    with _LOCK:
        row = conn.execute(
            "SELECT base_key, base_name, audio_path, audio_size, number, mtime, position "
            "FROM episodes WHERE base_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        artifacts = conn.execute(
            "SELECT kind, fmt, path, size, mtime, position "
            "FROM artifacts WHERE base_key = ? ORDER BY position",
            (key,),
        ).fetchall()

    artifact_dicts = [dict(a) for a in artifacts]
    pairs = [(a["kind"], a["fmt"]) for a in artifact_dicts]
    episode = _episode_dict(row, pairs)
    # The detail page needs the rows themselves (size, mtime, path per file),
    # so this dict is a superset of what list_episodes returns.
    episode["artifacts"] = artifact_dicts
    return episode


def list_unmatched(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return the files that belong to no episode, in scan order.

    These are visible in the UI on purpose (ADR-008 / TASK-002 AC #3): silently
    ignoring them is how a mis-named artifact disappears without anyone noticing.
    """
    with _LOCK:
        rows = conn.execute(
            "SELECT path, name, ext, size, mtime, position FROM unmatched ORDER BY position"
        ).fetchall()
    return [dict(row) for row in rows]


def get_meta(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Return index metadata: schema version, last scan, counts, db location."""
    with _LOCK:
        last_scan = _read_meta_value(conn, "last_scan")
        podcast_dir = _read_meta_value(conn, "podcast_dir")
        stored_version = _read_meta_value(conn, "schema_version")
        episode_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        artifact_count = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        unmatched_count = conn.execute("SELECT COUNT(*) FROM unmatched").fetchone()[0]
        database_path = _database_path(conn)

    scanned_at = _as_float(last_scan)
    return {
        "schema_version": _as_int(stored_version, SCHEMA_VERSION),
        "last_scan": scanned_at,
        "last_scan_iso": _iso(scanned_at),
        "podcast_dir": podcast_dir,
        "database_path": database_path,
        "episode_count": episode_count,
        "artifact_count": artifact_count,
        "unmatched_count": unmatched_count,
        "artifact_kinds": list(ARTIFACT_KINDS),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _episode_dict(
    row: sqlite3.Row, artifact_pairs: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """Shape one episode row plus its artifacts into an API-ready dict.

    ``artifact_pairs`` is one ``(kind, fmt)`` per *file*, so ``artifact_count``
    counts files (episode_10 has 19 across 8 kinds), not kinds.

    ``formats`` maps kind -> the formats on disk for it, and is what the status
    matrix renders: the overview lists episodes without nesting their artifact
    rows, so a template that reached for ``episode.artifacts`` here would find
    nothing and silently draw every cell as missing.
    """
    unique = {kind for kind, _fmt in artifact_pairs}
    # Order against ARTIFACT_KINDS so this matches Episode.kinds exactly.
    kinds = [kind for kind in ARTIFACT_KINDS if kind in unique]
    # Kinds the scanner produced that are not in the vocabulary cannot happen
    # today, but appending them keeps the count honest if that ever changes.
    kinds += sorted(k for k in unique if k not in _KIND_ORDER)

    formats: Dict[str, List[str]] = {}
    for kind, fmt in artifact_pairs:
        bucket = formats.setdefault(kind, [])
        if fmt not in bucket:
            bucket.append(fmt)
    for bucket in formats.values():
        bucket.sort(key=lambda f: (_FMT_ORDER.get(f, len(_FMT_ORDER)), f))

    return {
        "base_key": row["base_key"],
        "base_name": row["base_name"],
        "number": row["number"],
        "audio_path": row["audio_path"],
        "audio_size": row["audio_size"],
        "has_audio": row["audio_path"] is not None,
        "mtime": row["mtime"],
        "mtime_iso": _iso(row["mtime"]),
        "position": row["position"],
        "artifact_count": len(artifact_pairs),
        "kinds": kinds,
        "formats": formats,
    }


def _artifacts_by_episode(conn: sqlite3.Connection) -> Dict[str, List[Tuple[str, str]]]:
    """Map base_key -> one ``(kind, fmt)`` pair per artifact file, in scan order."""
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for row in conn.execute(
        "SELECT base_key, kind, fmt FROM artifacts ORDER BY base_key, position"
    ):
        grouped.setdefault(row["base_key"], []).append((row["kind"], row["fmt"]))
    return grouped


def _escape_like(text: str) -> str:
    r"""Escape LIKE wildcards so "episode_1" is a literal, not a pattern.

    Base names are full of underscores, and an unescaped ``_`` matches any
    single character - searching "episode_1" would also return "episode-1".
    """
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _read_meta_value(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def _write_meta_value(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def _database_path(conn: sqlite3.Connection) -> Optional[str]:
    """The file backing the 'main' schema; empty string for an in-memory db."""
    for row in conn.execute("PRAGMA database_list"):
        if row["name"] == "main":
            return row["file"]
    return None


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _iso(epoch: Optional[float]) -> Optional[str]:
    """Local-time ISO 8601 string for an epoch timestamp, or None."""
    if epoch is None:
        return None
    try:
        return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat()
    except (OSError, OverflowError, ValueError):
        return None
