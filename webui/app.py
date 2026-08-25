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
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import TemplateNotFound
from starlette.exceptions import HTTPException as StarletteHTTPException

import webui as webui_package
from webui import db, jobs

# The same import :mod:`webui.jobs` makes, for the same reason: every use of
# the shared connection serialises on this lock, which is also what keeps a
# read from landing inside ``rescan``'s open transaction. This module only
# needs it for the one raw query it runs itself (:func:`_queue_or_503`);
# everything else goes through ``db`` or ``jobs``, which take it themselves.
from webui.db import _LOCK as _DB_LOCK
from whycast.episodes import ARTIFACT_FORMATS, ARTIFACT_KINDS
from whycast.errors import ConfigurationError, WhycastError

__all__ = [
    "DEFAULT_ALLOWED_HOSTS",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_PODCAST_DIR",
    "EPISODE_ACTION_TYPES",
    "EPISODE_JOB_HISTORY",
    "JOB_LIST_LIMIT",
    "MAX_JOB_BODY_BYTES",
    "META_KEYS",
    "SSE_HEARTBEAT_SECONDS",
    "SSE_POLL_SECONDS",
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
_ARTIFACT_HEADERS = {
    "Content-Security-Policy": (
        "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

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
    _register_job_routes(app, templates)
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
                "job_types": job_types_payload(),
                "episode_jobs": episode_jobs,
                "active_job": active,
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
            raise HTTPException(status_code=404, detail="Unknown artifact kind")
        if fmt is not None and fmt not in ARTIFACT_FORMATS:
            raise HTTPException(status_code=404, detail="Unknown artifact format")

        episode = _get_episode_or_404(request, base_name)
        artifact = _pick_artifact(episode, kind, fmt)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        path = _safe_file(request.app.state.podcast_dir, artifact.get("path"))
        if path is None:
            raise HTTPException(status_code=404, detail="Artifact not available")

        media_type = _ARTIFACT_MEDIA_TYPES.get(
            artifact.get("fmt"), "text/plain; charset=utf-8"
        )
        # Served as bytes, never decoded: podcasts/ holds artifacts from years
        # of runs and a legacy cp1252 file would otherwise be a 500 instead of
        # a page. The browser applies the charset from the media type.
        return FileResponse(
            path,
            media_type=media_type,
            headers=dict(_ARTIFACT_HEADERS),
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
EPISODE_ACTION_TYPES = ("postprocess", "speakers", "force_episode")


def _episode_actions() -> List[Dict[str, Any]]:
    """The enqueue actions the episode page offers, in :data:`EPISODE_ACTION_TYPES` order."""
    catalogue = {spec["type"]: spec for spec in job_types_payload()}
    return [catalogue[name] for name in EPISODE_ACTION_TYPES if name in catalogue]


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


def _refuse_oversized_body(request: Request) -> None:
    """Reject an over-large job request before its body is read into memory.

    ``await request.body()`` buffers the whole thing, and whatever is in
    ``params`` then gets serialised into the ``jobs`` table - which lives in the
    same SQLite file as the episode index the UI depends on. A 32 MB parameter
    string was accepted and grew the database to 41 MB. A job request is a job
    type, an episode key and a handful of options; nothing legitimate comes
    close to the cap.
    """
    raw = request.headers.get("content-length")
    if not raw:
        return
    try:
        length = int(raw)
    except ValueError:
        return
    if length > MAX_JOB_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Job request body is larger than {MAX_JOB_BODY_BYTES} bytes",
        )


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
    _refuse_oversized_body(request)
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    if content_type.lower() == "application/x-www-form-urlencoded":
        form = await request.form()
        payload: Dict[str, Any] = {
            key: value for key, value in form.items() if isinstance(value, str)
        }
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

    body = await request.body()
    if len(body) > MAX_JOB_BODY_BYTES:
        # A chunked body carries no Content-Length, so the header check above
        # cannot see it; this is the one that always holds.
        raise HTTPException(
            status_code=413,
            detail=f"Job request body is larger than {MAX_JOB_BODY_BYTES} bytes",
        )
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
