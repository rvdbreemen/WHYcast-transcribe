"""
Entry point for the WHYcast web UI: ``python -m webui``.

Starts uvicorn on the FastAPI app in :mod:`webui.app`. Bound to 127.0.0.1
unless the operator deliberately says otherwise (ADR-008 Decision Contract:
"The web server must bind to 127.0.0.1 by default").

Command-line flags are a convenience over the ADR-007 environment variables,
not a second configuration store: each flag sets the variable it mirrors before
the app is imported, so ``WHYCAST_PODCAST_DIR=... python -m webui`` and
``python -m webui --podcast-dir ...`` end in exactly the same state.

**Import order is the whole contract here.** :mod:`webui.app` runs
``app = create_app()`` at module level, which reads those variables *once*, at
import time. This module therefore imports :mod:`webui.app` inside
:func:`main`, after the flags have been written into ``os.environ`` - never at
the top of the file. It used to import it at the top, and the result was that
``python -m webui --podcast-dir X --db Y`` silently served the real production
index and podcast directory while the startup banner reported X and Y, because
the banner was recomputed from the freshly-set environment and the app was not.
A "scratch" instance that validates episode names against the real index and
enqueues into the real queue is not a scratch instance.

Two habits keep it honest for the next reader:

* nothing from :mod:`webui.app` may be imported at module scope (the parser's
  ``--help`` text takes :data:`webui.DEFAULT_HOST` / :data:`webui.DEFAULT_PORT`
  from the package, which imports nothing);
* the banner prints what the *served application object* actually holds, read
  back from ``app.state``, not a second computation that can agree with the
  intent while the app disagrees with both.

    python -m webui
    python -m webui --port 9000 --podcast-dir D:/podcasts
    python -m webui --reload            # development

This is the one place allowed to exit: it is a CLI, not library code (ADR-008
forbids ``exit()`` in ``whycast/``, and the web layer keeps to the same rule
everywhere except here).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

# Only the package - it imports nothing and builds nothing. Everything from
# webui.app is imported inside main(), after the flags reach os.environ; see the
# module docstring.
from webui import DEFAULT_HOST, DEFAULT_PORT
from whycast.errors import ConfigurationError

logger = logging.getLogger("webui")

EXIT_OK = 0
EXIT_CONFIG = 2


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the web UI server."""
    parser = argparse.ArgumentParser(
        prog="python -m webui",
        description="Serve the WHYcast web UI (read-only episode browser, ADR-008).",
    )
    parser.add_argument(
        "--host",
        default=None,
        metavar="ADDRESS",
        help=(
            f"Bind address (default: $WHYCAST_WEBUI_HOST or {DEFAULT_HOST}). "
            "Anything other than loopback exposes the UI to your network."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help=f"Bind port (default: $WHYCAST_WEBUI_PORT or {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--podcast-dir",
        default=None,
        metavar="DIR",
        help="Directory to index (default: $WHYCAST_PODCAST_DIR or ./podcasts).",
    )
    parser.add_argument(
        "--db",
        default=None,
        metavar="FILE",
        help=(
            "SQLite index file (default: $WHYCAST_WEBUI_DB or "
            "webui/whycast_webui.db). The index is disposable: delete it and "
            "it is rebuilt from the podcast directory."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart on source changes (development only).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        help="Uvicorn log level (default: info).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the server. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Flags win over the environment; both feed the same variables, which the
    # app reads when uvicorn imports it (including in a --reload child).
    if args.podcast_dir:
        os.environ["WHYCAST_PODCAST_DIR"] = os.path.abspath(args.podcast_dir)
    if args.db:
        os.environ["WHYCAST_WEBUI_DB"] = os.path.abspath(args.db)
    if args.host:
        os.environ["WHYCAST_WEBUI_HOST"] = args.host
    if args.port is not None:
        os.environ["WHYCAST_WEBUI_PORT"] = str(args.port)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Now, and not one line earlier: importing this module builds the
    # application from the environment we have just finished writing.
    from webui import app as app_module

    try:
        host = app_module.host_from_env()
        port = app_module.port_from_env()
    except ConfigurationError as exc:
        parser.error(str(exc))  # exits with status 2
        return EXIT_CONFIG  # pragma: no cover - parser.error does not return

    if not 1 <= port <= 65535:
        parser.error(f"--port out of range: {port}")

    # Rebuild and rebind the served object rather than trusting whatever was
    # constructed at import time. In a fresh process the two are identical (the
    # environment was already set); when something imported webui.app earlier -
    # a test, an embedding script - this is what makes the flags authoritative
    # instead of "whoever imported first wins". uvicorn.run("webui.app:app")
    # resolves the attribute at call time, so it picks this up, and a --reload
    # child re-imports with the same environment and lands on the same state.
    app_module.app = app_module.create_app()
    served = app_module.app

    # Derive the Host allowlist from the address we are about to bind, not from
    # a second read of the environment. Those two disagreed before: with
    # WHYCAST_WEBUI_HOST=0.0.0.0 in .env and `--host 127.0.0.1` on the command
    # line, the process bound loopback (safe) and enforced allowed_hosts=["*"]
    # (fail open) - any web page could then reach the UI as same-origin, which
    # also collapsed the cross-origin check, because that derives the expected
    # origin from the Host header it just accepted. A --reload child recomputes
    # this from WHYCAST_WEBUI_HOST, which main() has already set to `host`, so
    # both paths land on the same list.
    served.state.allowed_hosts = app_module.allowed_hosts_from_env(host)

    # Everything below reports what the application actually holds. The banner
    # used to recompute these from the environment, which is how it managed to
    # confirm flags that had been ignored.
    podcast_dir = served.state.podcast_dir
    db_path = served.state.db_path
    allowed_hosts = list(served.state.allowed_hosts)

    if not os.path.isdir(podcast_dir):
        # Not fatal: the UI boots empty and says so, which beats a stack trace.
        logger.warning(
            "Podcast directory does not exist: %s - the overview will be empty",
            podcast_dir,
        )
    if not app_module.is_loopback_host(host):
        logger.warning(
            "Binding to %s, which is reachable from your network. The web UI has "
            "no authentication (ADR-008 assumes localhost, single user).",
            host,
        )
    if "*" in allowed_hosts:
        # Off-loopback, or an explicit "*": no Host allowlist to protect
        # against DNS rebinding. Say so rather than implying it is enforced.
        logger.warning("Host header check disabled - every Host value is accepted.")
    else:
        logger.info(
            "Accepting requests for Host: %s (set WHYCAST_WEBUI_ALLOWED_HOSTS "
            "to reach the UI under another name).",
            ", ".join(allowed_hosts),
        )

    try:
        import uvicorn
    except ImportError:
        parser.error(
            "uvicorn is not installed. Install the web UI dependencies: "
            "pip install fastapi uvicorn jinja2"
        )
        return EXIT_CONFIG  # pragma: no cover - parser.error does not return

    logger.info("Serving %s on http://%s:%s", podcast_dir, host, port)
    logger.info("Index: %s", db_path)
    uvicorn.run(
        "webui.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
