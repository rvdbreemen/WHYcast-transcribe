"""
WHYcast web UI (ADR-008).

The FastAPI layer over the extracted ``whycast`` pipeline library. This package
is a thin presentation and orchestration shell:

* ``webui.db``     - the rebuildable SQLite index over ``podcasts/``
* ``webui.app``    - FastAPI + Jinja2/HTMX pages and the JSON API (phase 1)
* ``webui.worker`` - the single serial job runner (phase 2)

Importing this package has no side effects: no database connection, no server,
no filesystem writes. Connections are created explicitly by callers via
:func:`webui.db.init_db`.

The two bind defaults live here rather than in :mod:`webui.app` because
``python -m webui`` needs them to build its argument parser *before* it may
import the app: importing :mod:`webui.app` runs ``app = create_app()``, which
reads the ADR-007 environment variables, and the command-line flags have to be
written into the environment first or they are silently ignored. This package
module imports nothing, so ``from webui import DEFAULT_HOST`` is free.
:mod:`webui.app` re-exports both names, so the app remains the single place to
look for them.
"""

#: Bind address the web UI defaults to. Loopback, always: ADR-008 assumes
#: localhost with a single user and no authentication.
DEFAULT_HOST = "127.0.0.1"

#: Bind port the web UI defaults to.
DEFAULT_PORT = 8420
