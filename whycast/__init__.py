"""
WHYcast pipeline library (ADR-008).

Importing this package has no side effects beyond reading configuration from
environment variables and .env: no GPU initialization, no model loading, no
filesystem writes. Pipeline modules live under whycast.pipeline and are
imported explicitly by callers (CLI shim, web UI worker, batch scripts).
"""

from whycast.config import VERSION as __version__  # noqa: F401
