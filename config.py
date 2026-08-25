"""
Compatibility shim (ADR-008): configuration moved to whycast/config.py.

Existing imports (`from config import VERSION, ...`) keep working unchanged.
New code should import from whycast.config directly.
"""
from whycast.config import *  # noqa: F401,F403
