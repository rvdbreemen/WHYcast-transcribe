"""
Exceptions for the WHYcast pipeline library (ADR-008).

Library code raises exceptions instead of calling exit(); only the CLI shim
and the web UI worker translate them into exit codes or failed-job states.
"""


class WhycastError(Exception):
    """Base class for all WHYcast pipeline errors."""


class SecurityError(WhycastError):
    """Raised when a security-related validation fails."""


class ConfigurationError(WhycastError):
    """Raised when configuration is missing or invalid (e.g. missing API key)."""


class DependencyError(WhycastError):
    """Raised when a required optional dependency is not installed."""


class PipelineError(WhycastError):
    """Raised when a pipeline step fails in a way that ends the run."""


class EpisodeNotFoundError(PipelineError):
    """Raised when no episode could be fetched or located."""
