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


class SpeakerMappingError(PipelineError):
    """Raised when ``<base>_speakers.json`` exists but must not be used as is.

    Two cases, both of which need a person to look at the file (ADR-010):

    * it does not parse, or holds something that is not a mapping - a typo in a
      hand-edited file;
    * it is stale: it records the fingerprint of a different transcript than the
      one being processed, so a re-transcription has almost certainly renumbered
      the ``SPEAKER_xx`` clusters underneath it.

    Neither may fall back to the paid analysis silently. A silent fallback hides
    the typo, costs money, and - in the stale case - would happily put one
    person's words in another's mouth. So this ends the step instead, and the
    web UI turns it into a failed job whose message names the file to fix.
    """
