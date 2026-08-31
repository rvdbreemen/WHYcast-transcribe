"""
Optional-dependency shims for the WHYcast pipeline (ADR-008).

Mirrors the legacy top-of-transcribe.py try/except import blocks, but without
sys.exit() and without logging at import time.

What each shim does when its package is missing, because they differ:

* ``openai`` -> ``OpenAI`` is None. Every caller checks ``openai_available``
  first and fails with a ConfigurationError naming the package, rather than
  letting "'NoneType' object is not callable" surface halfway through a run.
  ``speakers.py`` additionally skips the step and returns None.
* ``feedparser`` -> ``feedparser`` is None. ``pipeline/feed.py`` checks
  ``feedparser_available`` before parsing and raises ConfigurationError.
* ``tqdm`` -> replaced by the fallback class below, which forwards progress to
  the event sink. Nothing has to check ``tqdm_available``; the substitute is a
  real implementation, not a None.

The flags are the contract. Adding a shim without checking its flag at the call
site reintroduces exactly the failure this module exists to prevent.
"""

openai_available = False
feedparser_available = False
tqdm_available = False

try:
    from openai import OpenAI, BadRequestError  # noqa: F401
    openai_available = True
except ImportError:
    OpenAI = None  # type: ignore[assignment]

    class BadRequestError(Exception):  # type: ignore[no-redef]
        pass

try:
    import feedparser  # noqa: F401
    feedparser_available = True
except ImportError:
    feedparser = None  # type: ignore[assignment]

try:
    from tqdm import tqdm  # noqa: F401
    tqdm_available = True
except ImportError:
    # Simple fallback for tqdm if not available (legacy behavior preserved)
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = len(iterable) if iterable is not None else 0
            self.n = 0
            self.desc = kwargs.get('desc', '')

        def __iter__(self):
            # Progress goes through the event sink, never to stdout: this is
            # library code (ADR-008 Decision Contract). The import is local so
            # this shim stays importable on its own and costs nothing when the
            # real tqdm is installed.
            from whycast.events import emit

            for obj in self.iterable:
                yield obj
                self.n += 1
                if self.n % 10 == 0:
                    emit(
                        self.desc or "progress",
                        f"{self.desc}: {self.n}/{self.total}",
                        progress=(self.n / self.total) if self.total else None,
                        current=self.n,
                        total=self.total,
                    )

        def update(self, n=1):
            self.n += n

        def close(self):
            pass
