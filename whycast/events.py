"""
Progress events for the WHYcast pipeline (ADR-008).

Pipeline code emits ProgressEvent objects through a context-local sink instead
of calling print(). The CLI installs a ConsoleSink to reproduce the old console
output; the web UI worker installs a JSONL sink. Library code only ever calls
emit() - it never prints and never knows who is listening.

Usage inside pipeline code:

    from whycast.events import emit
    emit("diarization", "Running speaker diarization ...")

Installing a sink (CLI, worker, tests):

    from whycast.events import ConsoleSink, use_sink
    with use_sink(ConsoleSink()):
        full_workflow(...)
"""

import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class ProgressEvent:
    """A single progress or status message from a pipeline step."""

    step: str                       # e.g. "transcription", "diarization", "summary"
    message: str                    # human-readable text (the old print() line)
    level: str = "info"             # "info" | "warning" | "error"
    progress: Optional[float] = None  # 0.0-1.0 within the step, when known
    data: dict = field(default_factory=dict)  # structured extras (counts, paths)
    timestamp: float = field(default_factory=time.time)


class EventSink(Protocol):
    def emit(self, event: ProgressEvent) -> None: ...


class NullSink:
    """Discards all events. The default when no sink is installed."""

    def emit(self, event: ProgressEvent) -> None:
        pass


class ConsoleSink:
    """Prints event messages verbatim, reproducing the legacy CLI output.

    On Windows a redirected stdout may use a legacy code page (cp1252) that
    cannot encode the emoji in pipeline messages; fall back to replacement
    characters instead of crashing the run.
    """

    def emit(self, event: ProgressEvent) -> None:
        try:
            print(event.message)
        except UnicodeEncodeError:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(event.message.encode(encoding, errors="replace").decode(encoding))


class CollectSink:
    """Collects events in a list; useful in tests."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


_current_sink: ContextVar[EventSink] = ContextVar("whycast_event_sink", default=NullSink())


def emit(step: str, message: str, level: str = "info",
         progress: Optional[float] = None, **data) -> None:
    """Emit a progress event to the currently installed sink."""
    _current_sink.get().emit(
        ProgressEvent(step=step, message=message, level=level,
                      progress=progress, data=data)
    )


def set_sink(sink: EventSink) -> None:
    """Install a sink for the current context (process-wide in practice)."""
    _current_sink.set(sink)


@contextmanager
def use_sink(sink: EventSink):
    """Temporarily install a sink; restores the previous one afterwards."""
    token = _current_sink.set(sink)
    try:
        yield sink
    finally:
        _current_sink.reset(token)
