---
id: "ADR-008"
title: "Web UI as FastAPI layer over extracted pipeline library"
status: "Accepted"
date: "2026-08-24"
binding: false
gate: null
documents_shipped: false
verified_in: []
supersedes: []
superseded_by: null
related:
  - "ADR-001"
  - "ADR-002"
  - "ADR-007"
  - "ADR-009"
  - "ADR-011"
topics:
  - "web-ui"
  - "pipeline-architecture"
  - "job-orchestration"
  - "refactoring"
aliases:
  - "webui"
  - "web interface"
  - "dashboard"
  - "whycast package"
components:
  - "whycast"
  - "webui"
  - "worker"
symbols:
  - "full_workflow"
  - "process_transcript_workflow"
  - "ProgressEvent"
  - "EventSink"
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-008 Web UI as FastAPI layer over extracted pipeline library

## Status

Accepted, 2026-08-24.

**Decision Maker:** User: Robert van den Breemen (approved the plan in the 2026-08-24 planning session; stack and phasing proposed by agent, accepted by user).

## Status History

```yaml
status_history:
  - date: 2026-08-24
    status: Proposed
    changed_by: Robert van den Breemen
    reason: Initial proposal
    changed_via: adr-kit
  - date: 2026-08-24
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-001
    changed_via: adr-kit lifecycle
  - date: 2026-08-24
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-002
    changed_via: adr-kit lifecycle
  - date: 2026-08-24
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-007
    changed_via: adr-kit lifecycle
  - date: 2026-08-24
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Accepted by Robert via explicit choice in planning session 2026-08-24
    changed_via: adr-kit lifecycle
  - date: 2026-08-25
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-009
    changed_via: adr-kit lifecycle
  - date: 2026-08-26
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-011
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

The project needs a local web interface to browse episodes, start and follow pipeline runs, and support correction workflows (speaker names, vocabulary, prompts). Today the pipeline lives in `transcribe.py`, a 3129-line monolith (`wc -l`, 2026-08-24) in which orchestration, `print()` console output, and `exit()` calls are interwoven (e.g. `transcribe.py:2565` `full_workflow()`, `transcribe.py:3059` CLI (command-line interface) entry). It cannot be imported as a service: importing triggers side effects, progress exists only as console text, and failure paths call `exit()`.

Constraints that shape the solution:

* One NVIDIA CUDA (Compute Unified Device Architecture) GPU (graphics processing unit). Transcription (faster-whisper, ADR-001) and diarization (pyannote, ADR-002) both claim VRAM (video memory); concurrent GPU jobs are not possible. Jobs run tens of minutes per episode.
* The filesystem is the source of truth: `podcasts/` holds ~48 episodes across 812 files with inconsistent naming; artifacts are `_transcript`, `_ts`, `_cleaned`, `_summary`, `_blog`, `_history`, `_assignment` in `.txt`/`.html`/`.wiki`.
* Configuration is environment variables plus file-based prompts (ADR-007). Secrets (`OPENAI_API_KEY`, HuggingFace token) live in `.env` and must never be exposed by a web layer.
* Single user, single Windows/CUDA machine, localhost use. No multi-tenant or cloud requirements.
* The existing CLI (`python transcribe.py ...`) must keep working during and after the change.

## Decision Drivers

* The web layer needs in-process access to pipeline steps with structured progress events; wrapping the CLI in a subprocess yields no step granularity and no typed errors.
* One GPU forces strictly serial GPU job execution; the design must make over-scheduling impossible rather than merely discouraged.
* Crash isolation: a CUDA out-of-memory error in a job must not take down the web server.
* Preference for simple, explainable solutions over infrastructure: one machine, one user, no external services.
* ADRs 001-007 must remain valid; this decision layers on top of them without modifying them.

## Considered Options

* Option A: Extract a `whycast` pipeline library; FastAPI app + serial worker subprocess + SQLite (index/queue) + Jinja2/HTMX frontend with SSE.
* Option B: Flask (or FastAPI) app that shells out to the existing `transcribe.py` CLI per job.
* Option C: Full SPA (Single-Page Application; Vite + React/Svelte) frontend against a Python API, extraction as in A.
* Option D: Do nothing; keep CLI plus the existing batch scripts (`batch_update_speaker_assignments.py`, `quick_batch_update.py`).

## Decision Outcome

Chosen option: **Option A**, because only library extraction gives the web layer structured progress, typed errors, and per-step invocation, while the serial worker subprocess matches the one-GPU reality and isolates crashes. SQLite and HTMX keep the stack to the Python ecosystem already in use, with no build toolchain and no external services.

Concretely:

1. Pipeline code moves from `transcribe.py` into a `whycast/` package (steps: feed, audio, transcription, diarization, speakers, vocabulary, postprocess, outputs; orchestrator: `workflow.py`). Every `print()` becomes an event on an `EventSink` protocol plus `logging`; every `exit()` becomes an exception. `transcribe.py` remains as a thin CLI shim with a console sink, producing byte-identical outputs (golden-master test).
2. The web UI is a FastAPI app (`webui/`) serving Jinja2/HTMX pages and a JSON API (Application Programming Interface). Long-running jobs are queued in SQLite and executed by a single separate worker process, one job at a time; job progress is written as JSON-lines event files and streamed to the browser via SSE (Server-Sent Events).
3. The filesystem stays authoritative; SQLite holds only a rebuildable episode index and the job queue.

### Confirmation

* Golden-master test in `tests/`: one reference episode processed through the pre-refactor CLI and through the extracted library; all output artifacts must be identical.
* End-to-end check: one full episode processed start-to-finish from the web UI, with the browser closed and reopened mid-run and no loss of progress display.

## Decision Contract

### Must

* Pipeline code must be importable (`import whycast`) without side effects: no GPU initialization, no model loading, no filesystem writes at import time.
* Pipeline progress and errors must flow through the event sink and `logging`; library code paths must raise exceptions instead of calling `exit()`.
* The web server must bind to `127.0.0.1` by default.
* At most one GPU job may run at a time, enforced by the single worker process.
* The filesystem remains the source of truth; the SQLite database must be deletable and rebuildable from `podcasts/` without data loss.
* Any API surface that exposes configuration must use an allowlist of visible keys; secrets (`OPENAI_API_KEY`, HuggingFace token) must never be returned, masked or otherwise.

### Must Not

* No external job brokers or services (Celery, Redis, RabbitMQ, cloud queues).
* No `exit()`/`sys.exit()` in `whycast/` library code.
* The web layer must not write episode artifacts directly; all artifact writes go through the pipeline library, which writes atomically (write-to-temp, then rename) and backs up files it overwrites.

### Exceptions

* None.

### Verification

* Golden-master diff test in `tests/` (added in phase 0).
* Enforcement patterns below (`exit(` forbidden in `whycast/`).
* Manual: `python -c "import whycast"` completes without GPU or network activity.

## Consequences

### Positive

* Web UI, CLI, and batch scripts all drive one library; the batch scripts' current fragile imports into the monolith (see graphify report: `batch_update_speaker_assignments()` -> `speaker_assignment_step()`, INFERRED edges) can be replaced by supported calls.
* Per-step invocation enables cheap re-runs: LLM (Large Language Model)-only post-processing without re-transcribing (GPU minutes saved per iteration).
* Jobs survive browser disconnects; event logs on disk make failed runs debuggable after the fact.
* Crash isolation: CUDA OOM (out-of-memory) kills the worker's job subprocess, not the web server.

### Negative

* Phase 0 (extraction) is roughly 60% of the total effort before any visible UI exists (estimate, not a measurement). Mitigation: golden-master test gates every extraction merge; the CLI shim keeps the tool usable throughout.
* Refactoring a 3129-line file risks subtle behavior changes. Mitigation: byte-identical golden-master comparison; mechanical transformation rules (print -> event, exit -> exception) rather than redesign.
* Two processes (server + worker) instead of one script adds operational surface on Windows. Mitigation: a single `run` entry point that starts both; Task Scheduler integration deferred to phase 4.
* HTMX may prove too limited for a rich transcript editor in phase 3. Mitigation: escape hatch is a single Vite island for the editor only; the decision to add one would amend this ADR.

## Pros and Cons of the Options

### Option A: FastAPI + serial worker + SQLite + HTMX over extracted library

* Good, because structured progress, typed errors, and per-step calls come from in-process library use.
* Good, because SSE + JSONL event files give live progress that survives reconnects with no message broker.
* Good, because the whole stack is Python plus one template layer; no frontend build toolchain.
* Bad, because it requires the phase-0 extraction investment before the UI can exist.

### Option B: Web app shelling out to the existing CLI

* Good, because it needs no refactoring; fastest first screenshot.
* Bad, because progress is unparseable console text with emoji; no step granularity, no typed errors.
* Bad, because the CLI calls `exit()` and deletes files on `--force`; driving it programmatically from a server is fragile.
* Bad, because every future feature (per-step re-run, speaker editor) would still require the extraction later, after building throwaway plumbing.

### Option C: Full SPA (Single-Page Application; Vite + React/Svelte) + Python API

* Good, because it is the strongest foundation for a rich editor UI.
* Bad, because ~80% of the UI is lists and rendered text, which does not need a SPA.
* Bad, because it adds a Node build toolchain and dependency surface to a Python project for uncertain benefit; conflicts with the simplicity preference.

### Option D: Do nothing

* Good, because zero effort and zero risk.
* Bad, because episode status is invisible (812 loosely named files), runs are console-only, and corrections require hand-run batch scripts wired into monolith internals.

## Open Questions

- [x] Should v1 run all jobs strictly serially, or allow LLM-only jobs in parallel with a GPU job? (Proposal on the table: strictly serial.) — **Answered 2026-08-24 by User: Robert van den Breemen:** Strictly serial in v1: one job at a time, GPU or not. Decided by Robert van den Breemen, 2026-08-24. A parallel LLM lane can be added later via an amending ADR.
- [x] Phase 3 editing granularity: mapping-level speaker renames only, or also segment-level reassignment? (Proposal: mapping-level first; covers most corrections.) — **Answered 2026-08-24 by User: Robert van den Breemen:** Mapping-level renames only in phase 3; covers most corrections at a fraction of the UI cost. Segment-level reassignment deferred; would be an amendment. Decided by Robert van den Breemen, 2026-08-24.

## Related Decisions

* **ADR-001 (faster-whisper on CUDA)** and **ADR-002 (pyannote diarization)**: the shared single-GPU stack is why the worker is strictly serial.
* **ADR-003 (per-task OpenAI model selection)**: unchanged; the web UI selects pipeline steps, not models.
* **ADR-007 (env-var configuration and file-based prompts)**: the web UI reads the same env-var config and edits the same prompt files; it introduces no parallel configuration store.

## References

* `transcribe.py:2565` (`full_workflow()`), `transcribe.py:3059` (CLI entry) — the monolith being extracted.
* `config.py` — env-var configuration surface the web UI must respect (ADR-007).
* `graphify-out/GRAPH_REPORT.md` (2026-08-23) — dependency graph evidence: 14 communities, batch scripts coupled to monolith internals via INFERRED call edges.
* Planning session 2026-08-24: phased plan (0: extraction, 1: read-only browser, 2: jobs, 3: correction workflows, 4: polish).

## Enforcement

```json
{
  "forbid_import": [],
  "forbid_pattern": [
    {
      "pattern": "^\\s*(sys\\.)?exit\\([^)]*\\)\\s*(#.*)?$",
      "path_glob": "whycast/**/*.py",
      "message": "Library code must raise exceptions, not exit() (ADR-008 Decision Contract)"
    },
    {
      "pattern": "^\\s*print\\(.*\\)\\s*(#.*)?$",
      "path_glob": "whycast/pipeline/**/*.py",
      "message": "Pipeline steps must emit events via EventSink and logging, not print() (ADR-008)"
    }
  ],
  "require_pattern": []
}
```
