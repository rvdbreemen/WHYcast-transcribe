---
id: "ADR-005"
title: "Recursive chunked summarization for long transcripts"
status: "Accepted"
date: "2026-08-23"
binding: false
gate: null
documents_shipped: true
verified_in:
  - "config.py"
  - "transcribe.py"
supersedes: []
superseded_by: null
related:
  - "ADR-012"
topics:
  - "summarization"
  - "chunking"
  - "llm"
  - "context-window"
aliases:
  - "recursive summary"
  - "chunked summarization"
  - "map-reduce summary"
components:
  - "transcribe.summarize_large_transcript"
  - "transcribe.split_into_chunks"
  - "config.MAX_CHUNK_SIZE"
symbols: []
context_scope: "selective"
format: "madr"
llm_judge: true
---

<!-- markdownlint-disable MD025 -->

# ADR-005 Recursive chunked summarization for long transcripts

## Status

Accepted, 2026-08-23.

## Status History

```yaml
status_history:
  - date: 2026-08-23
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Initial proposal
    changed_via: adr-kit
  - date: 2026-08-23
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Documented already-shipped behavior
    changed_via: adr-kit lifecycle
  - date: 2026-08-23
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Auto-accepted already-shipped ADR with verified evidence
    changed_via: adr-kit lifecycle
  - date: 2026-08-31
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-012
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

Full episode transcripts exceed practical single-call limits (up to 200 kB). Commit 'Add recursive summary support' (2025-07-07) added `USE_RECURSIVE_SUMMARIZATION` (default on), `MAX_CHUNK_SIZE=40000`, `CHUNK_OVERLAP=1000` (`config.py:31-33`). `summarize_large_transcript` (`transcribe.py:704`) splits via `split_into_chunks` (`transcribe.py:666`), summarizes each chunk with half the token budget, then summarizes the intermediate summaries. `truncate_transcript` (`transcribe.py:604`) remains as the non-recursive fallback.

## Decision Drivers

* Never silently truncate episode content
* Stay within model input limits
* Keep cost bounded

## Considered Options

* Recursive map-reduce summarization with overlap
* Truncate to MAX_INPUT_TOKENS
* Rely solely on a 1M-context model

## Decision Outcome

Chosen option: **Recursive map-reduce summarization with overlap**, because it preserves coverage of the whole episode independent of model context size, and overlap keeps topic continuity across chunk borders.

### Confirmation

Verified by inspecting `transcribe.py:704-750`; `transcribe.py:666`; `config.py:31-33`.

## Decision Contract

### Must

* Summaries of inputs above `MAX_INPUT_TOKENS` go through `summarize_large_transcript` when `USE_RECURSIVE_SUMMARIZATION` is on
* Chunk size and overlap are config constants

### Must Not

* Do not truncate transcript content as the default summary path

### Exceptions

* None

### Verification

* `transcribe.py:704-750`
* `transcribe.py:666`
* `config.py:31-33`

## Consequences

### Positive

* Complete coverage of long episodes
* Model-agnostic

### Negative

* N+1 API calls per summary; intermediate summaries lose detail (mitigation: overlap, large chunk size)

## Pros and Cons of the Options

### Recursive map-reduce summarization with overlap

* Good, because full coverage
* Bad, because more calls, slower

### Truncate to MAX_INPUT_TOKENS

* Good, because one call
* Bad, because it drops the end of every long episode

### Rely solely on a 1M-context model

* Good, because simplest
* Bad, because it ties the pipeline to one vendor tier; quality degrades on very long inputs

## Open Questions

None.

## Related Decisions

* ADR-003 (model switch by length).

## References

* `config.py`
* `transcribe.py`

