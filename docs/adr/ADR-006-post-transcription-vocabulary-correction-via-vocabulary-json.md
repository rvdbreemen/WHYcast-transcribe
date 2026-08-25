---
id: "ADR-006"
title: "Post-transcription vocabulary correction via vocabulary.json"
status: "Accepted"
date: "2026-08-23"
binding: false
gate: null
documents_shipped: true
verified_in:
  - "vocabulary.json"
  - "transcribe.py"
supersedes: []
superseded_by: null
topics:
  - "vocabulary"
  - "transcription"
  - "post-processing"
  - "quality"
aliases:
  - "vocabulary.json"
  - "term correction"
  - "WHY misspellings"
components:
  - "transcribe.load_vocabulary_mappings"
  - "transcribe.apply_vocabulary_corrections"
  - "vocabulary.json"
symbols: []
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-006 Post-transcription vocabulary correction via vocabulary.json

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
```

## Context and Problem Statement

Whisper consistently mishears domain terms (WHY2025 -> 'Y2025', WHYcast -> 'WAIcast'). Rather than tuning Whisper's `initial_prompt`, the project keeps a JSON map of `{wrong: right}` in `vocabulary.json` (the most-churned data file per `adr-discover`). `apply_vocabulary_corrections` (`transcribe.py:1754`) applies longest-first, word-boundary, case-insensitive regex replacement after transcription; `USE_CUSTOM_VOCABULARY` (`config.py:55`) toggles it.

## Decision Drivers

* Fix recurring domain-term errors deterministically
* Non-developers can add terms
* Independent of Whisper model version

## Considered Options

* Post-hoc JSON replacement map
* Whisper initial_prompt with vocabulary
* LLM cleanup prompt handles terms

## Decision Outcome

Chosen option: **Post-hoc JSON replacement map**, because it is deterministic, testable, editable without code, and survives model swaps; prompt-based hints are probabilistic and limited in length.

### Confirmation

Verified by inspecting `transcribe.py:1690-1800`; `vocabulary.json`; `config.py:55-56`.

## Decision Contract

### Must

* Domain term fixes go into `vocabulary.json`, not into code or prompts
* Replacement is word-boundary and longest-match-first

### Must Not

* Do not hard-code term replacements in `transcribe.py`

### Exceptions

* None

### Verification

* `transcribe.py:1690-1800`
* `vocabulary.json`
* `config.py:55-56`

## Consequences

### Positive

* Deterministic, reviewable diffs in git
* Zero token cost

### Negative

* Regex replacement can over-match short keys like `Y` (mitigation: word boundaries; review short entries)

## Pros and Cons of the Options

### Post-hoc JSON replacement map

* Good, because deterministic
* Bad, because blind to context

### Whisper initial_prompt with vocabulary

* Good, because it fixes at the source
* Bad, because ~224 token limit, probabilistic

### LLM cleanup prompt handles terms

* Good, because context-aware
* Bad, because inconsistent across runs, costs tokens

## Open Questions

None.

## Related Decisions

* ADR-001 (corrects Whisper output).

## References

* `vocabulary.json`
* `transcribe.py`
* `config.py`

## Enforcement

```json
{
  "forbid_import": [],
  "forbid_pattern": [],
  "require_pattern": []
}
```
