---
id: "ADR-003"
title: "Per-task OpenAI model selection"
status: "Superseded"
date: "2026-08-31"
binding: false
gate: null
documents_shipped: true
verified_in:
  - "config.py"
  - "transcribe.py"
supersedes: []
superseded_by: "ADR-012"
topics:
  - "openai"
  - "llm"
  - "models"
  - "cost"
aliases:
  - "model selection"
  - "gpt-4.1"
  - "o4-mini"
components:
  - "config.OPENAI_MODEL"
  - "config.OPENAI_SPEAKER_MODEL"
  - "transcribe.choose_appropriate_model"
symbols: []
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-003 Per-task OpenAI model selection

## Status

Superseded by ADR-012, 2026-08-31.

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
    status: Superseded
    changed_by: "User: Robert van den Breemen"
    reason: "Superseded by ADR-012: the four model slots collapse to two by default, and the parameter contract for gpt-5.6 removes temperature as a tuning knob"
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

The post-processing pipeline makes several OpenAI calls with different needs: cleanup/summary/blog need a large context window; speaker analysis needs reasoning. `config.py:17-20` defines four model env-vars (`OPENAI_MODEL`, `OPENAI_LARGE_CONTEXT_MODEL`, `OPENAI_HISTORY_MODEL` default `gpt-4.1`; `OPENAI_SPEAKER_MODEL` default `o4-mini`). `choose_appropriate_model` (`transcribe.py:629`) switches to the large-context model above `MAX_INPUT_TOKENS`.

## Decision Drivers

* Reasoning model for speaker attribution, cheaper general model elsewhere
* Swap models without code changes
* Handle 200 kB transcripts

## Considered Options

* Per-task model env-vars with length-based switch
* One global model for everything
* Local LLM (Ollama) for all steps

## Decision Outcome

Chosen option: **Per-task model env-vars with length-based switch**, because each step has a different cost/quality profile and the maintainer iterates on prompts frequently; env-vars make model swaps a zero-code operation.

### Confirmation

Verified by inspecting `config.py:17-26`; `transcribe.py:629` choose_appropriate_model; `transcribe.py:942` analyze_speakers_with_o4.

## Decision Contract

### Must

* Every OpenAI call takes its model from a `config.py` `OPENAI_*_MODEL` constant
* Long inputs route through `choose_appropriate_model`

### Must Not

* Do not hard-code a model name string at `transcribe.py` call sites

### Exceptions

* None

### Verification

* `config.py:17-26`
* `transcribe.py:629` choose_appropriate_model
* `transcribe.py:942` analyze_speakers_with_o4

## Consequences

### Positive

* Cheap to experiment with new models
* Reasoning model only where it pays off

### Negative

* Four knobs to document; defaults go stale as models are retired (mitigation: defaults live in one file)

## Pros and Cons of the Options

### Per-task model env-vars with length-based switch

* Good, because fine-grained cost control
* Bad, because more configuration surface

### One global model for everything

* Good, because simple
* Bad, because it pays reasoning-model prices for cleanup, or loses speaker quality

### Local LLM (Ollama) for all steps

* Good, because free and private
* Bad, because quality on long Dutch transcripts is unproven; not evaluated

## Open Questions

None.

## Related Decisions

* None.

## References

* `config.py`
* `transcribe.py`

## Enforcement

```json
{
  "forbid_import": [],
  "forbid_pattern": [
    {
      "pattern": "model\\s*=\\s*[\"'](gpt-|o[0-9])",
      "path_glob": "transcribe.py",
      "message": "Model names come from config.OPENAI_*_MODEL, not literals (ADR-003)"
    }
  ],
  "require_pattern": []
}
```
