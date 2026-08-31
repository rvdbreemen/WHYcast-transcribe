---
id: "ADR-007"
title: "Environment-variable configuration and file-based prompts"
status: "Accepted"
date: "2026-08-23"
binding: false
gate: null
documents_shipped: true
verified_in:
  - "config.py"
  - "prompts/summary_prompt.txt"
supersedes: []
superseded_by: null
related:
  - "ADR-008"
  - "ADR-012"
topics:
  - "configuration"
  - "prompts"
  - "env-vars"
  - "conventions"
aliases:
  - "config.py"
  - "prompts directory"
  - ".env"
components:
  - "config"
  - "prompts/"
  - "transcribe.read_prompt_file"
symbols: []
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-007 Environment-variable configuration and file-based prompts

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
  - date: 2026-08-24
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-008
    changed_via: adr-kit lifecycle
  - date: 2026-08-31
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-012
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

All tunables live in `config.py` as module constants read from `os.environ` with defaults (`python-dotenv` loads `.env`). Every LLM prompt is a plain text file under `prompts/` referenced by `PROMPT_*_FILE` constants (`config.py:47-52`) and loaded via `read_prompt_file` (`transcribe.py:575`). Prompts are iterated on far more often than code (commit 30cd91a).

## Decision Drivers

* Tune prompts without touching Python
* Per-machine settings (GPU, models) without code edits
* Secrets (OpenAI, HF token) stay out of git

## Considered Options

* Env-vars + config.py constants + prompts/*.txt
* CLI flags for everything
* YAML/TOML config file with embedded prompts

## Decision Outcome

Chosen option: **Env-vars + config.py constants + prompts/*.txt**, because it keeps secrets in `.env`, machine settings in env, and prompts as diffable text; zero parsing code.

### Confirmation

Verified by inspecting `config.py`; `transcribe.py:575`; `prompts/`.

## Decision Contract

### Must

* New tunables are added to `config.py` as `os.environ.get` with a default
* New prompts are files in `prompts/` with a `PROMPT_*_FILE` constant

### Must Not

* Do not embed multi-line prompt text in Python source
* Do not read `os.environ` outside `config.py` except for secrets

### Exceptions

* None

### Verification

* `config.py`
* `transcribe.py:575`
* `prompts/`

## Consequences

### Positive

* Prompt edits are reviewable text diffs
* Secrets never committed

### Negative

* No schema validation of env values (mitigation: typed casts with defaults in `config.py`)

## Pros and Cons of the Options

### Env-vars + config.py constants + prompts/*.txt

* Good, because simple, diffable
* Bad, because no validation

### CLI flags for everything

* Good, because discoverable via --help
* Bad, because unwieldy for ~25 knobs and long prompts

### YAML/TOML config file with embedded prompts

* Good, because single file
* Bad, because multi-line prompts in YAML are awkward; adds a parser

## Open Questions

None.

## Related Decisions

* ADR-003 (model env-vars are an instance of this convention).

## References

* `config.py`
* `prompts/`
* `transcribe.py`

## Enforcement

```json
{
  "forbid_import": [],
  "forbid_pattern": [],
  "require_pattern": []
}
```
