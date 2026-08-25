---
id: "ADR-001"
title: "Use faster-whisper on CUDA for transcription"
status: "Accepted"
date: "2026-08-23"
binding: false
gate: null
documents_shipped: true
verified_in:
  - "requirements.txt"
  - "config.py"
  - "transcribe.py"
supersedes: []
superseded_by: null
related:
  - "ADR-008"
topics:
  - "transcription"
  - "whisper"
  - "gpu"
  - "cuda"
aliases:
  - "faster-whisper"
  - "CTranslate2 whisper"
  - "whisper large-v3"
components:
  - "transcribe.setup_model"
  - "transcribe.transcribe_audio"
  - "config.MODEL_SIZE"
symbols: []
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-001 Use faster-whisper on CUDA for transcription

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
```

## Context and Problem Statement

WHYcast episodes are 1-2 hour Dutch/English podcasts. Transcription must run locally on the maintainer's NVIDIA GPU in reasonable wall-clock time with high accuracy. `requirements.txt` pins `faster-whisper`, `torch`, `torchaudio`; `config.py:11-14` defaults to `large-v3`, `cuda`, `float16`, beam size 5. `transcribe.py:1990` (`setup_model`) builds a `WhisperModel`/`BatchedInferencePipeline`; `force_cuda_device` (`transcribe.py:1947`) refuses silent CPU fallback.

## Decision Drivers

* Local GPU inference, no audio leaves the machine
* Throughput on long episodes (batched CTranslate2 inference)
* Highest available accuracy (`large-v3`)

## Considered Options

* faster-whisper (CTranslate2) on CUDA float16
* openai-whisper reference implementation
* OpenAI hosted Whisper API

## Decision Outcome

Chosen option: **faster-whisper (CTranslate2) on CUDA float16**, because it gives roughly 4x the throughput of the reference implementation at equal accuracy, keeps audio local, and exposes batched inference and int8/float16 compute types through env-vars.

### Confirmation

Verified by inspecting `transcribe.py:1990` setup_model; `transcribe.py:1924-1990` CUDA checks; `config.py:11-14`.

## Decision Contract

### Must

* Transcription uses `faster_whisper.WhisperModel`; model size, device, compute type and beam size come from `config.py` env-vars
* CUDA is the default device; a missing GPU is reported loudly, never silently downgraded

### Must Not

* Do not send audio to a hosted transcription API
* Do not hard-code model size or device outside `config.py`

### Exceptions

* None

### Verification

* `transcribe.py:1990` setup_model
* `transcribe.py:1924-1990` CUDA checks
* `config.py:11-14`

## Consequences

### Positive

* Fast local transcription on consumer GPUs
* Compute type tunable per machine (float16/int8)

### Negative

* Requires CUDA toolchain + matching torch build; setup friction on non-NVIDIA hardware (mitigation: env-vars allow `cpu`/`int8`)

## Pros and Cons of the Options

### faster-whisper (CTranslate2) on CUDA float16

* Good, because batched GPU inference is the fastest local option
* Bad, because the CUDA dependency chain is brittle

### openai-whisper reference implementation

* Good, because simplest install
* Bad, because 3-4x slower, no batching

### OpenAI hosted Whisper API

* Good, because no GPU needed
* Bad, because audio leaves the machine, per-minute cost, 25 MB upload limit

## Open Questions

None.

## Related Decisions

* None.

## References

* `requirements.txt`
* `config.py`
* `transcribe.py`

## Enforcement

```json
{
  "forbid_import": [
    {
      "pattern": "^\\s*(import whisper\\b|from whisper\\b)",
      "path_glob": "*.py",
      "message": "Use faster_whisper, not the openai-whisper reference package (ADR-001)"
    }
  ],
  "forbid_pattern": [],
  "require_pattern": []
}
```
