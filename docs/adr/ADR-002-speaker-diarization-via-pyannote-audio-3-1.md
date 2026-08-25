---
id: "ADR-002"
title: "Speaker diarization via pyannote.audio 3.1"
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
  - "ADR-008"
topics:
  - "diarization"
  - "speakers"
  - "pyannote"
  - "gpu"
aliases:
  - "speaker diarization"
  - "pyannote.audio"
  - "who spoke when"
components:
  - "transcribe.diarize_audio"
  - "transcribe.prepare_audio_for_diarization"
  - "config.DIARIZATION_MODEL"
symbols: []
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-002 Speaker diarization via pyannote.audio 3.1

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

Podcast transcripts need speaker labels so later LLM steps can attribute lines. `config.py:36-41` enables diarization by default with `pyannote/speaker-diarization-3.1` (1-10 speakers). `transcribe.py:2742` (`diarize_audio`) runs the pyannote `Pipeline` on GPU from an in-memory waveform shared with Whisper; a Hugging Face token is required (`get_huggingface_token`, `transcribe.py:409`). Diarization output is merged into Whisper segments as `SPEAKER_NN` prefixes.

## Decision Drivers

* Accurate speaker turns on multi-host podcast audio
* Run locally on the same GPU as Whisper
* Gated model access must be explicit (HF token)

## Considered Options

* pyannote.audio 3.1 speaker-diarization pipeline
* WhisperX (whisper + pyannote wrapper)
* No diarization; let the LLM guess speakers from text

## Decision Outcome

Chosen option: **pyannote.audio 3.1 speaker-diarization pipeline**, because it is the best-performing open diarization model, runs on the existing torch/CUDA stack, and integrating it directly avoids WhisperX pinning its own whisper fork.

### Confirmation

Verified by inspecting `transcribe.py:2742` diarize_audio; `transcribe.py:2621` call site in full_workflow; `config.py:36-41`.

## Decision Contract

### Must

* Diarization runs before transcription alignment; speaker segments are passed into `transcribe_audio`
* Model id, min/max speakers come from `config.py` env-vars
* A missing HF token is prompted for, never silently skipped

### Must Not

* Do not depend on WhisperX or another wrapper that pins its own whisper build

### Exceptions

* None

### Verification

* `transcribe.py:2742` diarize_audio
* `transcribe.py:2621` call site in full_workflow
* `config.py:36-41`

## Consequences

### Positive

* Speaker labels available for all downstream LLM steps
* Shares waveform in memory, one audio decode

### Negative

* Gated HF model and token handling; pyannote version drift against torch (mitigation: `DIARIZATION_ALTERNATIVE_MODEL` fallback)

## Pros and Cons of the Options

### pyannote.audio 3.1 speaker-diarization pipeline

* Good, because state-of-the-art open model on GPU
* Bad, because gated download + token ceremony

### WhisperX (whisper + pyannote wrapper)

* Good, because alignment built in
* Bad, because it pins its own whisper fork, conflicting with ADR-001

### No diarization; let the LLM guess speakers from text

* Good, because zero extra deps
* Bad, because LLM speaker guesses are unreliable on long episodes

## Open Questions

None.

## Related Decisions

* ADR-001 (shares GPU/torch stack).

## References

* `config.py`
* `transcribe.py`
* `requirements.txt`

## Enforcement

```json
{
  "forbid_import": [
    {
      "pattern": "^\\s*(import whisperx\\b|from whisperx\\b)",
      "path_glob": "*.py",
      "message": "Diarization goes through pyannote.audio directly, not WhisperX (ADR-002)"
    }
  ],
  "forbid_pattern": [],
  "require_pattern": []
}
```
