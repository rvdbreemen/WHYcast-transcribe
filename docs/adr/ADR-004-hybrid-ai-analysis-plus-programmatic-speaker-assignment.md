---
id: "ADR-004"
title: "Hybrid AI-analysis plus programmatic speaker assignment"
status: "Accepted"
date: "2026-08-23"
binding: false
gate: null
documents_shipped: true
verified_in:
  - "transcribe.py"
  - "prompts/speaker_analysis_prompt.txt"
supersedes: []
superseded_by: null
related:
  - "ADR-010"
topics:
  - "speakers"
  - "llm"
  - "attribution"
  - "pipeline"
aliases:
  - "speaker assignment"
  - "two-phase speaker mapping"
  - "SPEAKER_UNKNOWN attribution"
components:
  - "transcribe.speaker_assignment_step"
  - "transcribe.analyze_speakers_with_o4"
  - "transcribe.apply_speaker_mapping_programmatically"
  - "transcribe.attribute_unknown_speakers_with_ai"
symbols: []
context_scope: "selective"
format: "madr"
llm_judge: true
---

<!-- markdownlint-disable MD025 -->

# ADR-004 Hybrid AI-analysis plus programmatic speaker assignment

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
  - date: 2026-08-25
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-010
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

Earlier versions asked the LLM to rewrite the whole transcript with speaker names, which hallucinated and dropped lines. Commits 599147f and 30cd91a replaced this with a two-phase approach in `speaker_assignment_step` (`transcribe.py:1439`): (1) `analyze_speakers_with_o4` asks the reasoning model only for a `SPEAKER_NN -> name` mapping; (2) `apply_speaker_mapping_programmatically` (`transcribe.py:1136`) applies it with string replacement, then `attribute_unknown_speakers_with_ai` handles leftover `SPEAKER_UNKNOWN`. `speaker_assignment_fallback` (`transcribe.py:1504`) keeps the old path when analysis fails. `analyze_transcript_changes` (`transcribe.py:1110`) diffs original vs processed to detect content loss.

## Decision Drivers

* Transcript text must not be altered by the LLM
* Speaker naming still needs reasoning over context
* Graceful degradation when the model fails

## Considered Options

* Two-phase: LLM produces mapping, code applies it
* Single LLM rewrite of the full transcript
* Fully programmatic (voice-print/heuristic) naming

## Decision Outcome

Chosen option: **Two-phase: LLM produces mapping, code applies it**, because separating judgement (who is SPEAKER_02) from mechanics (replace the label) removes content-loss hallucinations while keeping LLM reasoning where it adds value.

### Confirmation

Verified by inspecting `transcribe.py:1439-1504`; `transcribe.py:1136`; `transcribe.py:1110`.

## Decision Contract

### Must

* Speaker names are applied by deterministic code from a parsed mapping
* Any LLM step that rewrites transcript text is followed by a change analysis
* Failure of the analysis phase falls back rather than aborting

### Must Not

* Do not reintroduce a single prompt that outputs the complete renamed transcript as the primary path

### Exceptions

* None

### Verification

* `transcribe.py:1439-1504`
* `transcribe.py:1136`
* `transcribe.py:1110`

## Consequences

### Positive

* No dropped or altered lines from the speaker step
* Cheap second phase (no tokens)

### Negative

* Mapping parser is brittle to prompt output format (mitigation: `parse_speaker_mapping_from_analysis` tolerant regexes + fallback path)

## Pros and Cons of the Options

### Two-phase: LLM produces mapping, code applies it

* Good, because deterministic text integrity
* Bad, because parsing LLM output format

### Single LLM rewrite of the full transcript

* Good, because one call
* Bad, because observed hallucination and line loss

### Fully programmatic (voice-print/heuristic) naming

* Good, because no LLM cost
* Bad, because it needs enrolled voice profiles per host; not available

## Open Questions

None.

## Related Decisions

* ADR-002 (consumes diarization labels).
* ADR-003 (uses OPENAI_SPEAKER_MODEL).

## References

* `transcribe.py`
* `prompts/speaker_analysis_prompt.txt`
* `prompts/speaker_unknown_attribution_prompt.txt`

