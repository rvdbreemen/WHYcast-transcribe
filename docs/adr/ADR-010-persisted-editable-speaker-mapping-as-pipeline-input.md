---
id: "ADR-010"
title: "Persisted editable speaker mapping as pipeline input"
status: "Accepted"
date: "2026-08-25"
binding: false
gate: null
documents_shipped: false
verified_in:
  - "whycast/pipeline/speakers.py"
  - "whycast/episodes.py"
  - "tests/test_speaker_mapping_precedence.py"
  - "tests/test_webui_speakers_editor.py"
supersedes: []
superseded_by: null
related:
  - "ADR-004"
  - "ADR-009"
  - "ADR-011"
topics:
  - "speaker-assignment"
  - "correction-workflow"
  - "pipeline-inputs"
aliases:
  - "speakers.json"
  - "speaker mapping"
  - "speaker editor"
components:
  - "whycast.pipeline.speakers"
  - "whycast.episodes"
symbols:
  - "speaker_assignment_step"
  - "analyze_speakers_with_o4"
  - "load_speaker_mapping"
  - "save_speaker_mapping"
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-010 Persisted editable speaker mapping as pipeline input

## Status

Accepted, 2026-08-25.

**Decision Maker:** User: Robert van den Breemen (chose the approach on 2026-08-25 while grilling ADR-009: "Mapping bewerken, niet output").

## Status History

```yaml
status_history:
  - date: 2026-08-25
    status: Proposed
    changed_by: Robert van den Breemen
    reason: Phase 3 correction workflows need a place for human corrections that a re-run reproduces
    changed_via: adr-kit
  - date: 2026-08-25
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-004
    changed_via: adr-kit lifecycle
  - date: 2026-08-25
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-009
    changed_via: adr-kit lifecycle
  - date: 2026-08-25
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Accepted by Robert on request, 2026-08-25; implemented and verified in phase 3
    changed_via: adr-kit lifecycle
  - date: 2026-08-26
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-011
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

ADR-004 established two-phase speaker assignment: an LLM decides which name belongs to `SPEAKER_02`, and deterministic code applies that mapping to the transcript. The mapping itself is never persisted. `analyze_speakers_with_o4` (`whycast/pipeline/speakers.py:148`) returns it in memory, `speaker_assignment_step` uses it immediately, and only a human-readable report survives as `<base>_speaker_analysis.txt`. Nothing reads that report back.

Two consequences follow. The mapping is re-derived by a paid model on every re-run even when it was already correct. And there is nowhere for a person to fix `SPEAKER_02` being called "Nancy" when it is actually "Ad", short of hand-editing the generated transcript.

Hand-editing the generated transcript is now explicitly ruled out: ADR-009 dropped per-artifact backups, so any job may overwrite a generated file and the correction would be gone without trace. Its Decision Contract states that human input must live in a file the pipeline reads, never in one it writes. The phase 3 correction workflows (TASK-004) therefore need such a file to exist before they can be built.

## Decision Drivers

* A correction must survive a re-run. Anything else quietly destroys human work, which is the failure mode ADR-009 was written to prevent.
* Re-running speaker assignment on an episode whose names are already right should not pay an LLM to rediscover them.
* ADR-004's separation must hold: judgement produces a mapping, code applies it. Where the judgement came from - a model or a person - is not the applier's concern.
* The correction workflow must not need a database: the filesystem is the source of truth (ADR-008), and the web UI's SQLite is a rebuildable cache.

## Considered Options

* Option A: persist the mapping as `<base>_speakers.json` next to the episode; the file, when present, is the mapping, and the LLM phase is skipped.
* Option B: keep the mapping in memory and let the editor re-run the LLM with an extra instruction ("SPEAKER_02 is Ad").
* Option C: store corrections in the web UI's SQLite database.
* Option D: parse the mapping back out of the existing `<base>_speaker_analysis.txt` report.

## Decision Outcome

Chosen option: **Option A**. `<base>_speakers.json` becomes a first-class pipeline input: a JSON object mapping speaker labels to names, next to the episode audio it belongs to.

Precedence is simple enough to state in one sentence: if the file exists it is the mapping, and phase 1 is skipped; if it does not, the LLM produces one as it does today and the result is written to that path, so the next run - and the editor - has something to work with. A person editing the file therefore does not fight the pipeline; they replace its guess, permanently, and every later re-run reproduces their correction instead of overwriting it.

Option B was rejected because it pays for a model call to be told an answer we already have, and because the model may still disagree. Option C was rejected because it puts the only copy of human work in a cache that ADR-008 says must be deletable and rebuildable from disk. Option D was rejected because parsing prose back into data is a guess about a format nobody guaranteed; the report is written for a human to read, and its wording is free to change.

### Confirmation

* A test proves that with the file present, `speaker_assignment_step` produces the mapped transcript and makes no OpenAI call.
* A test proves that with the file absent, the LLM path runs as before and the derived mapping is written to `<base>_speakers.json`.
* A test proves a re-run after a hand edit reproduces the edited names rather than the model's original ones.

## Decision Contract

### Must

* The mapping lives at `<output_dir>/<base>_speakers.json` as a JSON object of `{"SPEAKER_00": "Name", ...}`.
* When that file exists and parses, it is the mapping; the paid analysis phase is skipped.
* When it does not exist, the LLM-derived mapping is saved to it, so the next run can be corrected and re-run for free.
* Writes to it use `backup=True`: it is human input, unlike generated artifacts (ADR-009).
* The scanner recognises it as a distinct kind, so it does not appear as an unmatched file.
* Deterministic code still applies the mapping (ADR-004 is unchanged in that respect).
* Re-transcribing an episode marks its mapping stale rather than deleting it. The file records which transcript it was made for; when that no longer matches, the mapping is not applied automatically and the person is asked to check the names first.

### Must Not

* Do not store speaker corrections in the web UI database or any other cache.
* Do not delete a mapping automatically, on re-transcription or otherwise. It is human work; discarding it is a decision only a person makes.
* Do not parse the mapping back out of `<base>_speaker_analysis.txt`; that report is prose for a human reader.
* Do not let a malformed or unreadable mapping file silently fall back to the LLM - report it, so a typo in a hand-edited file is visible rather than expensively papered over.

### Exceptions

* None.

## Consequences

### Positive

* A correction is made once and holds. Re-running speaker assignment, post-processing, or the whole episode reproduces it.
* Re-runs of a corrected episode skip the most expensive judgement call, so iterating on prompts or formatting stops paying for speaker analysis each time.
* The editor in TASK-004 becomes a small thing to build: it reads and writes one JSON file, and enqueues an existing job.
* The mapping is inspectable and diffable, which the in-memory version never was.

### Negative

* One more file per episode in `podcasts/`, a directory that already needed a cleanup pass. It is small, and unlike a `.bak` it carries meaning.
* Precedence can surprise: after a hand edit, re-running speaker assignment will *not* consult the model, which is the point but is not obvious from the button label. The web UI must say which mapping will be used, and offer a way to discard the file and let the model decide again.
* A mapping that is stale relative to a re-transcription - different diarization, different `SPEAKER_xx` labels for the same people - will map the wrong names confidently. Re-transcription must therefore be treated as invalidating the mapping; unmapped labels remaining after application are already reported (`whycast/pipeline/speakers.py:610`) and that warning becomes the signal.

## Pros and Cons of the Options

### Option A: persisted `<base>_speakers.json` as pipeline input

* Good, because a correction survives every re-run, which is the whole requirement.
* Good, because it removes a paid call from the common case.
* Good, because it needs no new storage: it is a file next to the episode, like everything else here.
* Bad, because it adds a precedence rule a user must understand, and a stale mapping after re-transcription is a real trap.

### Option B: re-run the LLM with the correction as an extra instruction

* Good, because nothing new is stored.
* Bad, because it pays a model to be told the answer, and it may still disagree; the correction is a request, not a guarantee.

### Option C: corrections in the web UI database

* Good, because the editor can write structured data without touching `podcasts/`.
* Bad, because ADR-008 requires that database to be deletable and rebuildable from disk; the only copy of human work must not live there.

### Option D: parse the mapping back out of the analysis report

* Good, because no new file at all.
* Bad, because it turns prose written for a human into a parsing contract nobody agreed to, and it breaks the moment the report's wording changes.

## Open Questions

- [x] Should re-transcribing an episode delete or flag its `<base>_speakers.json`, given the diarization may assign different `SPEAKER_xx` labels to the same people? Leaving it risks confidently applying the wrong names. — **Answered 2026-08-25 by User: Robert van den Breemen:** Flag it, never delete it. A re-transcription marks the mapping stale (a flag inside the JSON, recording the transcript it was made for) instead of removing it. A stale mapping is not applied automatically: the job reports it and the UI asks the person to check the names, offering both 'apply anyway' and 'discard and let the model decide again'. Deleting would silently destroy human work, which is precisely the failure ADR-009 exists to prevent; applying it blindly would confidently attribute one speaker's words to another, because pyannote labels are per-run cluster ids and not identities. The cost is one confirmation click after each re-transcription, which the user accepted explicitly. Decided by Robert van den Breemen, 2026-08-25.

## Related Decisions

* **ADR-004 (Hybrid AI-analysis plus programmatic speaker assignment)**: amends it by giving the mapping a persisted home and letting a file replace the analysis phase. The separation it decided - judgement produces a mapping, code applies it - is unchanged.
* **ADR-009 (Discard partial downloads and drop per-artifact backups)**: this is the file its Decision Contract requires, the place human input lives now that generated artifacts carry no backup.
* **ADR-003 (Per-task OpenAI model selection)**: unaffected; when the analysis phase runs, it still selects its model the same way.

## References

* `whycast/pipeline/speakers.py:148` `analyze_speakers_with_o4` - produces the mapping in memory today.
* `whycast/pipeline/speakers.py:226` `parse_speaker_mapping_from_analysis` - the existing parse, kept for the LLM path.
* `whycast/pipeline/speakers.py:650` `speaker_assignment_step` - the two-phase entry point that gains the file check.
* `whycast/pipeline/speakers.py:610` - the unmapped-label warning that becomes the stale-mapping signal.
* Backlog TASK-004 (WebUI phase 3 correction workflows).

## Enforcement

```json
{
  "forbid_import": [],
  "forbid_pattern": [],
  "require_pattern": []
}
```
