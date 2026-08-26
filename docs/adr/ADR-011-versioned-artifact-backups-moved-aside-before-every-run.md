---
id: "ADR-011"
title: "Versioned artifact backups moved aside before every run"
status: "Accepted"
date: "2026-08-26"
binding: false
gate: null
documents_shipped: false
verified_in:
  - "whycast/backups.py"
  - "webui/runner.py"
  - "tests/test_artifact_backups.py"
supersedes: []
superseded_by: null
related:
  - "ADR-008"
  - "ADR-009"
  - "ADR-010"
topics:
  - "artifact-writes"
  - "data-durability"
  - "job-orchestration"
aliases:
  - "backups directory"
  - "move aside"
  - "run stamp"
components:
  - "whycast.backups"
  - "webui.runner"
symbols:
  - "move_to_backup"
  - "copy_to_backup"
  - "move_artifacts"
  - "run_stamp"
context_scope: "selective"
---

<!-- markdownlint-disable MD025 -->

# ADR-011 Versioned artifact backups moved aside before every run

## Status

Accepted, 2026-08-26.

**Decision Maker:** User: Robert van den Breemen ("Ik wil dat je alle bestaande artefacten altijd in een backup directory zet, verplaats ze, en zorg dat backups nooit overschreven worden", followed by "Het verplaatsen moet gebeuren voordat je echt gaat beginnen").

## Status History

```yaml
status_history:
  - date: 2026-08-26
    status: Proposed
    changed_by: Robert van den Breemen
    reason: Owner asked that no artifact a run replaces is ever lost
    changed_via: adr-kit
  - date: 2026-08-26
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-009
    changed_via: adr-kit lifecycle
  - date: 2026-08-26
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-010
    changed_via: adr-kit lifecycle
  - date: 2026-08-26
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-008
    changed_via: adr-kit lifecycle
  - date: 2026-08-26
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Accepted by Robert on request, 2026-08-26; implemented and verified against the real corpus
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

ADR-009 removed per-artifact `.bak` files: a full re-run of ~48 episodes doubled the file count in `podcasts/` and buried the corpus under rows nobody read. That decision solved the noise and accepted a cost, stated plainly at the time: **overwriting an artifact became unrecoverable**, because `podcasts/` is gitignored and no version history exists anywhere.

Running it made that cost concrete rather than theoretical. A forced re-run of `episode_0` replaced thirteen artifacts, and the only reason the previous versions survived was a phase-4 snapshot that had been built for the diff view, not for safety. The owner asked for the guarantee to be explicit: every artifact a run replaces is preserved, the preservation happens **before** the run starts, and a preserved version is never overwritten.

"Before the run starts" is not a detail. Preserving each file as it is written leaves a directory holding a mix of old and new output halfway through, with nothing to say which is which. Doing it up front means that afterwards, what is in `podcasts/` is from this run and what it displaced is in one place.

## Decision Drivers

* Human time is the scarce resource here: a GPU re-run costs 15 minutes and a post-processing run costs money, so an output nobody can get back is worse than a directory that is slightly larger.
* ADR-009's objection was to `.bak` files *next to artifacts*, never to keeping history. A separate tree keeps `podcasts/` exactly as clean.
* A backup that can be overwritten is not a backup. Two runs on the same day must not collide.
* The pipeline decides at runtime that it will not write some outputs, so "moved aside" must not become "quietly gone".

## Considered Options

* Option A: move what a run replaces into `backups/<base>/<run stamp>/` before it starts, copy what it both reads and writes, and put back anything it turned out not to rewrite.
* Option B: bring back `.bak`, reversing ADR-009.
* Option C: keep the phase-4 job snapshot as the only copy.
* Option D: make `podcasts/` a git repository of its own.

## Decision Outcome

Chosen option: **Option A**. Before a job runs, every artifact it is going to replace is moved into `backups/<base_name>/<run stamp>/`. The stamp is fixed once per process, so everything one run displaced lands together and can be read as "the state before this run".

Three refinements, each of which came from the implementation failing without it:

1. **A file the job both reads and writes is copied, not moved.** `<base>_merged.txt` is written by speaker assignment and then read by post-processing as its preferred input. Moving it away left the job with nothing to read, and it failed before writing anything.
2. **Anything moved but not rewritten is put back.** The speaker analysis is skipped when a saved mapping exists (ADR-010); the alternative blog is skipped when its prompt file is absent. Both were moved aside and never regenerated, so a run reported success while the episode silently lost an artifact it had held for a year.
3. **Names never collide.** A destination already taken gets a counter, so a second run within the same second cannot overwrite the first.

Option B was rejected because it recreates exactly the noise ADR-009 removed, and keeps only one previous version. Option C was rejected because that snapshot exists to answer "what changed", is bounded by job-log retention, and was never meant to be the only copy of anything. Option D was rejected as a large amount of machinery, and a poor fit: git is bad at 100 MB audio files and the corpus is not a codebase.

### Confirmation

* `tests/test_artifact_backups.py` proves artifacts are written without a `.bak`, and that the io_utils primitive still makes one when asked.
* A run against the real corpus moved 13 artifacts to a stamped directory and left `podcasts/` holding only what that run produced.
* A failed post-processing run had its 14 moved artifacts restored from the backup, which is the case this exists for.

## Decision Contract

### Must

* Before a job runs, every artifact kind it will write is moved into `backups/<base_name>/<run stamp>/`.
* A kind the job also *reads* is copied there instead, so the job still finds its input.
* A backup path that already exists gets a counter; an existing backup is never overwritten or removed.
* Anything moved aside that the run did not rewrite is restored when the job finishes.
* The run stamp is fixed once per process, so one run produces one directory.
* Pipeline inputs - `vocabulary.json`, `prompts/*.txt`, `<base>_speakers.json` - are never moved. They are read, not written (ADR-010).

### Must Not

* Do not reintroduce `.bak` files beside artifacts in `podcasts/` (ADR-009 stands).
* Do not delete anything from the backup tree automatically. Pruning it is the owner's decision, taken with the files in front of them.
* Do not let a failure to back up be silent: a job that cannot preserve what it is about to replace must refuse to start.

### Exceptions

* Audio downloads are not backed up: the source recording is not something a run replaces, and a 20-100 MB copy per download buys nothing.

### Verification

* `tests/test_artifact_backups.py`, `tests/test_step_jobs.py`.
* `whycast/backups.py` - `move_to_backup`, `copy_to_backup`, `_unique`.
* `webui/runner.py` - `_backup_before`, `_restore_unwritten`, `_JOB_OUTPUT_KINDS`, `_JOB_INPUT_KINDS`.

## Consequences

### Positive

* An overwrite is recoverable again, without the noise that made ADR-009 remove `.bak`: `podcasts/` gains nothing, and the backup tree keeps every version rather than one.
* A run that dies halfway leaves an unambiguous state - what is in `podcasts/` is from this run.
* The diff view reads the same backup, so "what did this run change" and "how do I get the old one back" are answered by one mechanism instead of two.

### Negative

* `backups/` grows without bound. Nothing prunes it, deliberately: automatic deletion is what this decision exists to prevent. The owner will have to look at it eventually, and a full re-run of the corpus is roughly the size of the corpus.
* The move/copy/restore rules depend on a per-job table of what it reads and writes. That table is hand-maintained, and both times it was wrong the failure was a job that could not find its input or an artifact that vanished. `tests/test_step_jobs.py` checks the table against the handlers, which narrows but does not close the gap.
* Backing up before the run means a job that fails immediately still leaves a stamped directory behind. Harmless, and honest about what was attempted.

## Pros and Cons of the Options

### Option A: move aside before the run, into a stamped directory

* Good, because the guarantee is simple to state: nothing a run replaces is lost.
* Good, because `podcasts/` stays exactly as clean as ADR-009 made it.
* Good, because more than one previous version survives, which `.bak` never allowed.
* Bad, because it needs a per-job table of reads and writes, which is hand-maintained and was wrong twice before it was right.

### Option B: bring back `.bak`

* Good, because it is one flag and no new concepts.
* Bad, because it doubles the file count in `podcasts/` and buries the unmatched view - the exact problem ADR-009 was written to solve.
* Bad, because it keeps only the immediately previous version.

### Option C: rely on the phase-4 job snapshot

* Good, because it already exists and needs no new code.
* Bad, because it is bounded by job-log retention, and a copy meant for a diff view is a poor place for the only copy of an artifact.

### Option D: make `podcasts/` its own git repository

* Good, because it would give real history with real tooling.
* Bad, because git handles 100 MB audio files poorly, and every run would need a commit step nobody asked for.

## Open Questions

- [x] When and how does `backups/` get pruned? Nothing removes anything today, on purpose. A retention rule - by age, by count per episode, or by hand - is a decision to take once the tree has grown enough to be worth measuring. — **Answered 2026-08-26 by User: Robert van den Breemen:** Nothing prunes it, and that is the decision rather than a gap. Automatic deletion of preserved versions is precisely what this ADR exists to prevent, so no retention rule ships with it: the tree grows, and the owner removes what they no longer want with the files in front of them - the same way the 543 MB of duplicate audio was handled on 2026-08-25. Revisit when the tree is large enough to measure, which needs a real number rather than a guess; a full re-run of the corpus is roughly the size of the corpus, so that moment is foreseeable but not here. Decided by Robert van den Breemen, 2026-08-26.

## Related Decisions

* **ADR-009 (Discard partial downloads and drop per-artifact backups)**: amends it. The `.bak` files stay gone; what returns is the ability to recover, in a place that does not pollute the corpus.
* **ADR-010 (Persisted editable speaker mapping as pipeline input)**: the reason inputs are never moved, and the reason the speaker analysis is skipped often enough that "restore what was not rewritten" had to exist.
* **ADR-008 (Web UI as FastAPI layer over extracted pipeline library)**: the job runner is where this happens, before the handler is called.

## References

* `whycast/backups.py` - the primitive and the naming scheme.
* `webui/runner.py:_backup_before`, `_restore_unwritten` - where a job uses it.
* Measured 2026-08-26: a forced re-run of `episode_0` moved 13 artifacts; a failed post-processing run had 14 restored from the backup.

## Enforcement

```json
{
  "forbid_import": [],
  "forbid_pattern": [],
  "require_pattern": []
}
```
