---
id: "ADR-009"
title: "Discard partial downloads and drop per-artifact backups"
status: "Accepted"
date: "2026-08-25"
binding: false
gate: null
documents_shipped: false
verified_in: []
supersedes: []
superseded_by: null
related:
  - "ADR-008"
  - "ADR-010"
  - "ADR-011"
topics:
  - "artifact-writes"
  - "downloads"
  - "data-durability"
aliases:
  - "backup"
  - ".bak"
  - "partial download"
  - "truncated mp3"
components:
  - "whycast.io_utils"
  - "whycast.pipeline.feed"
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-009 Discard partial downloads and drop per-artifact backups

## Status

Accepted, 2026-08-25.

**Decision Maker:** User: Robert van den Breemen (both changes requested directly on 2026-08-25: "zet uit per aanroep" for the backups, "afgebroken downloads mogen ook weggegooid worden, daar heb je niets aan" for the downloads).

## Status History

```yaml
status_history:
  - date: 2026-08-25
    status: Proposed
    changed_by: Robert van den Breemen
    reason: Amends ADR-008 after the phase 2 backup-noise and partial-download findings
    changed_via: adr-kit
  - date: 2026-08-25
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-008
    changed_via: adr-kit lifecycle
  - date: 2026-08-25
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Accepted by Robert on request, 2026-08-25; grill settled the open question (human input goes to pipeline inputs, not artifacts)
    changed_via: adr-kit lifecycle
  - date: 2026-08-25
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-010
    changed_via: adr-kit lifecycle
  - date: 2026-08-26
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-011
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

ADR-008 requires that "all artifact writes go through the pipeline library, which writes atomically (write-to-temp, then rename) and backs up files it overwrites". Phase 2 implemented that in `whycast/io_utils.py` and wired it into every artifact writer. Running it surfaced two problems that the original wording did not anticipate.

**Backup noise.** Every overwrite leaves a `<artifact>.bak` beside the artifact. A full re-run of the roughly 48 episodes roughly doubles the file count in `podcasts/` (measured: 812 entries today, 719 of them artifacts) and puts hundreds of `.bak` rows on the web UI's unmatched page, because `.bak` is deliberately not an artifact format. The backup buys a single previous version; the cost is a permanently noisy view of the corpus.

**Partial downloads.** `whycast/pipeline/feed.py` streams episode audio in 8 KB chunks straight onto the target path. An aborted download therefore leaves a truncated `.mp3` that the scanner treats as a complete episode and that the "skip files that already exist" rule then refuses to re-download. Buffering the whole file to reuse `atomic_write_bytes` is not an option: episodes are 20-100 MB.

Both decisions are the user's to make, and both were made explicitly on 2026-08-25.

## Decision Drivers

* The corpus view must stay readable; a feature that doubles the file count for marginal benefit is not worth it here.
* A truncated recording has no value: there is nothing to salvage from half an mp3, and its presence actively blocks the retry.
* Memory: a streaming download must not be turned into a buffered one.
* ADR-008's atomicity guarantee (never a partial file at the target path) must survive both changes untouched.

## Considered Options

* Option A: keep `backup=True` as the `io_utils` default, pass `backup=False` at each artifact call site; add a streaming atomic writer and use it for downloads.
* Option B: change the `io_utils` default to `backup=False`.
* Option C: keep backups and hide `.bak` files in the web UI instead.
* Option D: keep backups but prune them on a schedule (keep the newest N).

## Decision Outcome

Chosen option: **Option A**, because it removes the noise where it is generated without weakening the primitive: `whycast.io_utils.atomic_write_text/bytes` still back up by default, so any future caller that wants a safety copy gets one by asking for nothing. Downloads gain a streaming `atomic_writer` context manager that commits on clean exit and discards on any exception, so an aborted download leaves the target path exactly as it was.

Option B was rejected because a silent default is the wrong place for a policy decision: a call site that wants no backup should say so. Option C was rejected because hiding files does not stop them accumulating, and a view that differs from the disk is worse than a noisy honest one. Option D was rejected as machinery for a benefit nobody asked for.

### Confirmation

* `tests/test_io_utils.py` proves the primitive still backs up when `backup=True`, and that the artifact writers produce no `.bak`.
* A download test over a local `http.server` proves a completed download is byte-identical and an aborted or truncated one leaves no file at the target path and no phantom episode in a scan.

## Decision Contract

### Must

* Artifact writes stay atomic: temp file in the same directory, then `os.replace`. A partial file must never appear at the target path.
* Artifact call sites in `whycast/pipeline/` pass `backup=False` explicitly.
* `whycast.io_utils` keeps `backup=True` as its default, and that path stays tested.
* Audio downloads write through the streaming atomic writer and discard the temp file on any failure.
* A download whose body is shorter than the `Content-Length` the server advertised counts as failed, not complete.

### Must Not

* Do not change the `io_utils` default to `backup=False`.
* Do not buffer an entire episode in memory to make a download atomic.
* Do not back up `.env` (a `.env.bak` would be a second plaintext copy of a secret).
* Human input must not be stored in a file the pipeline writes. A correction workflow edits the pipeline's *input* (the speaker mapping, `vocabulary.json`, a prompt file), never a generated artifact, because a generated artifact carries no backup and any job may overwrite it.

### Exceptions

* None.

## Consequences

### Positive

* `podcasts/` stops growing a second copy of every artifact on every re-run, and the unmatched page stays readable.
* An aborted download no longer poisons the corpus with a truncated recording that also blocks its own retry.
* The streaming writer closes the last gap in ADR-008's atomicity guarantee, which previously covered only whole-payload writes.

### Negative

* **Overwriting an artifact is now unrecoverable.** `podcasts/` is gitignored, so there is no version history anywhere: a re-run that produces a worse summary destroys the better one. This is acceptable for machine-generated output, which any job can regenerate, and it is the reason the phase 3 correction workflows (TASK-004) must not put human work in an artifact. Settled 2026-08-25: the phase 3 editor edits the speaker *mapping*, and a re-run reproduces the correction rather than destroying it (see Open Questions). Human input therefore lives in a file the pipeline reads, never in one it writes.
* A correction that cannot be expressed as a mapping - rewriting an individual sentence - has nowhere to live under this decision. Accepted explicitly rather than designed around; it would need its own decision if it ever becomes a requirement.
* The planned phase 4 "diff previous versus new artifact after a re-run" (TASK-005) loses its cheap source of the previous version. It now needs its own snapshot before a re-run, or it drops out of scope.
* A killed process still leaves a `.tmp` file, because a killed process runs no cleanup handler. That is handled outside this decision by `whycast.io_utils.sweep_stale_temps`, called by the worker when a job settles.

## Pros and Cons of the Options

### Option A: backup=False per call site, streaming atomic writer for downloads

* Good, because the policy is visible at the call site rather than hidden in a default.
* Good, because the primitive keeps its capability, so re-enabling backups anywhere is one word.
* Good, because the streaming writer fixes downloads without buffering 100 MB.
* Bad, because eight call sites must each be found and changed, and a missed one silently keeps writing `.bak`.

### Option B: change the io_utils default to backup=False

* Good, because it is a one-line change and cannot miss a call site.
* Bad, because it hides a product decision in a library default, and any future caller silently loses its safety copy.

### Option C: keep backups, hide .bak in the web UI

* Good, because no code that writes files changes at all.
* Bad, because the files still accumulate, now invisibly, and the UI stops reflecting the disk.

### Option D: keep backups with scheduled pruning

* Good, because it keeps a recovery window.
* Bad, because it adds a retention policy, a scheduler and a failure mode for a benefit nobody asked for.

## Open Questions

- [x] Before TASK-004 (phase 3 correction workflows) ships: do hand-edited artifacts get a distinct path, or does that writer re-enable `backup=True`? Leaving it as-is means a later re-run silently destroys manual corrections. — **Answered 2026-08-25 by User: Robert van den Breemen:** Neither: the phase 3 editor edits the speaker MAPPING (the input), not the generated artifact. A re-run then reproduces the correction instead of destroying it, so no backup and no second path are needed for the artifacts. This follows ADR-004, which already separates judgement (which name belongs to SPEAKER_02) from application (the code that rewrites the transcript). Accepted limitation, stated by the user when choosing: a correction that cannot be expressed as a mapping - rewriting an individual sentence - has no home in this model and is out of scope for phase 3. Decided by Robert van den Breemen, 2026-08-25.

## Related Decisions

* **ADR-008 (Web UI as FastAPI layer over extracted pipeline library)**: amends its Decision Contract clause "and backs up files it overwrites". The atomicity requirement in that clause is unchanged and is extended here to streaming downloads.

## References

* `whycast/io_utils.py` - `atomic_write_text`, `atomic_write_bytes`, `atomic_writer`, `sweep_stale_temps`.
* `whycast/pipeline/feed.py` - the two audio download sites.
* Measured 2026-08-25: `podcasts/` holds 812 entries, 719 artifacts across 62 audio bases (scanner output).
* Session 2026-08-25: user decisions "zet uit per aanroep" and "afgebroken downloads mogen ook weggegooid worden".

## Enforcement

```json
{
  "forbid_import": [],
  "forbid_pattern": [
    {
      "pattern": "open\\(\\s*audio_file\\s*,\\s*['\\\"]wb['\\\"]",
      "path_glob": "whycast/pipeline/feed.py",
      "message": "Audio downloads must use the streaming atomic writer, not a direct open() (ADR-009)"
    }
  ],
  "require_pattern": []
}
```
