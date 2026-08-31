---
id: "ADR-012"
title: "Two reasoning models per task profile, not four env-var slots"
status: "Accepted"
date: "2026-08-31"
binding: false
gate: null
documents_shipped: true
verified_in:
  - "whycast/config.py"
  - "whycast/pipeline/llm.py"
  - "whycast/pipeline/speakers.py"
supersedes:
  - "ADR-003"
superseded_by: null
related:
  - "ADR-004"
  - "ADR-005"
  - "ADR-007"
topics:
  - "llm"
  - "openai"
  - "model-selection"
  - "cost"
aliases:
  - "gpt-5.6-luna"
  - "gpt-5.6-sol"
  - "reasoning effort"
  - "per-task model selection"
components:
  - "whycast.config.OPENAI_MODEL"
  - "whycast.config.OPENAI_SPEAKER_MODEL"
  - "whycast.config.OPENAI_REASONING_EFFORT"
  - "whycast.pipeline.llm.model_params"
  - "whycast.pipeline.llm.choose_appropriate_model"
symbols: []
context_scope: "selective"
format: "madr"
---

<!-- markdownlint-disable MD025 -->

# ADR-012 Two reasoning models per task profile, not four env-var slots

## Status

Accepted, 2026-08-31.

**Decision Maker:** User: Robert van den Breemen

## Status History

```yaml
status_history:
  - date: 2026-08-31
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Initial proposal
    changed_via: adr-kit
  - date: 2026-08-31
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: "Superseded by ADR-012: the four model slots collapse to two by default, and the parameter contract for gpt-5.6 removes temperature as a tuning knob"
    changed_via: adr-kit lifecycle
  - date: 2026-08-31
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-004
    changed_via: adr-kit lifecycle
  - date: 2026-08-31
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-005
    changed_via: adr-kit lifecycle
  - date: 2026-08-31
    status: Proposed
    changed_by: "User: Robert van den Breemen"
    reason: Related to ADR-007
    changed_via: adr-kit lifecycle
  - date: 2026-08-31
    status: Accepted
    changed_by: "User: Robert van den Breemen"
    reason: Accepted by Robert in session; supersedes ADR-003 with the measured GPT-5.6 parameter contract and cost
    changed_via: adr-kit lifecycle
```

## Context and Problem Statement

ADR-003 chose per-task OpenAI models with a length-based switch, on the
reasoning that each step has a different cost/quality profile. It defined four
slots: `OPENAI_MODEL`, `OPENAI_LARGE_CONTEXT_MODEL`, `OPENAI_HISTORY_MODEL` and
`OPENAI_SPEAKER_MODEL`, with `choose_appropriate_model()` switching to the
large-context slot past `MAX_INPUT_TOKENS`.

Two things have changed since.

**The four slots collapsed to two distinct models.** Three of them now default
to `gpt-5.6-luna` and only the speaker slot differs (`gpt-5.6-sol`). That is not
drift: a model that handles long inputs natively removes the reason the
large-context slot existed, and history extraction is the same kind of
compression as summarizing. The switch in `choose_appropriate_model()` therefore
returns the same model on both branches, while logging "Using large context
model" as though a decision had been taken. That log line is now conditional on
the two actually differing.

**The parameter contract changed underneath.** Measured against the live API on
2026-08-30, `gpt-5.6-luna` and `gpt-5.6-sol` reject `max_tokens` (they take
`max_completion_tokens`), reject any `temperature` other than 1, and accept
`reasoning_effort` of `none|low|medium|high|xhigh`. ADR-003 assumed a
chat-completions shape where `temperature` was a tuning knob; for these models it
is not a knob at all. `TEMPERATURE` has been removed from the codebase rather
than sent and rejected.

The old code decided which spelling to send from the model name's first letter
(`startswith("o") and not startswith("gpt")`), which put `gpt-5.6-luna` on the
legacy side and would have sent both rejected parameters.

## Decision Drivers

* Match what the models actually accept, verified rather than assumed
* Keep the per-task split where it earns its place, and stop pretending where it does not
* Reasoning effort is the real quality dial now, not temperature or model size
* Cost and wall-clock stay visible and bounded

## Considered Options

* Two models by profile: one for text work, one for speaker judgement
* Keep four genuinely distinct models, one per slot
* One model for everything, including speakers
* Revert to the gpt-4.1 / o4-mini pairing of ADR-003

## Decision Outcome

Chosen option: **two models by profile**, because the pipeline has two kinds of
work and not four. Cleanup, summary, blog and history are compression over text
the model can already hold; speaker assignment is judgement over a whole
transcript, where a stronger model earns its cost.

```
OPENAI_MODEL                = gpt-5.6-luna   reasoning_effort=high
OPENAI_LARGE_CONTEXT_MODEL  = gpt-5.6-luna   reasoning_effort=high
OPENAI_HISTORY_MODEL        = gpt-5.6-luna   reasoning_effort=high
OPENAI_SPEAKER_MODEL        = gpt-5.6-sol    reasoning_effort=high
```

The four env-vars stay. They are how a different machine, a cost experiment or a
future model gets differentiated without a code change (ADR-007), and
`choose_appropriate_model()` still switches when the two names differ. What
changes is that the shipped default no longer claims a distinction it does not
make.

Per-model parameters are built in one place, `llm.model_params()`, from an
explicit list of legacy exceptions with the modern spelling as the default - the
direction that survives a model released after the line was written.

### Confirmation

`whycast/config.py:29-38` for the defaults; `whycast/pipeline/llm.py`
`model_params` and `choose_appropriate_model`; `tests/test_llm_model_params.py`
and `tests/test_llm_chunked_payload.py`, which pin the contract on both call
paths.

## Decision Contract

### Must

* Model names and reasoning effort come from `config.py` env-vars, never literals
* Per-model request parameters are built by `llm.model_params()` and nowhere else
* Speaker work uses its own model and its own reasoning effort
* `finish_reason == "length"` is logged; a truncated answer must not pass silently

### Must Not

* Do not send `temperature` to any model
* Do not infer a model's parameter shape from its name's prefix
* Do not send audio or transcripts to a model chosen outside `config.py`

### Exceptions

* None

### Verification

* `tests/test_llm_model_params.py`
* `tests/test_llm_chunked_payload.py`
* `whycast/config.py:29-38`

## Consequences

### Positive

* The configuration says what the pipeline does. Three identical slots with a
  switch between them read as a cost strategy that was not running.
* The parameter contract is measured, not guessed, and pinned by tests on both
  the single-shot and chunked call paths.
* `reasoning_effort` gives a quality dial that survives model changes, where
  `temperature` no longer exists for these models.

### Negative

* **Cost and latency.** Measured over five calls per arm on identical input:
  the previous gpt-4.1 / o4-mini pairing used 15716 output tokens in 122s; the
  new pairing uses 22955 output tokens of which 13428 are reasoning, in 292s.
  That is 46% more output tokens and 2.4x the wall-clock. Accepted because
  transcription runs offline in a job queue, not interactively.
* **One model is a single point of failure for four steps.** A regression in
  `gpt-5.6-luna` degrades cleanup, summary, blog and history together. The
  env-vars are the mitigation: they can be pointed apart without a release.
* **Reasoning tokens share the output budget.** `MAX_TOKENS=16000` held across
  every call measured (`finish_reason=stop` on all ten), but a longer episode or
  a higher effort could truncate. The warning added for `finish_reason=length`
  is what makes that visible rather than silent.

### Risks and mitigations

* *Risk*: quality of the new pairing was never adjudicated - the A/B measured
  tokens, latency and truncation, not whether the summaries are better.
  *Mitigation*: the artifacts of both runs were kept side by side for human
  comparison; this ADR claims a contract, not a quality improvement.
* *Risk*: a future model is neither legacy nor current in shape.
  *Mitigation*: `model_params()` defaults to the modern spelling, so a new model
  fails with a 400 naming the parameter rather than silently taking the legacy
  path.

## Pros and Cons of the Options

### Two models by profile

* Good, because it matches the two kinds of work the pipeline actually does
* Good, because the defaults stop describing a distinction that is not made
* Bad, because four steps now share one model's failure modes

### Four genuinely distinct models

* Good, because it keeps ADR-003's cost/quality story literally true
* Bad, because there is no fourth and third model that earns its slot today;
  differentiating for its own sake spends money to prove a point

### One model for everything

* Good, because it is the simplest thing that could work
* Bad, because speaker assignment is the step where a stronger model measurably
  changes the answer, and it is a small share of total tokens

### Revert to gpt-4.1 / o4-mini

* Good, because it is 46% cheaper in output tokens and 2.4x faster
* Bad, because it gives up `reasoning_effort` entirely, and the o-series
  parameter handling it required is what produced the prefix-guessing bug this
  decision removes

## Open Questions

- [x] Should `OPENAI_MODEL` and `OPENAI_LARGE_CONTEXT_MODEL` ever diverge again, or should `choose_appropriate_model()` and the large-context slot be removed outright once a model handles every episode length this project sees? — **Answered 2026-08-31 by User: Robert van den Breemen:** Deferred, not decided. The slot is kept for now because removing it is irreversible in a way keeping it is not: an env-var that nobody sets costs nothing, while deleting choose_appropriate_model() throws away the only place a length-based switch could be reinstated without touching call sites. Revisit when an episode is measured that gpt-5.6-luna handles worse than a differentiated model would, or when the slot has gone a full release unused on every machine that runs this pipeline.
- [x] Is `reasoning_effort=high` the right default for the four text steps, given it is 2.4x the wall-clock and the quality difference has not been adjudicated? — **Answered 2026-08-31 by User: Robert van den Breemen:** Deferred, and knowingly so. high was chosen before there was anything to compare it against, and the A/B measured tokens, latency and truncation rather than quality, so there is no evidence that lowering it would cost anything. What is measured is the price: 46 percent more output tokens and 2.4 times the wall-clock. The honest next step is a run at medium on the same episode with the outputs placed side by side for a human read, which nobody has done. Until then high stays because it is the setting the current measurements were taken at, not because it is known to be right.

## Related Decisions

* **ADR-003 (Per-task OpenAI model selection)**: superseded by this ADR. Its
  driver - match the model to the step - is kept; its four-way split and its
  assumption that `temperature` is a tuning knob are not.
* **ADR-004 (Hybrid AI-analysis plus programmatic speaker assignment)**: the
  reason speaker work keeps its own model. The model produces the mapping, code
  applies it.
* **ADR-005 (Recursive chunked summarization for long transcripts)**: the
  consumer of `choose_appropriate_model()` and of `MAX_TOKENS`.
* **ADR-007 (Environment-variable configuration)**: why the four slots survive
  even though the default collapses them to two.

## References

* `whycast/config.py:29-38`
* `whycast/pipeline/llm.py` `model_params`, `choose_appropriate_model`
* `tests/test_llm_model_params.py`, `tests/test_llm_chunked_payload.py`
* Measured parameter contract, 2026-08-30, against `api.openai.com/v1/chat/completions`:
  `max_tokens` -> 400 "Use 'max_completion_tokens' instead"; `temperature=0.7` ->
  400 "Only the default (1) value is supported"; `reasoning_effort` accepts
  `none|low|medium|high|xhigh` and rejects `max`
* Backlog TASK-007 and TASK-008

## Enforcement

```json
{
  "forbid_pattern": [
    {
      "pattern": "[\"']temperature[\"']\\s*[:=]",
      "path_glob": "whycast/**/*.py",
      "message": "These models reject any temperature other than 1; do not send the parameter (ADR-012)"
    },
    {
      "pattern": "startswith\\([\"']o[\"']\\)",
      "path_glob": "whycast/pipeline/llm.py",
      "message": "Do not infer a model's parameter shape from its name prefix; use model_params() (ADR-012)"
    }
  ],
  "forbid_import": [],
  "require_pattern": []
}
```
