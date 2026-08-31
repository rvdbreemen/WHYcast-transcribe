---
id: TASK-008
title: o-series modeldetectie is gedupliceerd en berust op een prefix-heuristiek
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-30 19:40'
updated_date: '2026-08-30 20:42'
labels: []
dependencies: []
modified_files:
  - whycast/pipeline/llm.py
priority: low
ordinal: 8000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
De OpenAI-aanroepen bepalen op twee plaatsen zelf of een model o-series is, om te kiezen tussen max_tokens en max_completion_tokens en om temperature wel of niet mee te sturen.

Feiten uit de code:
- whycast/pipeline/llm.py:258 is_o_series_model = model_name.startswith('o') and not model_name.startswith('gpt')
- whycast/pipeline/llm.py:340 exact dezelfde regel, los onderhouden

Twee problemen. De regel staat twee keer, dus een correctie op de ene plek drift weg van de andere. En de heuristiek raadt op basis van de eerste letter: elk toekomstig OpenAI-model dat met een o begint maar geen reasoning-model is, wordt verkeerd behandeld, en omgekeerd. De bestaande comment noemt gpt-4o al als uitzondering die handmatig is uitgesloten.

Uitkomst: een enkele functie met een expliciete lijst of een expliciete configuratie, zodat de keuze op een plek staat en controleerbaar is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 De o-series-bepaling staat op precies een plek en beide aanroeppaden gebruiken die
- [ ] #2 De bepaling berust op een expliciete lijst of config, niet op een prefix-gok
- [ ] #3 Een test dekt minimaal een o-series model, een gpt-model en een onbekend model
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Vervang is_o_series_model (llm.py:258 en :340) door een enkele helper _token_param(model_name) met een expliciete legacy-lijst (gpt-4, gpt-3.5) en max_completion_tokens als default voor al het overige.
2. Verwijder TEMPERATURE volledig: uit config.py, uit beide params-dicts, uit .env.example.
3. Voeg reasoning_effort toe als doorgegeven parameter aan process_with_openai en process_large_text_in_chunks; alleen meesturen wanneer er een waarde is.
4. Log finish_reason wanneer die 'length' is, zodat afgekapte output zichtbaar wordt.
5. Tests: token-parameter voor legacy, modern en onbekend model; geen temperature in de payload; reasoning_effort wel/niet meegestuurd.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gemeten contract via de live API op 2026-08-30 voor gpt-5.6-luna en gpt-5.6-sol: max_tokens geeft 400 (gebruik max_completion_tokens), temperature != 1 geeft 400, reasoning_effort accepteert none/low/medium/high/xhigh en NIET max. De prefix-heuristiek classificeert gpt-5.6-* als niet-o-series, waardoor beide foute parameters worden verstuurd. Daarmee blokkeert deze taak TASK-007.

Geimplementeerd: model_params() in llm.py vervangt de prefix-heuristiek op beide aanroeppaden; LEGACY_CHAT_MODEL_PREFIXES noemt de uitzonderingen en max_completion_tokens is de default. TEMPERATURE volledig verwijderd uit config.py en llm.py. reasoning_effort doorgegeven via process_with_openai en process_large_text_in_chunks. _warn_if_truncated() logt finish_reason=length. Nieuwe tests in tests/test_llm_model_params.py. Suite: 847 passed, 1 skipped (was 827); test_cuda_runtime blijft falen op de OpenMP-omgevingsfout uit TASK-009.

Chunk-route geverifieerd met tests/test_llm_chunked_payload.py (gemockte client): beide payloadvormen in process_large_text_in_chunks sturen max_completion_tokens plus reasoning_effort en nooit een temperature, en een legacy model krijgt daar nog steeds max_tokens zonder reasoning_effort. Live verificatie van die route is niet gelukt: drie runs liepen in een time-out, oorzaak is TASK-013 (split_into_chunks maakt ~1000 loze staartchunks). De payloadvorm is dus bewezen, de doorlooptijd van die route niet.

Daarnaast de bestaande interpunctie-heuristiek in process_with_openai achter finish_reason != 'stop' gezet. Die vuurde in de A/B-run vier keer vals: drie sprekersaanroepen in de oude arm en een samenvatting in de nieuwe, alle vier met finish_reason=stop. Het antwoord eindigde daar op een tabelrij of een code-fence. Naast het gezaghebbende signaal was dat misleidend.

Suite na alle wijzigingen: 852 passed, 1 skipped.
<!-- SECTION:NOTES:END -->
