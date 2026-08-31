---
id: TASK-007
title: Per-taak modelkeuze uit ADR-003 is uitgeschakeld door identieke defaults
status: To Do
assignee: []
created_date: '2026-08-30 19:39'
updated_date: '2026-08-30 20:09'
labels: []
dependencies:
  - TASK-008
documentation:
  - docs/adr/ADR-003-per-task-openai-model-selection.md
modified_files:
  - whycast/config.py
  - whycast/pipeline/llm.py
priority: medium
ordinal: 7000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-003 kiest voor per-taak OpenAI-modellen met een lengte-afhankelijke omschakeling, omdat elke stap een ander kosten/kwaliteitsprofiel heeft. In de huidige defaults valt dat mechanisme stil.

Feiten uit de code:
- whycast/config.py:29 OPENAI_MODEL default gpt-4.1
- whycast/config.py:30 OPENAI_LARGE_CONTEXT_MODEL default gpt-4.1
- whycast/config.py:31 OPENAI_HISTORY_MODEL default gpt-4.1
- whycast/config.py:32 OPENAI_SPEAKER_MODEL default o4-mini

Drie van de vier env-vars hebben dezelfde default. Daardoor levert choose_appropriate_model() in whycast/pipeline/llm.py:135-139 hetzelfde model op, ongeacht de geschatte transcriptlengte: de tak logt 'Using large context model' en retourneert precies wat de andere tak ook zou retourneren. Alleen de sprekersstap wijkt echt af.

Uitkomst: beslis of de defaults uiteen moeten lopen zoals ADR-003 beoogt, of dat het ADR moet worden bijgewerkt naar wat er draait. Beide zijn verdedigbaar, maar de huidige situatie belooft iets wat niet gebeurt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 De defaults in config.py sluiten aan bij ADR-003, of ADR-003 is bijgewerkt of gesuperseded
- [ ] #2 De log-regel in llm.py:136 meldt alleen een groot-contextmodel wanneer dat daadwerkelijk afwijkt van OPENAI_MODEL
- [ ] #3 De .env-documentatie noemt per env-var waarvoor het model wordt gebruikt en wat een zinnige afwijkende waarde is
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Defaults omgezet naar gpt-5.6-luna (algemeen, groot-context, history) en gpt-5.6-sol (sprekers), beide reasoning_effort=high via OPENAI_REASONING_EFFORT en OPENAI_SPEAKER_REASONING_EFFORT. A/B-run 2026-08-30 door het echte productiepad, gemeten over 5 aanroepen per arm op dezelfde invoer: oud 34081 in / 15716 uit / 122s totaal; nieuw 34079 in / 22955 uit waarvan 13428 reasoning / 292s totaal. Nieuw kost dus circa 46 procent meer outputtokens en 2,4 maal de wandkloktijd. Geen enkele aanroep werd afgekapt (finish_reason=stop overal), dus MAX_TOKENS=16000 volstaat bij deze transcriptlengtes. De lengte-switch in choose_appropriate_model blijft een no-op zolang OPENAI_MODEL en OPENAI_LARGE_CONTEXT_MODEL gelijk zijn; dat hoort in het opvolgende ADR.
<!-- SECTION:NOTES:END -->
