---
id: TASK-013
title: 'split_into_chunks maakt ~1000 loze staartchunks, elk een betaalde API-aanroep'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-30 20:38'
updated_date: '2026-08-30 20:55'
labels: []
dependencies: []
documentation:
  - docs/adr/ADR-005-recursive-chunked-summarization-for-long-transcripts.md
modified_files:
  - whycast/pipeline/llm.py
priority: high
ordinal: 13000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
whycast/pipeline/llm.py:229 sluit de lus af met:

    start = max(start + 1, end - overlap)

Er is geen afbreekconditie voor de laatste chunk. Zodra end gelijk is aan len(text) springt start naar len(text) - overlap, wat nog steeds kleiner is dan len(text). De lus draait dan door: end blijft len(text), end - overlap blijft constant en kleiner dan start, dus start schuift nog maar 1 teken per ronde op. Dat levert precies CHUNK_OVERLAP extra chunks op van aflopende lengte, 1000, 999, 998 ... tot 1.

Gemeten op 2026-08-30, zonder API:
  split_into_chunks('y'*100000, max_chunk_size=80000, overlap=1000) -> 1002 chunks
  lengtes: 80000, 21000, 1000, 999, 998, ..., 2, 1

Twee correcte chunks, gevolgd door duizend restjes.

Beide aanroepers draaien per chunk een OpenAI-aanroep:
- llm.py:248 summarize_large_transcript (ADR-005, recursieve samenvatting)
- llm.py:395 process_large_text_in_chunks (opschonen van lange transcripten)

Gevolg: een transcript dat de chunk-route haalt kost ruwweg 1000 API-aanroepen in plaats van een handvol. Dit is bereikbaar met bestaand materiaal: podcasts/episode_46_ts.txt is 135213 tekens en overschrijdt de drempel MAX_INPUT_TOKENS // 2 in de opschoonstap.

De bug is niet nieuw en staat los van de modelmigratie, maar hij blokkeerde wel de verificatie van de chunk-route: drie live testruns liepen in een time-out omdat de code duizend aanroepen ging doen.

Vermoedelijke fix is een afbreekconditie zodra end de tekstlengte bereikt. Dat moet de opsteller verifieren, niet klakkeloos overnemen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 split_into_chunks levert voor een tekst van 100000 tekens bij max_chunk_size 80000 en overlap 1000 precies twee chunks
- [x] #2 Geen enkele chunk is korter dan de overlap, tenzij de hele tekst dat is
- [x] #3 De chunks dekken samen de volledige tekst, met overlap tussen opeenvolgende chunks
- [x] #4 Een test dekt de randgevallen: tekst korter dan max_chunk_size, exact gelijk aan max_chunk_size, en net erboven
- [x] #5 process_large_text_in_chunks doet voor episode_46_ts.txt een aantal aanroepen in de orde van het aantal echte chunks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduceer eerst zonder API en leg de huidige uitvoer vast als test die faalt (rode test).
2. Voeg in split_into_chunks een afbreekconditie toe zodra end de tekstlengte bereikt, direct na het toevoegen van de chunk. De bestaande max(start+1, ...) blijft staan als bescherming tegen een niet-opschuivende start wanneer de chunk korter is dan de overlap.
3. Toets de randgevallen: tekst korter dan, gelijk aan en net boven max_chunk_size; een tekst zonder zinsafbrekingen; een tekst met alineagrenzen.
4. Toets dat de chunks samen de hele tekst dekken en dat opeenvolgende chunks overlappen.
5. Verifieer het effect op de aanroepers met een gemockte client: het aantal API-aanroepen moet in de orde van het aantal echte chunks liggen.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fix: afbreekconditie 'if end >= len(text): break' direct na het toevoegen van de chunk (whycast/pipeline/llm.py). De bestaande max(start+1, end-overlap) blijft staan voor het geval een gekozen breekpunt binnen de overlap van start valt.

TDD gevolgd: tests/test_llm_chunking.py eerst rood (11 failed, 5 passed), na de fix groen (16 passed).

Effect gemeten met een gemockte client op podcasts/episode_46_ts.txt (133069 tekens, 2 echte chunks): process_large_text_in_chunks doet nu 3 API-aanroepen (2 chunks + 1 consistentiepas). Voor de fix zouden dat er 2 + CHUNK_OVERLAP = ~1002 zijn geweest.

Let op, bewuste gedragswijziging: tests/golden/split_into_chunks.snapshot.txt bevroor het buggy gedrag, inclusief de staart 6, 5, 4, 3, 2, 1. transcribe.py is sinds ADR-008 een dunne shim, dus oud == nieuw en alleen de snapshot pinde dit nog. Snapshot geregenereerd; long_2000_200 gaat van 216 naar 16 chunks, long_800_80 van 140 naar 60. De oude snapshot staat als back-up in de scratchpad van deze sessie. Elke resterende chunk is langer dan de overlap.

Volledige suite: 868 passed, 1 skipped, test_cuda_runtime gedeselecteerd (TASK-009).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
split_into_chunks liep door nadat de tekst gedekt was en produceerde precies CHUNK_OVERLAP extra staartchunks van aflopende lengte, elk goed voor een betaalde OpenAI-aanroep in summarize_large_transcript en process_large_text_in_chunks. Toegevoegd: een afbreekconditie zodra end de tekstlengte bereikt. Geverifieerd met tests/test_llm_chunking.py (16 tests, eerst rood dan groen) en met een gemockte meting op episode_46_ts.txt, waar het aantal API-aanroepen van ~1003 naar 3 gaat. De golden snapshot die het oude gedrag vastlegde is bewust geregenereerd.
<!-- SECTION:FINAL_SUMMARY:END -->
