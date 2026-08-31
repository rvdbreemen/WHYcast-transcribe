---
id: TASK-016
title: >-
  Custom vocabulary stuurt de transcriptie niet: word_list bestaat niet in
  faster-whisper
status: Done
assignee:
  - '@robert'
created_date: '2026-08-30 22:05'
updated_date: '2026-08-31 08:29'
labels: []
dependencies: []
documentation:
  - >-
    docs/adr/ADR-006-post-transcription-vocabulary-correction-via-vocabulary-json.md
modified_files:
  - whycast/pipeline/transcription.py
priority: medium
ordinal: 16000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
whycast/pipeline/transcription.py:339 roept model.transcribe(..., word_list=word_list) aan om de 111 termen uit vocabulary.json aan Whisper mee te geven. Die parameter bestaat niet.

Geverifieerd tegen de geinstalleerde faster-whisper 1.1.1:
  inspect.signature(WhisperModel.transcribe) -> 'word_list' in parameters: False
                                                'hotwords'  in parameters: True

vocabulary.json bestaat en bevat 111 termen, dus deze tak draait bij elke run. Het patroon eromheen vangt de TypeError op, logt 'word_list parameter not supported in this version of faster_whisper' en draait daarna opnieuw zonder vocabulaire. Het resultaat is dat Whisper nooit gestuurd wordt: elke transcriptie kost een mislukte opzet plus een waarschuwing, en de biasing die bedoeld was gebeurt niet.

Wat wel werkt is de post-hoc vervanging in process_segment (ADR-006), die 'Rai' naar 'WHY' corrigeert nadat Whisper het fout heeft verstaan. Het verschil is niet cosmetisch: hotwords zou voorkomen dat het fout verstaan wordt, waar de vervanging alleen repareert wat exact matcht.

Tweede punt in dezelfde functie: base_transcription_params (regel 237) wordt opgebouwd en nooit gebruikt. Het is een identieke kopie van transcription_params (regel 245). Een van de twee moet weg.

Bewust NIET meegenomen in TASK-014 en TASK-015: die meten sprekertoewijzing op aflevering 0, en het wijzigen van transcriptieparameters zou de vergelijking vervuilen. Dit hoort in een eigen meting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 De vocabulairetermen worden via hotwords meegegeven, of er is onderbouwd waarom niet
- [x] #2 De TypeError-vangst rond een niet-bestaande parameter is weg
- [x] #3 base_transcription_params is verwijderd
- [ ] #4 Op aflevering 0 is gemeten of hotwords het aantal post-hoc vervangingen terugdringt
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
word_list verwijderd, inclusief de TypeError-vangst eromheen. Die parameter bestaat niet in faster-whisper 1.1.1, dus elke run raakte hem kwijt aan een waarschuwing en draaide opnieuw zonder vocabulaire. base_transcription_params, een identieke ongebruikte kopie van transcription_params, is ook weg.

hotwords is BEWUST NIET gebruikt, op expliciete beslissing van Robert. De onderbouwing is gemeten: hotwords deelt Whispers promptvenster van 224 tokens met initial_prompt, en de 77 unieke termen uit vocabulary.json coderen tot 285 tokens (gemeten met de tokenizer van whisper-large-v3; 815 tekens, 2,86 tekens per token). Ze passen er vandaag al niet in, en zouden bij groei stil wegvallen. De vervangingsmap kent die grens niet en blijft dus schaalbaar uitbreidbaar.

De correctie gebeurt op twee plekken met dezelfde bron: per segment in process_segment voor de live-uitvoer, en per span in write_transcript_files voor de geschreven artefacten. Gemeten kosten van de tweede: 105 ms voor 61 spans, met vocabulary.json eenmaal ingelezen dankzij de bestaande module-cache. Verwaarloosbaar tegen een run van 500 seconden.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
word_list verwijderd, inclusief de TypeError-vangst eromheen, plus de dode kopie base_transcription_params. hotwords is bewust niet gebruikt: gemeten met de tokenizer van whisper-large-v3 coderen de 77 unieke termen tot 285 tokens terwijl Whispers promptvenster er 224 telt, dus ze passen nu al niet en zouden bij groei stil wegvallen. De vervangingsmap kent die grens niet. De correctie draait per segment op het live-pad en per span in write_transcript_files, gemeten op 105 ms voor 61 spans met het vocabulaire eenmaal ingelezen.
<!-- SECTION:FINAL_SUMMARY:END -->
