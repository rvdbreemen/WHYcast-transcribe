---
id: TASK-016
title: >-
  Custom vocabulary stuurt de transcriptie niet: word_list bestaat niet in
  faster-whisper
status: To Do
assignee: []
created_date: '2026-08-30 22:05'
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
- [ ] #2 De TypeError-vangst rond een niet-bestaande parameter is weg
- [ ] #3 base_transcription_params is verwijderd
- [ ] #4 Op aflevering 0 is gemeten of hotwords het aantal post-hoc vervangingen terugdringt
<!-- AC:END -->
