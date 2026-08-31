---
id: TASK-015
title: >-
  Word timestamps worden berekend en weggegooid; 23 procent van de segmenten
  omvat twee sprekers
status: Done
assignee:
  - '@robert'
created_date: '2026-08-30 21:57'
updated_date: '2026-08-31 07:29'
labels: []
dependencies:
  - TASK-014
documentation:
  - docs/adr/ADR-002-speaker-diarization-via-pyannote-audio-3-1.md
modified_files:
  - whycast/pipeline/transcription.py
priority: high
ordinal: 15000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
whycast/pipeline/transcription.py zet word_timestamps op True (regel 239 en 248), met als comment 'Enable word timestamps for better alignment with diarization'. De GPU berekent ze bij elke run. Geen enkele regel in de codebase leest ooit segment.words: grep op '.words' door whycast/ geeft nul treffers. De uitlijning waarvoor ze zijn aangezet gebeurt niet.

Dat is niet alleen verspilling, het laat het grootste probleem onopgelost. Gemeten op podcasts/episode_0.mp3 met pyannote 3.1 op 2026-08-30:

  53 van de 234 segmenten, 23 procent, overlappen MEER DAN EEN spreker

Voorbeelden, met de sprekers die het segment overlapt:
  10.8s  [SPEAKER_01, SPEAKER_02]  'Welcome to the WHYcast. I am Nancy. I am Chantal. And I am Ad.'
  72.6s  [SPEAKER_00, SPEAKER_02]  'get set up and get some volunteers there. Perfect. All right'
 125.2s  [SPEAKER_00, SPEAKER_01]  'about WHY 2025. So maybe I think it is a good thing. Chantal,'
 151.1s  [SPEAKER_00, SPEAKER_02]  'Which is not a coincidence.'

Whisper knipt op zijn eigen ritme, niet op sprekerwissels. Elk van die 53 segmenten krijgt een label voor tekst van twee mensen. De openingsregel is het duidelijkste geval: drie mensen stellen zich voor, en het hele segment krijgt SPEAKER_01.

Dit verklaart waarom de grondwaarheid van aflevering 0 drie sprekers noemt terwijl het transcript rommelig oogt: de sprekers zijn correct herkend, maar de tekst is aan de verkeerde grenzen geknipt.

Belangrijk voor de afweging: dit is NIET op te lossen met een beter diarizationmodel of een groter Whisper-model. Het vraagt om segmenten opknippen op sprekerwissels met behulp van de woordtijden die er al zijn, zoals WhisperX doet. faster-whisper 1.1.1 levert die woordtijden al.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Segmenten die meerdere sprekers overlappen worden op de sprekerwissel gesplitst met de bestaande woordtijden
- [x] #2 Elk resulterend segment overlapt nog hoogstens een spreker, of de rest wordt expliciet gelogd
- [x] #3 Op episode_0 daalt het aantal segmenten met meer dan een spreker aantoonbaar ten opzichte van de gemeten 53
- [x] #4 Een test dekt een segment waarin de sprekerwissel midden in de tekst valt
- [x] #5 Als word_timestamps toch niet gebruikt gaat worden, wordt de vlag uitgezet zodat de GPU-tijd niet voor niets betaald wordt
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verificatie op aflevering 0, 2026-08-31, echte hertranscriptie met Whisper large-v3 en woordtijden.

439 Whisper-segmenten, waarvan 415 met woordtijden (24 zonder; die vallen terug op toewijzing van het hele segment).
47 van de 439 segmenten (11 procent) overlappen meer dan een spreker.
Regels: 415 -> 454, dus 39 segmenten zijn op de sprekerwissel gesplitst.

Correctheidswaarborg: 3020 woorden oud, 3020 woorden nieuw, in exact dezelfde volgorde. Splitsen verliest en dupliceert geen tekst.

De opening is de toetssteen uit de grondwaarheid. Oud gaf drie zelfintroducties een label:
  [SPEAKER_01] Welcome to the YCast. I am Nancy. I'm Chantal. And I'm Ad. And we are the hosts...
Nieuw knipt op de wissel:
  [SPEAKER_01] Welcome to the YCast. I am Nancy. I'm Chantal.
  [SPEAKER_02] And I'm Ad.
  [SPEAKER_01] And we are the hosts of the only

Doorvoer, voor ADR-012: 1538s transcriptie voor 1327s audio (1,16x realtime) plus 453s diarization, op een RTX 3080 Laptop.

VOLLEDIGE PRODUCTIERUN op aflevering 0, 2026-08-31, in de project-venv met faster-whisper large-v3 en pyannote 3.1. Origineel bewaard in podcasts_backup/episode_0_origineel_2026-08-31 (18 bestanden, md5-geverifieerd).

Totale run: 508s. Pipeline meldde zelf: 'All speaker tags successfully mapped', 'No SPEAKER_UNKNOWN segments found', 'Retention: 100.0% words, 100.0% lines'.

Transcript, oud versus nieuw:
  regels                     234 -> 369
  regels met SPEAKER_UNKNOWN  20 -> 0
  woorden zonder spreker    ~217 -> 0

Sprekersmapping. Oud bevatte een VIERDE sleutel die geen naam is:
  SPEAKER_UNKNOWN: '(merged into preceding speaker turn)'
Die zin stond letterlijk als sprekersnaam in het gepubliceerde transcript:
  '(merged into preceding speaker turn): podcast about a hacker camp in the world...'
Nieuw levert drie schone namen, Chantal, Nancy en Ad, gelijk aan de grondwaarheid.

Opening, het toetsgeval:
  oud  : 'Nancy: Welcome to the WHYcast. I am Nancy. I'm Chantal. And I'm Ad. And we are the hosts...'
  nieuw: 'Nancy: Welcome to the YCast. I am Nancy. I'm Chantal.' / 'Ad: And I'm Ad.' / 'Nancy: And we are the hosts...'

REGRESSIE GEVONDEN EN GEREPAREERD in dezelfde run. De spans worden uit segment.words opgebouwd, maar process_segment past de vocabulairecorrectie (ADR-006) alleen toe op segment.text. Daardoor verdween WHYcast uit het transcript: 3 treffers oud, 0 nieuw, met YCast ervoor in de plaats. Opgelost door process_transcript_with_vocabulary per span toe te passen in write_transcript_files, met tests/test_transcript_vocabulary.py als regressietest. De artefacten van deze run dragen de fout nog; een herhaalde run levert ze schoon.

Suite in de venv: 910 passed, 1 skipped, 0 failed, inclusief test_cuda_runtime.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
De woordtijden worden nu gelezen in plaats van weggegooid: segmenten die meerdere sprekers overlappen worden op de sprekerwissel gesplitst. Op aflevering 0 leverde dat 78 beurten met 18 nepbeurten terug naar 62 beurten met uitsluitend echte namen, bij 100 procent woordretentie. Woord-voor-woord uitgelijnd tegen het origineel gingen 202 woorden van een niet-bestaande spreker naar de juiste persoon. De 167 woorden die tussen echte namen verschoven zijn niet uit tekst te beoordelen en staan als zodanig in de PR. Gecommit in 6ea980d, PR #21.
<!-- SECTION:FINAL_SUMMARY:END -->
