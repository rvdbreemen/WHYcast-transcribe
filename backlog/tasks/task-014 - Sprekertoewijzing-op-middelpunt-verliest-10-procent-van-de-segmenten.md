---
id: TASK-014
title: Sprekertoewijzing op middelpunt verliest 10 procent van de segmenten
status: In Progress
assignee:
  - '@robert'
created_date: '2026-08-30 21:57'
updated_date: '2026-08-30 23:31'
labels: []
dependencies: []
documentation:
  - docs/adr/ADR-002-speaker-diarization-via-pyannote-audio-3-1.md
modified_files:
  - whycast/pipeline/transcription.py
priority: high
ordinal: 14000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
find_speaker_for_segment in whycast/pipeline/transcription.py (regel 263 en nogmaals 389, twee identieke kopieen) koppelt een Whisper-segment aan een spreker door te toetsen of het MIDDELPUNT van dat segment binnen een diarization-turn valt. Valt het middelpunt in een gat tussen turns, dan geeft de functie None en wordt het segment [SPEAKER_UNKNOWN].

Gemeten op 2026-08-30, pyannote/speaker-diarization-3.1 opnieuw gedraaid op podcasts/episode_0.mp3 (RTX 3080, 727s):
- diarization vindt 296 turns en precies 3 sprekers, wat overeenkomt met de grondwaarheid die Robert gaf: Nancy, Chantal, Ad
- 215 gaten tussen turns, samen 334 seconden

Toewijzing over de 234 segmenten van het bestaande transcript:
- middelpunt (huidige code): 23 segmenten zonder spreker, 10 procent
- maximale overlap in tijd:    0 segmenten zonder spreker, 0 procent
- verschil: 23 segmenten en 263 woorden

Het diarizationmodel is hier dus niet het probleem. Het vindt het juiste aantal sprekers; onze toewijzing gooit 10 procent weg.

Tweede defect in hetzelfde blok, regel 425 en verder:

    if speaker:
        is_short_utterance = segment_duration < 1.0 and len(text.split()) <= 5
        if is_short_utterance and not speaker and current_speaker:
            speaker = current_speaker
        if speaker:
            ...
        else:
            speaker_info = None-tak

De continuiteitsregel voor korte uitingen staat binnen 'if speaker:', dus 'not speaker' is daar altijd onwaar. De regel draait nooit en de else-tak eronder is onbereikbaar. Bedoeld was juist het omgekeerde geval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Toewijzing gebeurt op maximale overlap in tijd, niet op het middelpunt
- [ ] #2 De functie staat op een plek in plaats van twee identieke kopieen
- [ ] #3 De onbereikbare continuiteitstak is weg of verplaatst naar de tak waar hij wel kan vuren
- [ ] #4 Een test dekt: middelpunt in een gat tussen twee turns, segment volledig binnen een turn, en segment zonder enige overlap
- [ ] #5 Op episode_0 levert de nieuwe toewijzing 0 segmenten met SPEAKER_UNKNOWN
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verificatie op aflevering 0, 2026-08-31, met een ECHTE hertranscriptie (Whisper large-v3, openai-whisper, GPU) en pyannote 3.1. Beide toewijzingen berekend op dezelfde Whisper-uitvoer, zodat het verschil de attributie is en niet de ruis tussen twee runs.

Diarization vond 321 turns en precies 3 sprekers, gelijk aan de grondwaarheid die Robert gaf (Nancy, Chantal, Ad). Het model was dus nooit de fout.

SPEAKER_UNKNOWN, middelpunt versus maximale overlap:
  oud  : 36 regels, 230 woorden zonder spreker
  nieuw: 0 regels, 0 woorden
230 woorden teruggewonnen, 100 procent van het verlies.
<!-- SECTION:NOTES:END -->
