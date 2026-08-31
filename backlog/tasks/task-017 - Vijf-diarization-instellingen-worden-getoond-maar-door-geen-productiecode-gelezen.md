---
id: TASK-017
title: >-
  Vijf diarization-instellingen worden getoond maar door geen productiecode
  gelezen
status: To Do
assignee: []
created_date: '2026-08-30 22:12'
labels: []
dependencies: []
documentation:
  - docs/adr/ADR-002-speaker-diarization-via-pyannote-audio-3-1.md
modified_files:
  - whycast/pipeline/diarization.py
  - whycast/config.py
priority: high
ordinal: 17000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
whycast/config.py definieert vijf instellingen voor diarization. webui/app.py:2846-2850 toont ze alle vijf in de config-viewer, en tests/test_phase4_diff_and_config.py:196 legt vast dat DIARIZATION_MODEL daar verschijnt. Geen enkele productieregel leest er ook maar een van.

Geverifieerd met grep over alle .py in de repo op 2026-08-31. De enige treffers zijn de definitie in config.py, de lijst in de webui, en die ene test:

  USE_SPEAKER_DIARIZATION       config.py:53  webui:2846  -- nergens gelezen
  DIARIZATION_MODEL             config.py:54  webui:2847  -- nergens gelezen
  DIARIZATION_ALTERNATIVE_MODEL config.py:56  webui:2848  -- nergens gelezen
  DIARIZATION_MIN_SPEAKERS      config.py:57  webui:2849  -- nergens gelezen
  DIARIZATION_MAX_SPEAKERS      config.py:58  webui:2850  -- nergens gelezen

whycast/pipeline/diarization.py:174 hardcodeert in plaats daarvan:

    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1', use_auth_token=hf_token)

en roept de pipeline aan zonder num_speakers, min_speakers of max_speakers.

Drie gevolgen, oplopend in ernst:

1. De schakelaar USE_SPEAKER_DIARIZATION doet niets. Diarization draait altijd.
2. Een ander diarizationmodel proberen kan niet via configuratie, alleen door code te wijzigen. Dat is precies wat de env-var belooft.
3. De sprekersgrenzen worden niet doorgegeven. pyannote accepteert min_speakers en max_speakers en gebruikt die om clustering te beperken. Op podcasts/episode_1_merged.txt leverde diarization zes genummerde labels op voor waarschijnlijk vier mensen, met SPEAKER_00 op 6 segmenten en SPEAKER_03 op 36. Dat is precies het geval waarin een bovengrens helpt.

De web-UI toont dus vijf knoppen die niets doen, en een test bevestigt dat ze getoond worden.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 diarize_audio leest DIARIZATION_MODEL in plaats van het pad te hardcoderen
- [ ] #2 min_speakers en max_speakers worden aan de pyannote-pipeline doorgegeven
- [ ] #3 USE_SPEAKER_DIARIZATION schakelt diarization daadwerkelijk uit
- [ ] #4 DIARIZATION_ALTERNATIVE_MODEL wordt gebruikt of verwijderd uit config en webui
- [ ] #5 Op episode_1 is gemeten wat max_speakers doet met het aantal gevonden labels
- [ ] #6 Een test dekt dat een gewijzigde DIARIZATION_MODEL ook echt bij Pipeline.from_pretrained aankomt
<!-- AC:END -->
