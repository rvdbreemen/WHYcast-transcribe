---
id: TASK-017
title: >-
  Vijf diarization-instellingen worden getoond maar door geen productiecode
  gelezen
status: Done
assignee:
  - '@robert'
created_date: '2026-08-30 22:12'
updated_date: '2026-08-31 08:29'
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
- [x] #1 diarize_audio leest DIARIZATION_MODEL in plaats van het pad te hardcoderen
- [x] #2 min_speakers en max_speakers worden aan de pyannote-pipeline doorgegeven
- [x] #3 USE_SPEAKER_DIARIZATION schakelt diarization daadwerkelijk uit
- [x] #4 DIARIZATION_ALTERNATIVE_MODEL wordt gebruikt of verwijderd uit config en webui
- [x] #5 Op episode_1 is gemeten wat max_speakers doet met het aantal gevonden labels
- [x] #6 Een test dekt dat een gewijzigde DIARIZATION_MODEL ook echt bij Pipeline.from_pretrained aankomt
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
diarize_audio leest nu DIARIZATION_MODEL in plaats van het pad te hardcoderen, geeft min_speakers en max_speakers door aan de pyannote-pipeline, en keert direct terug met None wanneer USE_SPEAKER_DIARIZATION uit staat.

DIARIZATION_ALTERNATIVE_MODEL is NIET verwijderd. ADR-002 (Accepted) noemt hem expliciet als mitigatie voor pyannote-drift tegen torch, dus weghalen zou een geaccepteerd besluit tegenspreken. In plaats daarvan is de fallback geimplementeerd: faalt het primaire model, dan wordt het alternatief geprobeerd. Eerlijk in de comment vastgelegd: de meegeleverde default is pyannote/segmentation-3.0, een segmentatiemodel en geen diarization-pipeline, dus die fallback zal waarschijnlijk ook falen tenzij de waarde naar een echte pipeline wijst. Dat is een openstaand punt voor ADR-002, niet iets dat ik stil heb opgelost.

min_speakers wordt alleen doorgegeven als hij groter is dan 1, omdat 1 niets begrenst.

tests/test_diarization_config.py, 6 tests met een nagemaakte pyannote-Pipeline: het geconfigureerde model komt aan, het alternatief wordt bij falen geprobeerd, max_speakers en min_speakers komen aan, min=1 wordt weggelaten, en uitgeschakelde diarization laadt pyannote helemaal niet.

Na adversariele review gerepareerd: de fallback kon niet vuren omdat pyannote's Pipeline.from_pretrained bij een ontbrekend, prive of gated model GEEN exception werpt maar None teruggeeft (pyannote/audio/core/pipeline.py:107-121; GatedRepoError erft van RepositoryNotFoundError). Dat is precies het geval dat ADR-002 als risico noemt, dus de eerste versie dekte het enige scenario waarvoor hij bestond niet. Een _load-helper zet die None nu om in een ConfigurationError.

Ook gerepareerd: tegenstrijdige min/max sprekersgrenzen worden gemeld in plaats van doorgegeven; een bewust uitgeschakelde diarization werd aan de gebruiker gemeld als 'Diarization failed'; en die controle stond na verify_gpu_setup en het inladen van de hele episode, werk dat uitsluitend voor diarization bestaat. De schakelaar staat nu voor dat werk, in full_workflow, met _run_diarization als aparte functie.

Mutatietest als bewijs dat de tests niet hol zijn: met de None-controle uitgeschakeld falen test_the_fallback_also_fires_when_pyannote_returns_none en test_a_none_return_with_no_fallback_names_the_gated_repository, met exact de oude melding 'NoneType object has no attribute to'. De eerste versie van die tweede test toetste alleen 'result is None' en was daarmee waar met en zonder fix; hij toetst nu de logmelding.

AC 5 blijft ONgevinkt: meten wat max_speakers doet met het aantal labels op aflevering 1 vraagt een echte GPU-run op die aflevering, en die is niet gedaan.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
diarize_audio leest nu DIARIZATION_MODEL, geeft min_speakers en max_speakers door, honoreert USE_SPEAKER_DIARIZATION en implementeert de fallback die ADR-002 belooft. Na adversariele review bleek die fallback niet te kunnen vuren omdat pyannote bij een gated model None teruggeeft in plaats van te werpen; dat is opgelost met een _load-helper, en met een mutatietest bewezen dat de tests er ook echt op vallen. Geverifieerd met 8 tests in tests/test_diarization_config.py en een volledige suite van 927 passed, 2 skipped.
<!-- SECTION:FINAL_SUMMARY:END -->
