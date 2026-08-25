---
id: TASK-004
title: 'WebUI fase 3: correctie-workflows (sprekers, vocabulary, prompts)'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24 19:59'
updated_date: '2026-08-25 21:34'
labels:
  - webui
  - frontend
dependencies:
  - TASK-003
references:
  - docs/adr/ADR-008-web-ui-as-fastapi-layer-over-extracted-pipeline-library.md
documentation:
  - backlog/docs/doc-001 - WebUI-plan-WHYcast-transcribe.md
ordinal: 4000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correctie-workflows die de losse batch-scripts vervangen: speaker-editor op mapping-niveau (hernoemen en programmatisch toepassen via het ADR-004-pad, AI-suggesties ter review), vocabulary-editor met per-aflevering toepassen, prompt-editor met her-run per stap. Elk schrijfpad maakt eerst een .bak-backup. Granulariteit (mapping- vs segment-niveau) is een open vraag in ADR-008; beslis bij oppakken. Zie doc-001 fase 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Speaker-mapping aanpasbaar in de UI en programmatisch toegepast op de artefacten van een aflevering
- [x] #2 Vocabulary bewerken en toepassen op een gekozen aflevering werkt vanuit de UI
- [x] #3 Prompts bewerkbaar en losse pipelinestap opnieuw uitvoerbaar per aflevering
- [x] #4 Handmatige correcties gaan naar de INPUT van de pipeline (spreker-mapping, vocabulary.json, promptbestand), nooit naar een gegenereerd artefact - een her-run reproduceert de correctie in plaats van hem te overschrijven (ADR-009)
- [x] #5 Bewerkte inputbestanden krijgen een .bak-backup bij overschrijven (dit is mensenwerk, in tegenstelling tot artefacten die altijd regenereerbaar zijn)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Nieuwe pipeline-INPUT: <base>_speakers.json per aflevering (ADR-010, amendeert ADR-004). speaker_assignment_step leest hem als hij bestaat en slaat fase 1 (betaalde o4-analyse) over; ontbreekt hij, dan draait de LLM zoals nu EN wordt de afgeleide mapping weggeschreven zodat hij bewerkbaar wordt. Bijvangst: een her-run met bestaande mapping is gratis.
2. Scanner: .json-formaat en soort 'speakers_map' toevoegen, anders belandt het inputbestand in unmatched.
3. webui speaker-editor: toont SPEAKER_xx uit het transcript met contextfragmenten, laat namen invullen, schrijft de mapping (backup=True want mensenwerk, ADR-009), en biedt de speakers-job aan.
4. webui vocabulary-editor: vocabulary.json bewerken (globaal) + per aflevering toepassen.
5. webui prompt-editor: prompts/*.txt bewerken + losse stap opnieuw draaien per aflevering.
6. Alle input-writes via atomic_write_text met backup=True; artefacten blijven backup=False.
7. Tests: mapping-bestand wint van LLM, ontbrekend bestand valt terug op LLM en persisteert de mapping, her-run met mapping doet geen betaalde call, scanner herkent speakers_map, editor-routes met path-traversal-guards, .bak op inputbestanden.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-25 (ADR-009, beslissing Robert): artefacten krijgen geen .bak meer, dus mag menselijk werk daar niet in landen. De spreker-editor bewerkt de mapping, niet het gegenereerde _speaker_assignment-bestand. Sluit aan op ADR-004 (oordeel gescheiden van toepassing). Oorspronkelijke AC#4 ('elk overschreven bestand krijgt .bak') vervangen door twee criteria die dit onderscheid maken. Bekende beperking, expliciet geaccepteerd: een correctie die niet als mapping uit te drukken is (losse zin herschrijven) past niet in dit model en valt buiten fase 3.

Gebouwd via workflow (8 agents). Zelf geverifieerd op de draaiende server: alle vier voorrangstakken (opgeslagen mapping past toe zonder enige betaalde aanroep - bewezen door analyze_speakers_with_o4 te vervangen door iets dat gooit; stale weigert met bruikbare melding; kapot bestand weigert; ontbrekend bestand valt terug op LLM en persisteert). Editor vond 5 SPEAKER-labels in episode_13, opslaan gaf .bak bij tweede keer, next_run sprong van 'model' naar 'saved', scanner toonde speakers_map als invoer, discard verwijderde de mapping. Traversal -> 404, geen secrets in prompts-API. Testdata daarna opgeruimd zodat podcasts/ ongewijzigd bleef. 790 passed / 1 skipped (volledig). Twee losse eindjes uit de review zelf gefixt: json-mediatype ontbrak in webui/app.py en de fmt-validatie gebruikte de globale ARTIFACT_FORMATS in plaats van formats_for_kind.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Correctie-workflows opgeleverd. Spreker-mapping is nu een persistente pipeline-invoer (<base>_speakers.json, ADR-010): aanwezig wint van het model en slaat de betaalde analyse over, ontbrekend laat het model draaien en bewaart het antwoord, kapot of verouderd weigert met een melding die het bestand noemt en beide uitwegen geeft. Nooit automatisch verwijderd. Vocabulary- en prompt-editor volgen dezelfde regel: bewerk wat de pipeline leest, atomisch met .bak, promptlijst als vaste allowlist. Geverifieerd met 790 passed/1 skipped en een handmatige end-to-end sessie op de draaiende server.
<!-- SECTION:FINAL_SUMMARY:END -->
