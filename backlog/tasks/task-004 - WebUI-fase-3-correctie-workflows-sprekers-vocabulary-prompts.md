---
id: TASK-004
title: 'WebUI fase 3: correctie-workflows (sprekers, vocabulary, prompts)'
status: To Do
assignee: []
created_date: '2026-08-24 19:59'
updated_date: '2026-08-25 17:43'
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
- [ ] #1 Speaker-mapping aanpasbaar in de UI en programmatisch toegepast op de artefacten van een aflevering
- [ ] #2 Vocabulary bewerken en toepassen op een gekozen aflevering werkt vanuit de UI
- [ ] #3 Prompts bewerkbaar en losse pipelinestap opnieuw uitvoerbaar per aflevering
- [ ] #4 Handmatige correcties gaan naar de INPUT van de pipeline (spreker-mapping, vocabulary.json, promptbestand), nooit naar een gegenereerd artefact - een her-run reproduceert de correctie in plaats van hem te overschrijven (ADR-009)
- [ ] #5 Bewerkte inputbestanden krijgen een .bak-backup bij overschrijven (dit is mensenwerk, in tegenstelling tot artefacten die altijd regenereerbaar zijn)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-25 (ADR-009, beslissing Robert): artefacten krijgen geen .bak meer, dus mag menselijk werk daar niet in landen. De spreker-editor bewerkt de mapping, niet het gegenereerde _speaker_assignment-bestand. Sluit aan op ADR-004 (oordeel gescheiden van toepassing). Oorspronkelijke AC#4 ('elk overschreven bestand krijgt .bak') vervangen door twee criteria die dit onderscheid maken. Bekende beperking, expliciet geaccepteerd: een correctie die niet als mapping uit te drukken is (losse zin herschrijven) past niet in dit model en valt buiten fase 3.
<!-- SECTION:NOTES:END -->
