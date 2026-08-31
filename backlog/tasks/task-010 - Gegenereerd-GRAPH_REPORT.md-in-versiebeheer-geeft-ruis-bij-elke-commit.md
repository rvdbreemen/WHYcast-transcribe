---
id: TASK-010
title: Gegenereerd GRAPH_REPORT.md in versiebeheer geeft ruis bij elke commit
status: To Do
assignee: []
created_date: '2026-08-30 19:40'
labels: []
dependencies: []
modified_files:
  - .gitignore
priority: low
ordinal: 10000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Op branch docs/graphify-knowledge-graph is graphify-out/GRAPH_REPORT.md onder versiebeheer gebracht. Het .gitignore-patroon is correct git: graphify-out/* met daarna !graphify-out/GRAPH_REPORT.md, want een negatie werkt niet als de map zelf genegeerd is.

De afweging staat in de comment: 70 KB leesbare tekst is de moeite waard in een diff, terwijl graph.html en graph.json 4 MB machineleesbare blob zijn. Dat klopt, maar heeft een prijs. De projectinstructie in CLAUDE.md schrijft voor om na elke codewijziging 'graphify update .' te draaien, en het bestand wordt daarbij opnieuw gegenereerd. Elke ongerelateerde commit sleept daardoor een diff van honderden regels mee, wat review verzwaart en merge-conflicten uitlokt.

Uitkomst: bepalen wanneer het rapport wordt bijgewerkt. Denk aan alleen bij bewuste regeneratie in een eigen commit, of via een pre-commit-guard die het rapport buiten gemengde commits houdt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Er is een vastgelegde afspraak wanneer GRAPH_REPORT.md wordt gecommit
- [ ] #2 Een ongerelateerde codewijziging kan worden gecommit zonder de rapport-diff mee te nemen
- [ ] #3 De afspraak staat in CLAUDE.md naast de bestaande graphify-regels
<!-- AC:END -->
