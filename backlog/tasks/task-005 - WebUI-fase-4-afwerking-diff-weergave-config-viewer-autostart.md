---
id: TASK-005
title: 'WebUI fase 4: afwerking (diff-weergave, config-viewer, autostart)'
status: To Do
assignee: []
created_date: '2026-08-24 19:59'
labels:
  - webui
dependencies:
  - TASK-004
references:
  - docs/adr/ADR-008-web-ui-as-fastapi-layer-over-extracted-pipeline-library.md
documentation:
  - backlog/docs/doc-001 - WebUI-plan-WHYcast-transcribe.md
priority: low
ordinal: 5000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Optionele afwerking na fase 3, op volgorde van nut: diff-weergave oud/nieuw artefact na her-run, config-viewer met allowlist van toonbare keys (secrets nooit, ook niet gemaskeerd), LAN-toegang met basic auth alleen op expliciet verzoek, Windows-autostart voor server en worker (Task Scheduler). Zie doc-001 fase 4 en ADR-008 Decision Contract (secrets-allowlist).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Diff-weergave toont verschillen tussen vorige en nieuwe versie van een artefact na her-run
- [ ] #2 Config-viewer toont uitsluitend allowlisted keys; OPENAI_API_KEY en HF-token komen in geen enkele API-response voor
- [ ] #3 Server en worker starten automatisch mee met Windows indien geconfigureerd
<!-- AC:END -->
