---
id: TASK-005
title: 'WebUI fase 4: afwerking (diff-weergave, config-viewer, autostart)'
status: Done
assignee: []
created_date: '2026-08-24 19:59'
updated_date: '2026-08-25 22:09'
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
- [x] #1 Diff-weergave toont verschillen tussen vorige en nieuwe versie van een artefact na her-run
- [x] #2 Config-viewer toont uitsluitend allowlisted keys; OPENAI_API_KEY en HF-token komen in geen enkele API-response voor
- [x] #3 Server en worker starten automatisch mee met Windows indien geconfigureerd
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Zelf gebouwd (geen workflow). Diff werkt via een before-snapshot in logs/jobs/<id>/before/ in plaats van .bak - ADR-009 haalde die weg uit podcasts/ en dit brengt ze niet terug. Snapshot is best-effort: een job mag nooit falen omdat de kopie voor latere inspectie misging. Config-viewer is read-only met allowlist; twee tests bewijzen dat er geen secret in komt, ook niet wanneer er een nieuwe in whycast.config wordt geplant. 16 nieuwe tests, volledige suite 806 passed / 1 skipped. Config-pagina op de draaiende server opgevraagd: 200, host 127.0.0.1, 23 settings en 7 padverwijzingen, geen sleutels.
AC #3 NIET afgevinkt: scripts/install-autostart.ps1 is geschreven en syntactisch gecontroleerd, maar ik heb de scheduled tasks niet geregistreerd - dat is een wijziging aan de machine-instellingen en die hoort Robert zelf te draaien. Pas na 'scripts\install-autostart.ps1' plus een herstart is dit criterium echt aantoonbaar.

2026-08-26: Robert heeft install-autostart.ps1 zelf gedraaid; beide taken geregistreerd en gestart. Daarbij kwam een bug in mijn eigen launcher boven: pythonw.exe onder Task Scheduler heeft geen console, dus sys.stdout is None en uvicorn's StreamHandler liet de server sterven voordat hij de poort bond (LastTaskResult 1, geen spoor). Vanuit een shell niet reproduceerbaar omdat die wel een pipe meegeeft. Gefixt met een guard die ontbrekende streams naar logs/webui-server.log wijst, plus twee regressietests. Daarna via de scheduler geverifieerd: beide taken Running, GET / -> 200, en een selftest-job via HTTP opgepakt door de scheduler-worker met exit 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Afwerking opgeleverd: diff-weergave per job tegen een before-snapshot in de job-logmap (niet via .bak, ADR-009), read-only config-viewer met allowlist die geen secrets kan tonen, en een autostart-script voor server en worker als twee losse logon-taken. Geverifieerd met 806 passed/1 skipped en een sessie op de draaiende server. Autostart is bewust niet geinstalleerd: dat wijzigt machine-instellingen en is aan de eigenaar.
<!-- SECTION:FINAL_SUMMARY:END -->
