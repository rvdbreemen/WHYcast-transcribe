---
id: TASK-002
title: 'WebUI fase 1: read-only episode-browser'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24 19:58'
updated_date: '2026-08-24 22:49'
labels:
  - webui
  - frontend
dependencies:
  - TASK-001
references:
  - docs/adr/ADR-008-web-ui-as-fastapi-layer-over-extracted-pipeline-library.md
documentation:
  - backlog/docs/doc-001 - WebUI-plan-WHYcast-transcribe.md
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
FastAPI-app (webui/) met Jinja2/HTMX die alle afleveringen uit podcasts/ toont: overzicht met artefact-statusmatrix per aflevering en detailpagina met tabs per artefact (transcript, ts, cleaned, summary, blog, history, assignment) plus audio-element. SQLite als herbouwbare index; scanner tolerant voor de bestaande naamgevingschaos, met zichtbare unmatched-files-lijst. Geen schrijfpaden. Zie ADR-008 en het WebUI-plan (doc-001).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Alle afleveringen uit podcasts/ zichtbaar met correcte artefact-status per type
- [x] #2 Detailpagina rendert HTML-artefacten en toont txt-artefacten; mp3 afspeelbaar
- [x] #3 Niet-gematchte bestanden zichtbaar in de UI in plaats van stil genegeerd
- [x] #4 SQLite-index verwijderbaar en herbouwbaar via rescan zonder dataverlies
- [x] #5 Server bindt standaard op 127.0.0.1
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. whycast/episodes.py: tolerante scanner. Ankeren op audiobestanden (64 mp3/m4a) als canonieke basenames; artefacten koppelen via longest-base-match met separator-regel (base gevolgd door '_' of '.') zodat episode_1 niet episode_10_blog opslokt. Wees-artefacten zonder audio krijgen eigen episode via suffix-stripping. Alles wat nergens matcht komt in unmatched-lijst (thumbnails, losse jpg/mp4).
2. Artefactsoorten: transcript (<base>.txt en <base>_transcript.txt), ts, cleaned, summary, blog, blog_alt1, history, speaker_assignment, analysis, merged - elk met varianten .txt/.html/.wiki/.md.
3. webui/db.py: SQLite-index (episodes, artifacts, unmatched) - herbouwbaar, wegwerpbaar; rescan-functie.
4. webui/app.py: FastAPI + Jinja2/HTMX. Overzicht met statusmatrix, detailpagina met tabs per artefact, audio-element, unmatched-sectie. Bind 127.0.0.1.
5. API: GET /api/episodes, /api/episodes/{id}, /api/episodes/{id}/artifacts/{kind}, POST /api/rescan. Path-traversal-guard op artefact-serving.
6. Tests: scanner-unit tests op echte podcasts/-structuur (incl. episode_1 vs episode_10 collision), API-tests via httpx/TestClient, rescan-idempotentie.
7. requirements.txt bijwerken (fastapi, uvicorn, jinja2).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Gebouwd via workflow (9 agents: scanner-substraat, dan db/api/templates parallel, 3 adversariele reviews, integratie, tests). Scanner meet op echte podcasts/: 62 audio-bases, 719 gekoppelde artefacten, 31 unmatched; accounting-invariant 31+719+62=812 sluit exact. Twee blockers gevonden en gefixt: (1) stored XSS - artefact-tabs deden fetch+innerHTML, nu navigatie naar sandboxed iframe zodat de response-CSP werkelijk geldt, met echte browser-proof dat script-executie geblokkeerd wordt; (2) phantom-episode-bug in scanner bij onbekende suffix-rest. Zelf geverifieerd: 218 passed / 1 skipped (symlink-test vereist privilege, zelfde regel gedekt door 9 _safe_file-tests), server geboot op 127.0.0.1:8420 en met browser doorlopen - overzicht, detail episode_13 (16 artefacten), artefact laadt echt in sandboxed frame. .gitignore uitgebreid met webui/*.db (cache hoort niet in git, ADR-008) en .playwright-mcp/.
OPEN PRODUCTVRAAG voor Robert: 62 bases zijn ~48 logische afleveringen. Dubbele audio onder andere schrijfwijze (episode-1 vs episode_1, whycast-episode-13 vs episode_13, Episode 28 vs Episode_28) staat nu als aparte regel zonder artefacten. Dedup-by-number hoort in de indexlaag met menselijke review, niet in de scanner. Niet gebouwd - wacht op beslissing.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Read-only episode-browser opgeleverd: whycast/episodes.py (tolerante scanner, case-insensitive, longest-base-match met separator-regel tegen de episode_1/episode_10-val), webui/db.py (SQLite-cache, idempotente transactionele rescan, herbouwbaar), webui/app.py + templates + gevendorde htmx (overzicht met statusmatrix, detail met audio-speler en artefact-tabs, unmatched-lijst). Geverifieerd met 218 passed/1 skipped en handmatige browsersessie op de draaiende server; twee blockers (XSS, phantom-episode) gevonden door adversariele review en gefixt.
<!-- SECTION:FINAL_SUMMARY:END -->
