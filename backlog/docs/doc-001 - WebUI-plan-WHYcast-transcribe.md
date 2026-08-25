---
id: doc-001
title: WebUI-plan WHYcast-transcribe
type: specification
created_date: '2026-08-24 19:57'
updated_date: '2026-08-24 19:58'
---
# WebUI-plan WHYcast-transcribe

Vastgesteld 2026-08-24. Architectuurbeslissing: ADR-008 (Proposed) — `docs/adr/ADR-008-web-ui-as-fastapi-layer-over-extracted-pipeline-library.md`. Dit document is de faseringsspec; het ADR is leidend bij conflict.

## Doel

Lokale webinterface (single-user, `127.0.0.1`, Windows/CUDA-machine) om afleveringen te bekijken, pipeline-runs te starten en te volgen, en correctie-workflows uit te voeren (sprekers, vocabulary, prompts). Filesystem (`podcasts/`) blijft source of truth; bestaande CLI blijft werken.

## Architectuur (samenvatting; detail in ADR-008)

- Pipeline geëxtraheerd uit `transcribe.py` (3129 regels) naar `whycast/`-package met `EventSink`-voortgangsprotocol; `print()` → events + logging, `exit()` → exceptions. CLI wordt dunne shim, byte-identieke output (golden-master test).
- WebUI: FastAPI + Jinja2/HTMX (+ Alpine.js waar nodig), SSE voor live voortgang.
- Jobs: SQLite-queue, één apart worker-proces, strikt serieel (één GPU), job-subprocess voor crash-isolatie, voortgang als JSONL-eventlog per job.
- SQLite alleen als herbouwbare index + queue; artefacten via pipeline-library met atomische writes en backups.
- Secrets (OPENAI_API_KEY, HF-token) nooit via API; allowlist voor config-weergave.

## Beoogde packagestructuur

```
whycast/
  config.py          # pydantic-settings; env-vars blijven (ADR-007)
  events.py          # ProgressEvent, EventSink-protocol
  episodes.py        # episode-scanner: base_name -> artefacten
  pipeline/
    feed.py transcription.py diarization.py speakers.py
    vocabulary.py postprocess.py outputs.py audio.py
    workflow.py      # orchestrator, emit events
webui/
  app.py api/ worker.py db.py templates/ static/
```

## Fasen

### Fase 0 — Pipeline-extractie (~60% van het werk, schatting)
Golden-master test eerst: één referentie-aflevering door oud CLI-pad en nieuw library-pad, diff op alle artefacten. Mechanische regels: print→event, exit→exception, geen input()-prompts, geen GPU-init bij import. Acceptatie: CLI ongewijzigd, golden-master groen, `import whycast` zonder side-effects.

### Fase 1 — Read-only episode-browser
Scanner + SQLite-index over `podcasts/` (tolerant voor naamgevingschaos, unmatched-files zichtbaar). Overzicht met artefact-statusmatrix per aflevering; detailpagina met tabs per artefact, audio-element voor mp3.
API: `GET /api/episodes`, `GET /api/episodes/{id}`, `GET /api/episodes/{id}/artifacts/{kind}`, `POST /api/rescan`.

### Fase 2 — Jobs: starten, volgen, annuleren
Worker-proces pollt queue, draait één job tegelijk. Job-typen: `full`, `full:episode`, `retranscribe`, `speakers-only`, `postprocess-only`, `fetch-all`. SSE-stream van JSONL-events; job-dashboard met voortgang per stap, log-tail, queue. Cancel = subprocess kill; atomische writes voorkomen halve artefacten. v1 strikt serieel (openstaande vraag in ADR-008).

### Fase 3 — Correctie-workflows
Speaker-editor (mapping-niveau hernoemen, programmatisch toepassen via bestaand ADR-004-pad; AI-suggesties ter review). Vocabulary-editor met per-aflevering toepassen. Prompt-editor met her-run per stap. Elk schrijfpad maakt `.bak`. Vervangt de losse batch-scripts.

### Fase 4 — Afwerking (optioneel)
Diff-weergave oud/nieuw artefact; config-viewer (allowlist); LAN + basic auth alleen op verzoek; Windows-autostart.

## Risico's

| Risico | Mitigatie |
|---|---|
| Refactor breekt pipeline subtiel | Golden-master-diff gate per merge; CLI-shim blijft |
| CUDA-OOM/crash tijdens job | Subprocess-isolatie; job faalt met eventlog |
| Naamgevingschaos breekt scanner | Tolerante matching + unmatched-lijst in UI |
| Secrets lekken via config-UI | Allowlist, nooit denylist |
| Scope-kruip editor | Fase 3 pas ontwerpen als fase 2 draait |

## Open beslissingen (zie ook ADR-008 Open Questions)

1. Strikt serieel vs. LLM-lane parallel aan GPU-jobs (voorstel: serieel in v1).
2. Fase 3: mapping-niveau vs. segment-niveau herlabelen (voorstel: mapping eerst).
