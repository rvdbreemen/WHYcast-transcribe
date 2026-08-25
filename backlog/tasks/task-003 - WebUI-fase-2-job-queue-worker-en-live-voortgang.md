---
id: TASK-003
title: 'WebUI fase 2: job-queue, worker en live voortgang'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24 19:59'
updated_date: '2026-08-25 07:49'
labels:
  - webui
  - backend
dependencies:
  - TASK-002
references:
  - docs/adr/ADR-008-web-ui-as-fastapi-layer-over-extracted-pipeline-library.md
documentation:
  - backlog/docs/doc-001 - WebUI-plan-WHYcast-transcribe.md
ordinal: 3000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Jobs starten, volgen en annuleren vanuit de webui. SQLite-queue, één apart worker-proces dat strikt serieel draait (één GPU), job als subprocess voor crash-isolatie, voortgang als JSONL-eventlog per job en via SSE naar de browser. Job-typen: full, full:episode, retranscribe, speakers-only, postprocess-only, fetch-all. Zie ADR-008 Decision Contract en doc-001 fase 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Volledige aflevering start-tot-eind verwerkt via de UI met live voortgang per pipelinestap
- [x] #2 Browser sluiten en heropenen tijdens een run verliest geen voortgangsweergave (SSE herverbindt op eventlog)
- [x] #3 Nooit meer dan één GPU-job tegelijk, ook bij meerdere queued jobs
- [x] #4 Cancel beëindigt het job-subprocess en laat geen half geschreven artefact achter
- [x] #5 CUDA-OOM in een job laat de webserver draaien en markeert de job als failed met eventlog
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. whycast/io_utils.py: atomic_write_text/bytes (write naar .tmp in dezelfde map, dan os.replace) plus .bak-backup bij overschrijven. Bedraden in outputs.write_all_format, transcription.write_transcript_files, speakers.write_merged_transcript. ADR-008-eis; nodig omdat cancel een subprocess midden in een write kan killen. Golden tests bewaken dat de inhoud identiek blijft.
2. webui/jobs.py: queue-schema in dezelfde SQLite (jobs-tabel: id, type, params, status queued/running/done/failed/cancelled, pid, exit_code, timestamps, gpu-vlag) + claim/finish/cancel-API. Claim via atomische UPDATE zodat dubbele claims onmogelijk zijn.
3. webui/runner.py: kindproces dat een job uitvoert. Installeert JsonlEventSink (logs/jobs/<id>/events.jsonl), roept de whycast-library aan, vertaalt exceptions naar exit-codes. Job-typen: selftest (geen GPU/geen kosten, voor verificatie), fetch_all, fetch_latest, full_episode, force_episode, postprocess, speakers.
4. webui/worker.py: apart proces, strikt serieel (besluit Robert 2026-08-24), pollt queue, spawnt runner als subprocess met CREATE_NEW_PROCESS_GROUP, bewaakt exit, schrijft eindstatus. Lockfile zodat een tweede worker weigert te starten. Cancel = taskkill /T op de procesboom.
5. webui/app.py: SSE-endpoint /api/jobs/{id}/events dat de JSONL tailt en reconnect overleeft (Last-Event-ID), POST /api/jobs, POST /api/jobs/{id}/cancel, GET /jobs dashboard + /jobs/{id} detail, enqueue-knoppen op de episodepagina. Kosten-waarschuwing tonen bij job-typen die OpenAI aanroepen.
6. Tests: queue-claim onder gelijktijdigheid, cancel laat geen half artefact achter (atomic-write-bewijs), SSE levert events na reconnect, worker weigert tweede instantie, selftest-job end-to-end via echte worker+runner. Geen test mag een betaalde API-call doen.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stap 5 (webui/app.py) af: job-API + SSE.

Routes: POST/GET /api/jobs, GET /api/jobs/{id}, POST /api/jobs/{id}/cancel, GET /api/jobs/{id}/events (SSE), GET /api/job-types, GET /jobs, GET /jobs/{id}. Episodepagina krijgt episode_actions/episode_jobs/active_job in de context.

Twee waarborgen zitten elk op een plek. Een onbekend type is 400 (validatie tegen JOB_TYPES); een base_name die niet in de index staat is 400 - hij wordt als opzoeksleutel door db.get_episode gehaald voordat de rij wordt geschreven, nooit als pad. Kosten zijn een eigenschap van het type en staan op elke job- en job-type-response, zodat de UI kan waarschuwen zonder te raden.

SSE tailt events.jsonl vanaf een byte-offset, levert alleen complete regels, gebruikt seq als event-id en hervat op Last-Event-ID (header of query) in plaats van te herhalen. Volgorde bij afronden is bewust drain -> status -> nog een keer drain: de runner schrijft zijn laatste regel en stopt voordat de worker finish() aanroept. Sluit af met een end-frame, anders blijft EventSource eeuwig herverbinden. Bestandslezen en queue-lookups gaan via anyio.to_thread, zodat een rescan (die db._LOCK vasthoudt over een directory-walk) de event loop niet blokkeert.

ensure_job_schema staat in de lifespan voor de rescan, want rescan opent een transactie en de DDL weigert daarbinnen te draaien.

Geverifieerd: 274 tests groen en volgorde-onafhankelijk (tests/test_webui_jobs_api.py, 41 nieuwe). Plus een end-to-end run met echte uvicorn, echt worker-proces en echte httpx-stream op een selftest-job: 13 events live binnengekomen over ~6s (niet gebufferd - eerste event op t+0.9s terwijl de job nog running was), reconnect met Last-Event-ID: 3 leverde 4..13, en een cancel op een draaiende job kwam netjes uit op cancelled. Geen enkele betaalde call.

Gebouwd via workflow (10 agents: atomic-writes + queue parallel, dan runner/worker, dan API+SSE en templates parallel, 3 adversariele reviews, integratie, tests). 17 findings waarvan 7 blockers, alle gefixt. Scherpste vondst: op Windows pre-sized CopyFile2 de bestemming, dus een kill midden in shutil.copy2 liet een .bak achter met de juiste grootte en mtime maar vol NUL-bytes - stil, zonder foutpad. Backup gaat nu ook via temp+os.replace.
Zelf toegevoegd na eigen meting: .tmp-restanten na een hard kill werden nergens opgeruimd (gemeten: gekilde write laat .episode_99_summary.txt.<rnd>.tmp achter, scanner meldt hem als unmatched, stapelt op). Nieuw: whycast.io_utils.sweep_stale_temps + aanroep in worker bij elke settelende job (min_age 0 na tree-kill, 300s routinematig omdat een CLI-run in een andere terminal buiten de seriele garantie valt). Eerste versie sweepte alleen na kill; eigen end-to-end meting liet zien dat een coooperatieve cancel dan overslaat en historische rommel blijft liggen - conditie verruimd. 4 tests toegevoegd.
Zelf geverifieerd: 534 passed / 1 skipped (volledige suite incl. slow). Echte run met server+worker: selftest-job via POST /api/jobs, SSE leverde events met oplopende id's en progress-waarden, eindstatus succeeded/exit 0; lange job geannuleerd via API -> status cancelled, exit_code 130, runner-pid weg; verouderde .tmp in podcasts/ werd bij de volgende settelende job opgeruimd (worker-log bevestigt). Dashboard en episodeacties in de browser bekeken: kostenbadges staan op de betaalde typen.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Job-systeem opgeleverd: SQLite-queue (race-vrije claim), losse worker (strikt serieel, lockfile, tree-kill met commandline-verificatie tegen pid-hergebruik), runner als kindproces met JSONL-eventlog, SSE-stream met Last-Event-ID-resume, dashboard en per-episode acties met kostenmarkering, plus atomische artefactwrites (temp+os.replace, .bak) en een sweep die restanten van gekilde writes opruimt. Kosteloos selftest-jobtype maakt de hele keten testbaar zonder API-uitgaven. Geverifieerd met 534 passed/1 skipped en handmatige end-to-end runs (start, SSE, cancel, sweep) op de draaiende server.
<!-- SECTION:FINAL_SUMMARY:END -->
