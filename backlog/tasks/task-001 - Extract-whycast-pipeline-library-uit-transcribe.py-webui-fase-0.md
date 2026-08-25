---
id: TASK-001
title: Extract whycast pipeline library uit transcribe.py (webui fase 0)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24 19:58'
updated_date: '2026-08-24 20:49'
labels:
  - webui
  - refactor
dependencies: []
references:
  - docs/adr/ADR-008-web-ui-as-fastapi-layer-over-extracted-pipeline-library.md
documentation:
  - backlog/docs/doc-001 - WebUI-plan-WHYcast-transcribe.md
ordinal: 1000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extraheer de pipeline uit de transcribe.py-monoliet (3129 regels) naar een importeerbaar whycast/-package met EventSink-voortgangsprotocol, zodat webui, CLI en batch-scripts dezelfde library aanroepen. Fundament voor de hele webui (ADR-008); zonder dit bestaat er geen betrouwbaar service-pad met voortgang en typed errors. Mechanische regels: print() naar event+logging, exit() naar exception, geen input()-prompts, geen GPU-init bij import. transcribe.py blijft als dunne CLI-shim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Golden-master test in tests/: één referentie-aflevering via oud CLI-pad en nieuw library-pad levert identieke output-artefacten
- [x] #2 python -c "import whycast" draait zonder GPU-initialisatie, netwerk of filesystem-writes
- [x] #3 Bestaande CLI-aanroepen (transcribe.py met input/--force/--fetch-all) werken ongewijzigd
- [x] #4 Geen exit()/sys.exit() en geen print() in whycast/ library-code (ADR-008 Enforcement)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Scaffold whycast/-package: events.py (ProgressEvent, EventSink via contextvar, ConsoleSink/NullSink), errors.py (PipelineError e.a.), config-verplaatsing met compat-shim in root config.py (pydantic-settings uitgesteld naar fase 1 - kleinere mechanische stap).
2. Inventory: per functie regelrange, globals, module-level side effects, print/exit/input-locaties in transcribe.py (3129 regels, 50 functies).
3. Parallelle extractie per doelmodule (feed, audio, gpu, transcription, diarization, speakers, vocabulary, llm, postprocess, outputs, workflow) - functies letterlijk kopieren, alleen mechanische regels: print->emit(), exit->raise, input-prompts alleen via expliciete parameter vanuit CLI-shim.
4. transcribe.py herschrijven als dunne CLI-shim met ConsoleSink (byte-identieke console-output en artefacten).
5. Verificatie: py_compile alle modules; import whycast zonder side effects; diff-review per module (alleen mechanische transformaties toegestaan); golden-tests niveau 1 (deterministische functies, fixtures) en niveau 2 (workflow met gemockte OpenAI); niveau 3 GPU-smoke op tests/data/dummy_audio.mp3 als aparte verificatiestap.
6. ADR-008 Enforcement checkt exit()/print() in whycast/ bij commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR-008 Accepted 2026-08-24. Open questions beantwoord: strikt serieel v1; mapping-niveau editing fase 3. Afwijking van doc-001: pydantic-settings uitgesteld naar fase 1, nu pure config-verplaatsing (kleiner refactor-risico).

Extractie uitgevoerd via 30-agent workflow (13 extract + 13 adversarial review + integratie + tests + shim). 2 review-blockers gevonden (logging_setup, diarization) en gefixt. Onafhankelijk geverifieerd: 13/13 golden snapshot-tests groen (tests/test_golden_extraction.py, snapshots in tests/golden/ vastgepind op oude monoliet-output; snapshot-mutatie-check gedaan), transcribe.py nu 227-regel CLI-shim met re-exports voor batch-scripts, import whycast zonder side effects, exit()/print()-grep-gates schoon. GPU-smoke (30s clip ep29, alleen transcriptie+diarisatie, geen OpenAI) loopt.

GPU-parity-bewijs (30s clip uit ep29, geen OpenAI-calls): oud monoliet-pad (uit git HEAD) en nieuw whycast-pad geven identiek gedrag - verify_gpu_setup OK, prepare_audio OK, diarisatie OK met exact 5 segmenten in beide, transcriptie faalt in BEIDE paden op dezelfde pre-existing omgevingsfout: RuntimeError cublas64_12.dll not found (ctranslate2 4.5.0 is CUDA-12-build, torch is 2.6.0+cu118, geen CUDA-12-runtime op dit systeem - stond los van de refactor). Extra fix tijdens smoke: ConsoleSink vangt UnicodeEncodeError op cp1252-console (emoji) met errors=replace i.p.v. crashen; tests daarna opnieuw 13/13 groen. Let op: tests/data/dummy_audio.mp3 is 0 bytes en als fixture onbruikbaar.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
transcribe.py-monoliet (3129 regels) geëxtraheerd naar whycast/-package (13 modules) met EventSink-voortgangsprotocol; transcribe.py is nu 227-regel CLI-shim met re-exports. Geverifieerd: 13/13 golden snapshot-tests (oud vs nieuw, snapshots vastgepind in tests/golden/), import whycast zonder side effects, exit()/print()-gates schoon (ADR-008 Enforcement), CLI --version en argparse identiek, GPU-parity op echte 30s audioclip (diarisatie beide 5 segmenten; transcriptie faalt in beide paden op dezelfde pre-existing cublas64_12.dll-omgevingsfout). Uitgevoerd via 30-agent workflow met adversarial review (2 blockers gevonden en gefixt).
<!-- SECTION:FINAL_SUMMARY:END -->
