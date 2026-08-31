---
id: TASK-009
title: CUDA-runtimetest faalt op dubbele OpenMP-runtime
status: Done
assignee: []
created_date: '2026-08-30 19:40'
updated_date: '2026-08-31 00:48'
labels: []
dependencies: []
documentation:
  - docs/adr/ADR-001-use-faster-whisper-on-cuda-for-transcription.md
modified_files:
  - tests/test_cuda_runtime.py
priority: medium
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/test_cuda_runtime.py::test_real_transcription_runs_on_gpu faalt op de ontwikkelmachine. De rest van de suite is groen: 827 passed, 1 skipped met deze test gedeselecteerd.

Waargenomen fout, letterlijk uit het subproces:

OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

De assertie op tests/test_cuda_runtime.py:112 ziet daardoor een lege stdout en meldt 'GPU transcription failed - CUDA runtime is not usable', wat de oorzaak verhult: het subproces sterft voordat er ook maar iets is getranscribeerd. De echte oorzaak is dat meerdere OpenMP-runtimes in hetzelfde proces zijn gelinkt, doorgaans doordat torch en een andere wheel elk hun eigen libiomp5md.dll meebrengen.

Dit is een omgevingsprobleem, niet veroorzaakt door recente codewijzigingen. Het maakt de suite wel rood op deze machine, dus het verdient een besluit: de omgeving opschonen, of de test laten skippen met een duidelijke reden zodat de fout niet als CUDA-defect wordt gelezen. KMP_DUPLICATE_LIB_OK=TRUE is nadrukkelijk geen oplossing: Intel documenteert dat als onveilig en het kan stil verkeerde resultaten opleveren.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 De oorzaak is vastgesteld: welke twee pakketten leveren libiomp5md.dll in deze omgeving
- [ ] #2 De test slaagt op een machine met CUDA, of skipt met een boodschap die de OpenMP-oorzaak noemt in plaats van CUDA de schuld te geven
- [ ] #3 De foutboodschap van de assertie toont stderr van het subproces, zodat een volgende lezer de echte oorzaak direct ziet
- [ ] #4 KMP_DUPLICATE_LIB_OK wordt niet als oplossing gebruikt
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Oorzaak gevonden op 2026-08-31, en het is breder dan de test. C:/Python312/Lib/site-packages/ctranslate2/ bevat drie DLLs: ctranslate2.dll, cudnn64_9.dll en libiomp5md.dll.

1. OpenMP-botsing (het symptoom van deze taak): ctranslate2 levert een eigen libiomp5md.dll, en torch levert er ook een. Beide in een proces geeft 'OMP: Error #15'. Dit treedt op zodra torch en faster-whisper in hetzelfde proces geladen worden, wat de pipeline doet.

2. Ernstiger, en dit blokkeert GPU-transcriptie volledig: ctranslate2 4.5.0 levert alleen cudnn64_9.dll, de loader van cuDNN 9. De bijbehorende cudnn_ops64_9.dll ontbreekt en is nergens op de C-schijf te vinden. Een GPU-transcriptie faalt daarom met 'Could not locate cudnn_ops64_9.dll' gevolgd door 'Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor'.

Waargenomen tijdens een echte hertranscriptie van podcasts/episode_0.mp3 op 2026-08-31: diarization slaagde (321 turns, 3 sprekers, 69s op de RTX 3080), transcriptie stierf direct op de ontbrekende DLL.

Onderliggende mismatch: torch is 2.3.1+cu118 met cuDNN 8.7 gebundeld in torch/lib, terwijl ctranslate2 4.5.0 cuDNN 9 verwacht. De twee stacks zijn niet op dezelfde CUDA-generatie gebouwd.

Twee bekende uitwegen, allebei nog niet uitgevoerd omdat ze de omgeving wijzigen: de ontbrekende cuDNN 9 bibliotheken installeren (bijvoorbeeld via nvidia-cudnn-cu12) met het risico dat torch daardoor verschuift, of diarization en transcriptie in gescheiden processen draaien zodat de OpenMP-botsing verdwijnt, wat het cuDNN-probleem echter niet oplost.

Aanvulling 2026-08-31, de scope is groter dan de titel suggereert. De OpenMP-botsing treedt ook op in een proces waarin ALLEEN faster_whisper wordt geimporteerd, zonder torch en zonder een eigen numpy-import. faster_whisper trekt zelf numpy en onnxruntime binnen, die elk een OpenMP-runtime meebrengen naast de libiomp5md.dll in site-packages/ctranslate2.

Getoetst met twee losse processen op 2026-08-31:
  import numpy + faster_whisper           -> OMP: Error #15, proces stopt
  alleen faster_whisper                   -> OMP: Error #15, proces stopt

Gevolg: WhisperModel kan op deze machine niet geladen worden, op GPU noch op CPU. De transcriptiestap van de pipeline is hier niet werkend. Dat is iets anders dan een falende test; de bestaande transcripten in podcasts/ zijn kennelijk gemaakt voor een pakketwijziging dit kapotmaakte.

KMP_DUPLICATE_LIB_OK=TRUE blijft ongeschikt als oplossing: Intel documenteert het als onveilig en met kans op stil verkeerde uitkomsten. Voor een transcriptie die daarna in de kennisbank belandt is stil verkeerd het slechtste falen.

RICHTINGSWIJZIGING, beslist door Robert op 2026-08-31 na de meting op aflevering 0.

De gemeten doorvoer van de referentie-implementatie is 1,16x realtime (1538s transcriptie voor 1327s audio) plus 453s diarization, op een RTX 3080 Laptop. Een aflevering van twee uur wordt daarmee ruim tweeeneenhalf uur. Dat is als te traag beoordeeld.

Drie besluiten:
1. Doorvoer weegt zwaarder dan het opruimen van de tweede native stack. faster-whisper wordt gerepareerd in plaats van vervangen. ADR-012 in zijn huidige vorm, die vervanging voorstelt, gaat daarmee niet door.
2. whycast houdt WEL een schakelaar tussen beide runtimes, zodat een machine met een intacte CUDA-12 stack de snelle route krijgt en een machine zonder hem alsnog kan transcriberen.
3. De OpenMP-botsing wordt eerst onderzocht, voordat er iets geaccepteerd of gemigreerd wordt. Dat is de dragende onbekende: is de botsing zelf te repareren.

ADR-012 blijft Proposed en ongewijzigd tot dat onderzoek klaar is; herschrijven op basis van een vermoeden zou het record laten liegen. Afhankelijk van de uitkomst wordt het herschreven naar 'repareer plus schakelaar' of ingetrokken ten gunste van een nieuw ADR.

CORRECTIE 2026-08-31, de diagnose in deze taak was fout en de taak is ongeldig.

Alles hierboven is gemeten met de GLOBALE interpreter C:/Python312/python.exe. Dit project heeft een eigen venv, venv/Scripts/python.exe, en daar bestaat geen van de beschreven problemen.

Gemeten in de venv:
  torch 2.6.0+cu118 (niet 2.3.1), bundelt cuDNN 9 in torch/lib/cudnn64_9.dll
  geen mkl en geen intel-openmp, dus geen tweede OpenMP-runtime en geen OMP Error #15
  nvidia-cublas-cu12 12.9.2.10 aanwezig, met venv/Lib/site-packages/nvidia/cublas/bin/cublas64_12.dll
  WhisperModel('tiny', device='cpu') bouwt zonder fout
  na ensure_cuda_libs(): WhisperModel('large-v3', device='cuda', compute_type='float16') laadt en
  transcribeert echt, met woordtijden, taal en herkend

Volledige testsuite in de venv, zonder deselect: 906 passed, 1 skipped, 0 failed. Ook
test_cuda_runtime.py::test_real_transcription_runs_on_gpu slaagt.

De OpenMP-botsing in de globale interpreter komt doordat torch 2.3.1 daar mkl<=2021.4.0 eist, wat
intel-openmp==2021.* meebrengt; die registreert als eerste vanuit C:/Python312/Library/bin, waarna
ctranslate2's eigen nieuwere kopie afbreekt. Dat is een eigenschap van die interpreter, niet van dit
project.

Conclusie: er is geen defect. faster-whisper werkt. Deze taak wordt gesloten als niet-reproduceerbaar
buiten een verkeerd gekozen interpreter.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Niet reproduceerbaar in de projectomgeving. De OpenMP-fout en de ontbrekende cuDNN 9 traden alleen op onder de globale interpreter C:/Python312, die via torch 2.3.1 een mkl- en intel-openmp-afhankelijkheid meebrengt. In venv/Scripts/python.exe (torch 2.6.0+cu118, cuDNN 9 gebundeld, nvidia-cublas-cu12 aanwezig) laadt en transcribeert faster-whisper large-v3 gewoon op de GPU, en slaagt de volledige suite met 906 passed, 1 skipped, 0 failed inclusief de CUDA-runtimetest.
<!-- SECTION:FINAL_SUMMARY:END -->
