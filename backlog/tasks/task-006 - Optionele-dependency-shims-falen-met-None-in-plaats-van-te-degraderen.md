---
id: TASK-006
title: Optionele-dependency shims falen met None in plaats van te degraderen
status: To Do
assignee: []
created_date: '2026-08-30 19:39'
labels: []
dependencies: []
documentation:
  - docs/adr/ADR-008-web-ui-as-fastapi-layer-over-extracted-pipeline-library.md
modified_files:
  - whycast/_deps.py
  - whycast/pipeline/feed.py
  - whycast/pipeline/llm.py
priority: medium
ordinal: 6000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
whycast/_deps.py levert shims voor optionele dependencies. De module-docstring stelt dat afnemers de bijbehorende *_available-vlag controleren voor gebruik, zodat een ontbrekende dependency alleen de betreffende stap degradeert (ADR-008). Twee van de drie shims doen dat niet.

Feiten uit de code:
- whycast/pipeline/feed.py:23 importeert de feedparser-shim en roept feedparser.parse() aan op :213 en :285 zonder feedparser_available te controleren. Zonder feedparser is de shim None, dus dit geeft AttributeError in plaats van een nette skip.
- whycast/pipeline/llm.py:17 importeert openai_available maar controleert het nergens; :242 doet OpenAI(api_key=api_key), wat zonder openai TypeError geeft (None is niet aanroepbaar).
- openai_available wordt alleen gecontroleerd in whycast/pipeline/speakers.py:164 en :1158.
- feedparser_available en tqdm_available worden nergens geimporteerd.

Gevolg: de docstring belooft een contract dat de code niet nakomt. Ofwel de guards toevoegen, ofwel de docstring corrigeren naar wat er echt staat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 feed.py controleert feedparser_available voor gebruik en geeft een nette fout of skip in plaats van AttributeError
- [ ] #2 llm.py controleert openai_available voor het aanmaken van de OpenAI-client, of de ongebruikte import verdwijnt
- [ ] #3 De module-docstring van whycast/_deps.py beschrijft exact welke shims wel en niet flag-guarded zijn
- [ ] #4 Een test dekt het pad waarin de optionele dependency ontbreekt
<!-- AC:END -->
