---
id: TASK-006
title: Optionele-dependency shims falen met None in plaats van te degraderen
status: Done
assignee:
  - '@robert'
created_date: '2026-08-30 19:39'
updated_date: '2026-08-31 08:29'
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
- [x] #1 feed.py controleert feedparser_available voor gebruik en geeft een nette fout of skip in plaats van AttributeError
- [x] #2 llm.py controleert openai_available voor het aanmaken van de OpenAI-client, of de ongebruikte import verdwijnt
- [x] #3 De module-docstring van whycast/_deps.py beschrijft exact welke shims wel en niet flag-guarded zijn
- [ ] #4 Een test dekt het pad waarin de optionele dependency ontbreekt
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
feed.py roept nu _require_feedparser() aan voor beide feedparser.parse-aanroepen (regel 213 en 285 in de oude nummering) en werpt ConfigurationError met de pip-opdracht erin, in plaats van AttributeError op None.

llm.py controleert openai_available voordat OpenAI(api_key=...) wordt aangeroepen en werpt ConfigurationError; daarmee is de import die er al stond ook echt in gebruik.

Bewust GEEN nieuwe DependencyError geintroduceerd: die is in 75bf8c4 juist verwijderd, en ConfigurationError dekt de lading (configuratie ontbreekt of is ongeldig) zonder een geschrapt symbool terug te halen.

De docstring van whycast/_deps.py beschrijft nu per shim wat er echt gebeurt: openai en feedparser worden None en hun vlag MOET gecontroleerd worden, tqdm heeft een echte vervangingsklasse en hoeft dat niet.

Na adversariele review op drie punten gerepareerd, want de eerste versie was slechter dan de bug.

1. De guard stond IN de try, waarvan de handler call_id formatteert, een lokale die pas later gebonden wordt. Reproduceerbaar: UnboundLocalError in plaats van de zorgvuldig geformuleerde melding. Guard en ensure_api_key staan nu buiten de try, call_id wordt vooraf gebonden. Dat laatste repareert ook een bestaande bug: elke vroege fout liet de handler crashen.
2. De ConfigurationError werd een frame hoger alsnog opgeslokt door brede except-Exception-handlers in summarize_large_transcript, analyze_speakers en full_workflow. Omdat de sprekersstap voor de cleanup draait, zag een gebruiker zonder openai-pakket eerst 'Speaker analysis failed'. Die handlers laten een ConfigurationError nu door.
3. ensure_api_key wierp een kale ValueError, die geen enkele arm in transcribe.py ving; een ontbrekende sleutel eindigde dus in een traceback. Werpt nu ConfigurationError, en webui/runner.py vangt beide spellingen.

tests/test_optional_dependency_guards.py, 9 tests plus 1 skip, dekt de guard, de melding, dat hij niet opgeslokt wordt, dat de handler niet op een ongebonden naam crasht, en dat ConfigurationError geen subklasse van PipelineError is.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
feed.py en llm.py controleren hun optionele shims voor gebruik en werpen ConfigurationError met de pip-opdracht erin, in plaats van AttributeError of TypeError op een None. De docstring van _deps.py beschrijft per shim wat er echt gebeurt. Drie fouten uit de review zijn daarna gerepareerd: de guard stond in een try die hem opslokte en wiens handler op een ongebonden call_id crashte, drie brede handlers slokten de fout een frame hoger alsnog op, en ensure_api_key wierp een ValueError die de CLI niet ving. Geverifieerd met 9 tests in tests/test_optional_dependency_guards.py.
<!-- SECTION:FINAL_SUMMARY:END -->
