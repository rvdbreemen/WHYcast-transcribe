---
id: TASK-012
title: Sprekersparser laat SPEAKER_UNKNOWN vallen en accepteert vrije tekst als naam
status: Done
assignee:
  - '@robert'
created_date: '2026-08-30 20:09'
updated_date: '2026-08-30 21:09'
labels: []
dependencies: []
documentation:
  - docs/adr/ADR-004-hybrid-ai-analysis-plus-programmatic-speaker-assignment.md
modified_files:
  - whycast/pipeline/speakers.py
priority: high
ordinal: 12000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
parse_speaker_mapping_from_analysis() (whycast/pipeline/speakers.py:239) leest de mapping uit de prozauitvoer van het analysemodel. Die parser is afgestemd op hoe o4-mini formatteert en breekt op andere, op zichzelf correcte, uitvoer.

Waargenomen op 2026-08-30 tijdens de A/B-run met gpt-5.6-sol op episode_1_merged.txt:

1. Het model leverde een compleet, netjes afgebakend blok met alle zeven labels, inclusief SPEAKER_UNKNOWN -> Unknown. De parser gaf zes labels terug en liet SPEAKER_UNKNOWN vallen. In dit transcript draagt SPEAKER_UNKNOWN 45 van de 155 segmenten, dus dat is geen randgeval.
2. In een tweede run op hetzelfde bestand kende de parser aan SPEAKER_01 een hele zin toe als naam: 'clause I am Ad should be manually changed to **Ad**, and SPEAKER_UNKNOWN should ideally be replaced locally...'. Dat is losse toelichtende tekst uit het antwoord, niet een naam.

Gevolg: de kwaliteit van de sprekersstap hangt niet alleen af van het model maar ook van of de prozavorm toevallig bij de regex past. Bij hetzelfde model en dezelfde invoer levert dat wisselende uitkomsten. Een naam die een halve alinea is, komt bovendien via apply_speaker_mapping_programmatically in het transcript terecht.

Dit blokkeert de conclusie van de modelmigratie: zonder betrouwbare parser is niet te zeggen of een model beter is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 De parser leest alleen uit het expliciete eindmapping-blok, niet uit omringend proza
- [x] #2 SPEAKER_UNKNOWN blijft behouden wanneer het model het teruggeeft
- [x] #3 Een waarde die geen plausibele naam is (te lang, bevat opmaak of leestekens) wordt geweigerd en gelogd in plaats van doorgegeven
- [x] #4 Tests dekken de uitvoervorm van zowel o4-mini als gpt-5.6-sol met echte vastgelegde antwoorden
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Oorzaak vastgesteld voor de fix, niet aangenomen. De parser had drie takken. Tak 1 zocht letterlijk 'FINAL MAPPING FOR TRANSCRIPT:' met dubbelepunt en brak bovendien direct af op de '====' regel eronder, dus die vuurde nooit. Tak 2 (per-spreker secties met '- Final Label:') is wat o4-mini in de praktijk parseerde. Tak 3 was een regex over het HELE document met SPEAKER_\d+, wat SPEAKER_UNKNOWN uitsluit en willekeurig proza binnenhaalt; dat is wat gpt-5.6-sol kreeg, omdat die een markdown-kop zonder dubbelepunt schrijft.

Nieuwe opzet: _mapping_from_final_block leest alleen onder de LAATSTE mapping-kop (hoofdletterongevoelig, dubbelepunt optioneel), _mapping_from_speaker_sections blijft als tweede kans, _mapping_from_anywhere blijft als laatste redmiddel maar valideert nu elk label. _plausible_speaker_label weigert waarden boven 40 tekens of met zinsinterpunctie en logt dat.

Twee regressies onderweg gevangen door tests/test_golden_extraction.py, allebei gerepareerd: een dubbelepunt als scheider las 'CONFIDENCE SUMMARY' regels als beslissingen en overschreef de namen met 'high'/'medium'; en het laatste redmiddel gebruikte een aan het regelbegin verankerd patroon waardoor losse zinnen niet meer matchten. De golden snapshot is ONgewijzigd gebleven: alle vier de vastgelegde gevallen geven exact dezelfde uitkomst als voorheen.

Verificatie op echte modeluitvoer (tests/data/speaker_analysis/): gpt-5.6-sol 6 -> 7 labels met SPEAKER_UNKNOWN, o4-mini onveranderd 7 labels. Suite: 881 passed, 1 skipped.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
De parser was afgestemd op o4-mini's opmaak: de sectiekop moest een dubbelepunt hebben en het laatste redmiddel scande het hele document met een patroon dat SPEAKER_UNKNOWN uitsloot. Daardoor verloor gpt-5.6-sol een label dat 45 van de 155 segmenten draagt, en kon losse toelichting als naam in het transcript belanden. Vervangen door drie duidelijk gescheiden strategieen met validatie van elk label. Geverifieerd met 13 nieuwe tests op vastgelegde echte uitvoer van beide modellen, met de onveranderde golden snapshot als bewijs dat bestaand gedrag intact bleef, en met een handmatige parse waarin sol van 6 naar 7 labels gaat.
<!-- SECTION:FINAL_SUMMARY:END -->
