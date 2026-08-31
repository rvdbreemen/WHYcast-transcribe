---
id: TASK-011
title: Verwijdering van DependencyError vastleggen als publieke wijziging
status: To Do
assignee: []
created_date: '2026-08-30 19:40'
labels: []
dependencies: []
documentation:
  - docs/adr/ADR-008-web-ui-as-fastapi-layer-over-extracted-pipeline-library.md
modified_files:
  - whycast/errors.py
priority: low
ordinal: 11000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Op branch docs/graphify-knowledge-graph is de exceptieklasse DependencyError uit whycast/errors.py verwijderd, samen met io_utils._make_backup en de Jinja-macro formats_of uit webui/templates/_macros.html.

Geverifieerd: geen enkele verwijzing meer in de repo. Doorzocht op py, html, md en toml; nul treffers voor alle drie de namen. De testsuite is groen op 827 passed.

Resterend punt: whycast/errors.py is de foutmodule van een als bibliotheek bedoeld pakket (ADR-008 maakt de webui een laag boven de geextraheerde pipeline-bibliotheek). Een exceptieklasse daaruit weghalen is een breaking change voor iedere afnemer buiten deze repo. Vaststellen of die er zijn, en zo niet, de verwijdering vastleggen zodat een latere lezer niet gaat zoeken naar een klasse die bewust is geschrapt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Vastgesteld of er afnemers buiten deze repo zijn die whycast.errors.DependencyError importeren
- [ ] #2 De verwijdering staat vermeld in de changelog of releasenotes als breaking change, of er is onderbouwd waarom dat niet nodig is
<!-- AC:END -->
