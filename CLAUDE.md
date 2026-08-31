<!-- ADR-KIT CLAUDE START -->
## ADR Kit

Read `.adr-kit/ADR-guide.md` before architectural changes. Architecture decisions live in `docs/adr/`. Use `/adr-kit:context` before implementation, `/adr-kit:adr` for new decisions, and `/adr-kit:judge` before commit.
<!-- ADR-KIT CLAUDE END -->

<!-- BACKLOG.MD GUIDELINES START -->
<!-- backlog.md-instructions-version: 1.50.1 -->
<CRITICAL_INSTRUCTION>

## Backlog.md Workflow

This project uses Backlog.md for task and project management.

**For every user request in this project, run `backlog instructions overview` before answering or taking action.**

Use the overview to decide whether to search, read, create, or update Backlog tasks.

Before task lifecycle actions, read the matching detailed guide:
- `backlog instructions task-creation` before creating or splitting tasks
- `backlog instructions task-execution` before planning, changing status or assignee, adding a plan or implementation notes, or implementing task work
- `backlog instructions task-finalization` before checking acceptance criteria, writing final summaries, or moving tasks to terminal statuses

Use `backlog <command> --help` before running unfamiliar commands. Help shows options, fields, and examples.

Do not edit Backlog task, draft, document, decision, or milestone markdown files directly. Use the `backlog` CLI so metadata, relationships, and history stay consistent.

</CRITICAL_INSTRUCTION>
<!-- BACKLOG.MD GUIDELINES END -->

## Generated knowledge graph

`graphify-out/` is ignored except for `GRAPH_REPORT.md`, which is tracked because
70 KB of readable text is worth reviewing in a diff where `graph.html` and
`graph.json` are 4 MB of blob.

That has a cost: `graphify update .` rewrites the report, so it turns up in the
working tree after almost any code change and drags a several-hundred-line diff
into commits it has nothing to do with.

So: **commit `GRAPH_REPORT.md` on its own, never alongside unrelated work.**
Regenerate it deliberately, in its own commit, with a message saying the graph
was rebuilt. If it shows up while you are committing something else, leave it
unstaged.
