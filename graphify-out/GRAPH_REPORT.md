# Graph Report - .  (2026-08-26)

## Corpus Check
- 113 files · ~167,219 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2158 nodes · 3804 edges · 126 communities (101 shown, 25 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 465 edges (avg confidence: 0.76)
- Token cost: at least 314,711 subagent tokens (chunks 1 and 2). Chunk 3's count was lost when this session's context was compacted, and the harness reports one combined figure per subagent rather than an input/output split, so neither half can be stated. Treat 314,711 as a floor, not a total.

## Provenance: what was repaired before this graph was built

This run needed four corrections. Each is listed so a later reader can tell the
graph apart from what the tooling produced on its own.

1. **159 internal dependency edges, reconnected.** graphify's AST pass names a
   module node `whycast_events_py` but emits its import edges to `whycast_events`,
   and collapses `from webui import runner` to a bare `webui`. Neither target
   exists as a node, and `build()` drops an edge with a missing endpoint.
   `validate_extraction()` does report them, but its output is dominated by
   legitimate stdlib imports (`os`, `sys`, `logging`), so the internal ones hide in
   the noise. 115 edges were reconnected by the `_py` suffix, 35 by reading the
   import line back off disk to recover the submodule, and 6 fell back to the
   package's `__init__.py` because the line imports a constant rather than a
   module. This is the dependency skeleton of the codebase; without the repair the
   graph did not contain it.

2. **Eight duplicated ADRs, collapsed.** Semantic extraction ran as three parallel
   subagents, and a subagent can only draw edges inside its own chunk. Chunk 2 held
   `ADR-INDEX.md` and so minted its own nodes for ADR-001..008 to give its ~40
   citation edges a target; chunk 1 held the ADR files themselves and extracted the
   same eight decisions. Both were right locally, and together they were eight
   decisions represented twice with half the edges each. The per-file node won -
   it is where the decision actually lives - and the index's edges were rewritten
   onto it.

3. **Three cached edges pointing at ADR concepts that were never extracted.**
   Cache entries for `prompts/*.txt` written by an earlier run guessed what the
   ADR's node would be called, and chunk 1 later slugged it differently. Repaired
   in the cache itself rather than only in this build, so a later
   `graphify update` cannot quietly reintroduce them.

4. **105 nodes from `webui/static/htmx.min.js`, dropped.** Vendored, minified,
   third-party. Its mangled names (`He`, `e`, `F`) carried no meaning, and two of
   them outranked real functions in the god-node list - the graph reporting on its
   own noise.

295 edges remain unconnected on purpose: they are imports of stdlib and
third-party packages (`os`, `logging`, `pytest`, `fastapi`), which have no node
because those libraries are not part of this corpus.

## Graph Freshness
- Built from commit: `cc15ccd4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Web UI templates and their shared macros|Web UI templates and their shared macros]]
- [[_COMMUNITY_Atomic artifact writes and their failure-mode tests|Atomic artifact writes and their failure-mode tests]]
- [[_COMMUNITY_RSS feed fetching and download verification|RSS feed fetching and download verification]]
- [[_COMMUNITY_End-to-end worker and runner tests|End-to-end worker and runner tests]]
- [[_COMMUNITY_Serial worker process control and tree-kill|Serial worker process control and tree-kill]]
- [[_COMMUNITY_Job HTTP API tests|Job HTTP API tests]]
- [[_COMMUNITY_Web UI HTTP layer tests and fixtures|Web UI HTTP layer tests and fixtures]]
- [[_COMMUNITY_Speaker mapping precedence and transcript fingerprinting|Speaker mapping precedence and transcript fingerprinting]]
- [[_COMMUNITY_Phase-2 regression tests|Phase-2 regression tests]]
- [[_COMMUNITY_The speaker mapping as pipeline input (ADR-010)|The speaker mapping as pipeline input (ADR-010)]]
- [[_COMMUNITY_Episode page artifact rendering and its acceptance criteria|Episode page artifact rendering and its acceptance criteria]]
- [[_COMMUNITY_Job queue unit tests|Job queue unit tests]]
- [[_COMMUNITY_The whycast pipeline step modules|The whycast pipeline step modules]]
- [[_COMMUNITY_Episode scanner guarantees on real filenames|Episode scanner guarantees on real filenames]]
- [[_COMMUNITY_The Backlog plan doc-001 phases and TASK-001..005|The Backlog plan: doc-001 phases and TASK-001..005]]
- [[_COMMUNITY_Single-step re-runs of one post-processing step|Single-step re-runs of one post-processing step]]
- [[_COMMUNITY_Job creation and listing over HTTP|Job creation and listing over HTTP]]
- [[_COMMUNITY_Speaker mapping editor tests and their paid-call tripwires|Speaker mapping editor tests and their paid-call tripwires]]
- [[_COMMUNITY_Transcription and diarization decisions, and the pipeline stage map|Transcription and diarization decisions, and the pipeline stage map]]
- [[_COMMUNITY_FastAPI app internals request state and SSE plumbing|FastAPI app internals: request state and SSE plumbing]]
- [[_COMMUNITY_Editor validators newlines, duplicate keys, rejections|Editor validators: newlines, duplicate keys, rejections]]
- [[_COMMUNITY_Golden-master tests for the monolith split|Golden-master tests for the monolith split]]
- [[_COMMUNITY_Episode scanner internals suffix peeling and kind ranking|Episode scanner internals: suffix peeling and kind ranking]]
- [[_COMMUNITY_Runner verdicts and feed jobs finish, fail, cancel, fetch|Runner verdicts and feed jobs: finish, fail, cancel, fetch]]
- [[_COMMUNITY_Versioned backups moved aside before a run (ADR-011)|Versioned backups moved aside before a run (ADR-011)]]
- [[_COMMUNITY_Speaker mapping application and staleness checks|Speaker mapping application and staleness checks]]
- [[_COMMUNITY_Recognising base_speakers.json as an input, not an artifact|Recognising <base>_speakers.json as an input, not an artifact]]
- [[_COMMUNITY_SSE progress stream tests|SSE progress stream tests]]
- [[_COMMUNITY_GPU setup CUDA selection and audio preparation|GPU setup: CUDA selection and audio preparation]]
- [[_COMMUNITY_Speaker mapping decisions ADR-004 and ADR-010|Speaker mapping decisions: ADR-004 and ADR-010]]
- [[_COMMUNITY_Index freshness and the rescan contract|Index freshness and the rescan contract]]
- [[_COMMUNITY_Beforeafter diff and the read-only config viewer|Before/after diff and the read-only config viewer]]
- [[_COMMUNITY_The rebuildable SQLite index over podcasts|The rebuildable SQLite index over podcasts/]]
- [[_COMMUNITY_The web UI's browser-side JavaScript|The web UI's browser-side JavaScript]]
- [[_COMMUNITY_The three human-input editors as an HTTP contract|The three human-input editors as an HTTP contract]]
- [[_COMMUNITY_Prompt editor path safety and its fixtures|Prompt editor path safety and its fixtures]]
- [[_COMMUNITY_Job log directories, parked verdicts and cancellation|Job log directories, parked verdicts and cancellation]]
- [[_COMMUNITY_Job context, cancellation and the free self-test|Job context, cancellation and the free self-test]]
- [[_COMMUNITY_The SQLite job queue claim, finish, list|The SQLite job queue: claim, finish, list]]
- [[_COMMUNITY_Before-snapshots and hostile-name safety|Before-snapshots and hostile-name safety]]
- [[_COMMUNITY_Backup call-site tests (ADR-009)|Backup call-site tests (ADR-009)]]
- [[_COMMUNITY_Runner event sink and the episode-level job handlers|Runner event sink and the episode-level job handlers]]
- [[_COMMUNITY_CUDA runtime regression tests (ADR-001)|CUDA runtime regression tests (ADR-001)]]
- [[_COMMUNITY_Runner handlers for post-processing and speaker steps|Runner handlers for post-processing and speaker steps]]
- [[_COMMUNITY_Discarded partial downloads and dropped .bak files (ADR-009)|Discarded partial downloads and dropped .bak files (ADR-009)]]
- [[_COMMUNITY_find_episodes.py rename-collision analysis script|find_episodes.py: rename-collision analysis script]]
- [[_COMMUNITY_Tests against the operator's real podcasts directory|Tests against the operator's real podcasts/ directory]]
- [[_COMMUNITY_Prompt editor tests|Prompt editor tests]]
- [[_COMMUNITY_Vocabulary editor validation and its 400s|Vocabulary editor validation and its 400s]]
- [[_COMMUNITY_Paid-call tripwires in the speaker tests|Paid-call tripwires in the speaker tests]]
- [[_COMMUNITY_Episode scanner tests prefix collisions|Episode scanner tests: prefix collisions]]
- [[_COMMUNITY_Loopback binding and the host allowlist (ADR-008)|Loopback binding and the host allowlist (ADR-008)]]
- [[_COMMUNITY_Application assembly and route registration|Application assembly and route registration]]
- [[_COMMUNITY_Post-processing steps cleanup, summary, blog, history|Post-processing steps: cleanup, summary, blog, history]]
- [[_COMMUNITY_Request body parsing JSON and form, with a size cap|Request body parsing: JSON and form, with a size cap]]
- [[_COMMUNITY_The speaker editor's view of disk state|The speaker editor's view of disk state]]
- [[_COMMUNITY_Recursive chunked summarization (ADR-005)|Recursive chunked summarization (ADR-005)]]
- [[_COMMUNITY_Queue robustness terminal jobs, foreign schemas, two runners|Queue robustness: terminal jobs, foreign schemas, two runners]]
- [[_COMMUNITY_Artifact naming shapes and duplicate slots|Artifact naming shapes and duplicate slots]]
- [[_COMMUNITY_The Host allowlist and DNS-rebinding guard|The Host allowlist and DNS-rebinding guard]]
- [[_COMMUNITY_Environment-variable configuration (ADR-007)|Environment-variable configuration (ADR-007)]]
- [[_COMMUNITY_Index queries episodes and their artifacts|Index queries: episodes and their artifacts]]
- [[_COMMUNITY__safe_file the only gate between a request and the filesystem|_safe_file: the only gate between a request and the filesystem]]
- [[_COMMUNITY_Model selection, prompts and the ADR index|Model selection, prompts and the ADR index]]
- [[_COMMUNITY_Command-line entry points|Command-line entry points]]
- [[_COMMUNITY_Output formats markdown, HTML and wiki|Output formats: markdown, HTML and wiki]]
- [[_COMMUNITY_Windows entry point no console, no stdout|Windows entry point: no console, no stdout]]
- [[_COMMUNITY_Chunking and truncation, pinned by golden snapshots|Chunking and truncation, pinned by golden snapshots]]
- [[_COMMUNITY_The episodes API and its status matrix|The episodes API and its status matrix]]
- [[_COMMUNITY_Chunked request bodies cannot bypass the size cap|Chunked request bodies cannot bypass the size cap]]
- [[_COMMUNITY_App lifespan opening the index and refreshing it when stale|App lifespan: opening the index and refreshing it when stale]]
- [[_COMMUNITY_Built-in fallback pages when a template is missing|Built-in fallback pages when a template is missing]]
- [[_COMMUNITY_The job catalogue the UI and API share|The job catalogue the UI and API share]]
- [[_COMMUNITY_Connection and lookup helpers, and their 404s and 503s|Connection and lookup helpers, and their 404s and 503s]]
- [[_COMMUNITY_Reading a hand-edited mapping file|Reading a hand-edited mapping file]]
- [[_COMMUNITY_Awkward but legal input wildcards, non-ASCII, zero bytes|Awkward but legal input: wildcards, non-ASCII, zero bytes]]
- [[_COMMUNITY_transcript_formatter.py episode header script|transcript_formatter.py: episode header script]]
- [[_COMMUNITY_Diarization and the HuggingFace token|Diarization and the HuggingFace token]]
- [[_COMMUNITY_Timestamped transcript writing|Timestamped transcript writing]]
- [[_COMMUNITY_Every io_utils writer call site, checked by AST|Every io_utils writer call site, checked by AST]]
- [[_COMMUNITY_Queue concurrency eight threads, two processes, one job each|Queue concurrency: eight threads, two processes, one job each]]
- [[_COMMUNITY_The operator's real input files must pass the editors' validators|The operator's real input files must pass the editors' validators]]
- [[_COMMUNITY_The artifact diff, read safely|The artifact diff, read safely]]
- [[_COMMUNITY_Editor file state and human-input writes|Editor file state and human-input writes]]
- [[_COMMUNITY_Queue schema creation and migration|Queue schema creation and migration]]
- [[_COMMUNITY_Vocabulary correction (ADR-006)|Vocabulary correction (ADR-006)]]
- [[_COMMUNITY_Path traversal, symlinks and null bytes|Path traversal, symlinks and null bytes]]
- [[_COMMUNITY_Episode detail API case, spaces, 404s|Episode detail API: case, spaces, 404s]]
- [[_COMMUNITY_Server bind flags versus environment|Server bind flags versus environment]]
- [[_COMMUNITY_Correction workflows and the config viewer (TASK-004 and TASK-005)|Correction workflows and the config viewer (TASK-004 and TASK-005)]]
- [[_COMMUNITY_Artifact response headers and framable 404s|Artifact response headers and framable 404s]]
- [[_COMMUNITY_Picking the right artifact for a kind|Picking the right artifact for a kind]]
- [[_COMMUNITY_The worker's environment and its logging sink|The worker's environment and its logging sink]]
- [[_COMMUNITY_Index schema versioning|Index schema versioning]]
- [[_COMMUNITY_A live TestClient for the SSE stream|A live TestClient for the SSE stream]]
- [[_COMMUNITY_The index is disposable and never writes to podcasts|The index is disposable and never writes to podcasts/]]
- [[_COMMUNITY_A missing podcast directory is not a crash|A missing podcast directory is not a crash]]
- [[_COMMUNITY_Package docstrings whycast, webui and the pipeline|Package docstrings: whycast, webui and the pipeline]]
- [[_COMMUNITY_Writing base_cleaned.txt only when cleanup changed something|Writing <base>_cleaned.txt only when cleanup changed something]]
- [[_COMMUNITY_Synthetic podcast directory fixtures|Synthetic podcast directory fixtures]]
- [[_COMMUNITY_SSE frame formatting|SSE frame formatting]]
- [[_COMMUNITY_Listing jobs and normalising the status filter|Listing jobs and normalising the status filter]]
- [[_COMMUNITY_The speaker prompts and the WHYcast co-hosts|The speaker prompts and the WHYcast co-hosts]]
- [[_COMMUNITY_Scratch queue fixtures|Scratch queue fixtures]]
- [[_COMMUNITY_Community 104|Community 104]]
- [[_COMMUNITY_Community 105|Community 105]]
- [[_COMMUNITY_Community 106|Community 106]]
- [[_COMMUNITY_Community 107|Community 107]]
- [[_COMMUNITY_Community 108|Community 108]]
- [[_COMMUNITY_Community 109|Community 109]]
- [[_COMMUNITY_Community 110|Community 110]]
- [[_COMMUNITY_Community 111|Community 111]]
- [[_COMMUNITY_Community 112|Community 112]]
- [[_COMMUNITY_Community 113|Community 113]]
- [[_COMMUNITY_Community 114|Community 114]]
- [[_COMMUNITY_Community 115|Community 115]]
- [[_COMMUNITY_Community 116|Community 116]]
- [[_COMMUNITY_Community 117|Community 117]]
- [[_COMMUNITY_Community 119|Community 119]]
- [[_COMMUNITY_Community 120|Community 120]]
- [[_COMMUNITY_Community 121|Community 121]]
- [[_COMMUNITY_Community 122|Community 122]]
- [[_COMMUNITY_Community 123|Community 123]]
- [[_COMMUNITY_Community 124|Community 124]]
- [[_COMMUNITY_Community 125|Community 125]]

## God Nodes (most connected - your core abstractions)
1. `scan_podcasts()` - 53 edges
2. `enqueue()` - 43 edges
3. `atomic_write_text()` - 39 edges
4. `emit()` - 38 edges
5. `ConfigurationError` - 36 edges
6. `Episode` - 35 edges
7. `speaker_map_path()` - 33 edges
8. `run_step()` - 32 edges
9. `The WHYcast Pipeline` - 31 edges
10. `main()` - 26 edges

## Surprising Connections (you probably didn't know these)
- `Audio Downloads Are Not Backed Up` --conceptually_related_to--> `atomic_writer()`  [INFERRED]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/docs/adr/ADR-011-versioned-artifact-backups-moved-aside-before-every-run.md → whycast/io_utils.py
- `Prompt Editor Page` --conceptually_related_to--> `split_into_chunks()`  [AMBIGUOUS]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/webui/templates/prompts.html → whycast/pipeline/llm.py
- `Speaker Mapping Editor Page` --references--> `parse_speaker_mapping_from_analysis()`  [AMBIGUOUS]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/webui/templates/speakers.html → whycast/pipeline/speakers.py
- `estimate_token_count()` --conceptually_related_to--> `ADR-005 Recursive chunked summarization for long transcripts`  [INFERRED]
  whycast/pipeline/llm.py → D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/docs/adr/ADR-005-recursive-chunked-summarization-for-long-transcripts.md
- `Overlapping Transcript Chunk Windows` --rationale_for--> `split_into_chunks()`  [INFERRED]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/tests/golden/split_into_chunks.snapshot.txt → whycast/pipeline/llm.py

## Hyperedges (group relationships)
- **Fase 0: monoliet naar importeerbare library met golden-master gate** — backlog_docs_doc_001___webui_plan_whycast_transcribe_fase_0_pipeline_extractie, backlog_tasks_task_001___extract_whycast_pipeline_library_uit_transcribe_py_webui_fase_0_task_001, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_whycast_pipeline_library, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_eventsink_progress_protocol, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_transcribe_py_cli_shim, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_golden_master_test [EXTRACTED 1.00]
- **Eén GPU dwingt seriële job-uitvoering met crash-isolatie** — docs_adr_adr_001_use_faster_whisper_on_cuda_for_transcription_faster_whisper_on_cuda, docs_adr_adr_002_speaker_diarization_via_pyannote_audio_3_1_pyannote_speaker_diarization_pipeline, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_serial_worker_subprocess, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_sqlite_index_and_queue, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_sse_jsonl_progress, backlog_tasks_task_003___webui_fase_2_job_queue_worker_en_live_voortgang_seriele_worker_met_lockfile [EXTRACTED 1.00]
- **Menselijke correcties leven in bewerkbare pipeline-invoerbestanden** — docs_adr_adr_006_post_transcription_vocabulary_correction_via_vocabulary_json_vocabulary_json_replacement_map, docs_adr_adr_007_environment_variable_configuration_and_file_based_prompts_env_vars_and_file_based_prompts, backlog_tasks_task_004___webui_fase_3_correctie_workflows_sprekers_vocabulary_prompts_speakers_json_pipeline_input, backlog_tasks_task_004___webui_fase_3_correctie_workflows_sprekers_vocabulary_prompts_input_first_correctie_principe [INFERRED 0.85]
- **Artifact Durability Regime (inputs preserved, outputs versioned)** — docs_adr_adr_009_discard_partial_downloads_and_drop_per_artifact_backups_adr_009, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_adr_010, docs_adr_adr_011_versioned_artifact_backups_moved_aside_before_every_run_adr_011, docs_adr_adr_011_versioned_artifact_backups_moved_aside_before_every_run_backups_directory, docs_adr_adr_009_discard_partial_downloads_and_drop_per_artifact_backups_human_input_never_in_generated_artifacts, docs_pipeline_input_output_rule [INFERRED 0.85]
- **Speaker Mapping Correction Loop** — docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_speakers_json, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_speaker_assignment_step, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_analyze_speakers_with_o4, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_mapping_file_precedence, tests_golden_apply_speaker_mapping_programmatically_snapshot_apply_speaker_mapping_programmatically, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_stale_mapping_flag [INFERRED 0.85]
- **Episode Processing Flow (audio to blog)** — docs_pipeline_prepare_audio, docs_pipeline_diarize, docs_pipeline_transcribe, docs_pipeline_merge_speaker_lines_step, docs_pipeline_speaker_assignment, docs_pipeline_cleanup, docs_pipeline_summary, docs_pipeline_blog [EXTRACTED 1.00]
- **Two Gates Before Any Paid API Call** — webui_templates_episode_episode_detail, webui_templates_jobs_job_dashboard, webui_templates_speakers_editor_page, webui_templates_vocabulary_editor_page, webui_templates_prompts_editor_page, webui_templates__macros_cost_badge, webui_templates_episode_two_gates [INFERRED 0.85]
- **Human-Written Pipeline Inputs Keep a .bak** — webui_templates_vocabulary_editor_page, webui_templates_prompts_editor_page, webui_templates_speakers_editor_page, webui_templates_vocabulary_input_gets_backup, webui_templates_job_diff_before_snapshot [INFERRED 0.85]
- **Untrusted Pipeline Output Never Enters the Live DOM** — webui_templates_base_page_skeleton, webui_templates__macros_never_safe, webui_templates_episode_artifact_tabs, webui_templates_job_detail_sse_log_replay, webui_templates_job_diff_diff_page, webui_templates_base_htmx_is_not_a_sanitiser [INFERRED 0.85]

## Communities (126 total, 25 thin omitted)

### Community 0 - "Web UI templates and their shared macros"
Cohesion: 0.05
Nodes (93): Layered Speaker Mapping Parse Fallbacks, abbrev() Macro, Artifact Row Shape (detail page only), cancel_form() Macro, Cancel Is a Plain POST Form That Works Without JavaScript, cell() Macro, cost_badge() Macro, duration() Macro (+85 more)

### Community 1 - "Atomic artifact writes and their failure-mode tests"
Cohesion: 0.05
Nodes (75): Atomic Artifact Write (temp file then os.replace), whycast.io_utils, leftovers(), _on_disk(), Tests for whycast.io_utils - atomic artifact writes (ADR-008).  Why these exis, A crash at the commit must not cost the reader the old file.      This is the, The other half of the failure surface: it breaks *before* the rename.      ``t, Same precondition as open(): io_utils does not create directories. (+67 more)

### Community 2 - "RSS feed fetching and download verification"
Cohesion: 0.05
Nodes (71): BaseHTTPRequestHandler, delete_episode_files(), download_all_episodes_from_rssfeed(), _download_audio(), _enclosure_length(), _header_length(), podcast_fetching_workflow(), Size the RSS enclosure claims, in bytes, or None when it does not say.      Pa (+63 more)

### Community 3 - "End-to-end worker and runner tests"
Cohesion: 0.05
Nodes (65): wait_until(), assert_queue_is_free(), enqueue_selftest(), peek_events(), End-to-end tests for :mod:`webui.worker` and :mod:`webui.runner` (ADR-008, TASK, Refuse to start a worker while the queue holds a job that costs money.      Th, One claim-run-exit cycle, after the cost guard has had its say., Start ``run_worker(once=True)`` in a thread; returns ``(thread, box)``.      ` (+57 more)

### Community 4 - "Serial worker process control and tree-kill"
Cohesion: 0.06
Nodes (49): _apply_pending_verdict(), _cancel_wanted(), is_runner_for_job(), _kernel32(), _kill_hint(), kill_process_tree(), lock_path(), _log_tail() (+41 more)

### Community 5 - "Job HTTP API tests"
Cohesion: 0.05
Nodes (39): finished_selftest(), job_env(), live_server(), Tests for the job HTTP API of the WHYcast web UI (ADR-008, TASK-003 phase 2)., Run the queue to completion with a real worker, after the cost guard.      The, Enqueue a self-test, run it with a real worker, and return the row., The request is wrong, not the URL: nothing about /api/jobs is missing., base_name is an index key. A path is simply a name the index does not have. (+31 more)

### Community 6 - "Web UI HTTP layer tests and fixtures"
Cohesion: 0.04
Nodes (29): awkward_client(), client_with_all_formats(), configured_env(), Tests for the WHYcast web UI HTTP layer (ADR-008 / TASK-002 phase 1).  Everyth, Every format in the scanner's vocabulary must be servable., Point the ADR-007 environment variables at the temporary directories.      ``c, TestArtifactFormats, TestHealth (+21 more)

### Community 7 - "Speaker mapping precedence and transcript fingerprinting"
Cohesion: 0.08
Nodes (43): fingerprint_transcript(), SHA-256 of the transcript a mapping was derived from.      Cheap and determini, Write ``<base>_speakers.json`` atomically, keeping a backup.      Args:, save_speaker_mapping(), ADR-010's precedence rule, checked where it actually costs money.  ``<base>_sp, ADR-010's first Confirmation: mapped transcript, no OpenAI call., Somebody who cannot see their correction being used will not trust it.      Th, ADR-010's second Confirmation: the LLM path is unchanged, plus a write. (+35 more)

### Community 8 - "Phase-2 regression tests"
Cohesion: 0.06
Nodes (26): queue(), Regression tests for the phase-2 review findings (ADR-008, TASK-003).  One cla, Force a job row into ``running`` with a given pid, bypassing the worker., It was ``<WHYCAST_JOBS_DIR>/worker.lock`` while the queue is     ``WHYCAST_WEBU, ``claim_next`` published ``running`` with ``pid`` still NULL, and the     reape, Belt and braces for rows written by an older build., ``_settle`` called ``finish`` once and swallowed ``sqlite3.Error``. One     tra, Otherwise the rescue channel rescues the status and loses the reason. (+18 more)

### Community 9 - "The speaker mapping as pipeline input (ADR-010)"
Cohesion: 0.07
Nodes (44): Path of the persisted mapping for one episode.      Args:         output_base, speaker_map_path(), test_the_path_is_the_one_the_scanner_and_the_editor_agree_on(), The speaker mapping as a pipeline input (ADR-010, TASK-004 phase 3).  ``tests/, Write a mapping file exactly as given - a hand edit, not a save., Pins the outcome the removed gpt-4o call did not affect.      Inside :func:`sp, The decision procedure itself, on a transcript that still has labels.      Sam, Different speakers either side is a real handover; guessing would be wrong. (+36 more)

### Community 10 - "Episode page artifact rendering and its acceptance criteria"
Cohesion: 0.04
Nodes (15): AC #1: the matrix must be *right*, not merely rendered.          The overview, AC #2: every artifact the episode has is reachable from the page., AC #2: the mp3 must be playable from the page., Artifacts are prompt-injectable LLM output; they load sandboxed.          A pr, AC #3: junk is visible, not silently dropped., ``list_unmatched`` yields dicts; iterating them as strings printed         the, Artifacts are LLM output: never trusted, never merged into a page., ``_speaker_assignment`` vs ``_ts_speaker_assignment``: serve the new one. (+7 more)

### Community 12 - "The whycast pipeline step modules"
Cohesion: 0.08
Nodes (24): Audio preparation step for the WHYcast pipeline (ADR-008).  Extracted verbatim, RSS feed fetching and episode download for the WHYcast pipeline (ADR-008).  Fu, LLM (OpenAI) processing step for the WHYcast pipeline (ADR-008).  Functions ex, # NOTE: GPT-4o is NOT considered an o-series model in this context - it uses sta, Output format conversion for the WHYcast pipeline (ADR-008).  Functions extrac, Transcript post-processing steps for the WHYcast pipeline (ADR-008).  Extracte, Whisper transcription step for the WHYcast pipeline (ADR-008).  Extracted verb, Vocabulary correction step for the WHYcast pipeline (ADR-008).  Extracted verb (+16 more)

### Community 13 - "Episode scanner guarantees on real filenames"
Cohesion: 0.09
Nodes (18): ``number`` is not a key. ``base_name`` is., The module's headline guarantee: nothing in the directory vanishes., The scanner reads directory metadata and nothing else (ADR-008)., NTFS holds ``Episode_28.mp3`` next to ``episode_28_summary.txt``.      Treatin, The pipeline can legitimately write an empty file., ``episode32.m4a`` next to ``episode32.mp3`` is live on disk., TestCaseInsensitivity, TestEpisodeNumbers (+10 more)

### Community 14 - "The Backlog plan: doc-001 phases and TASK-001..005"
Cohesion: 0.11
Nodes (36): WHYcast-transcribe Backlog Project Config, doc-001 Fase 0: Pipeline-extractie, doc-001 Fase 1: Read-only episode-browser, doc-001 Fase 2: Jobs starten, volgen, annuleren, doc-001 Fase 3: Correctie-workflows, doc-001 Fase 4: Afwerking (optioneel), doc-001 WebUI-plan WHYcast-transcribe, TASK-001 Multi-agent extractie met adversariële review (+28 more)

### Community 15 - "Single-step re-runs of one post-processing step"
Cohesion: 0.08
Nodes (32): The one episode with this base name, or a readable failure., _Ctx, Single-step re-runs: one post-processing step at a time.  Editing one prompt a, summary not found" leaves the reader to work out the next move., An empty file is not an input; sending it to a model wastes the call., The path comes from the index, and is still re-checked against the root., Running the blog step must not regenerate the summary as a side effect., ``import webui.runner`` must work without webui.jobs being loaded first. (+24 more)

### Community 16 - "Job creation and listing over HTTP"
Cohesion: 0.06
Nodes (33): Create a job through the API; fail loudly if it was not accepted., A row from a newer build must never make the UI under-warn., test_a_created_job_carries_the_cost_flag_everywhere_it_appears(), test_a_row_of_an_unknown_type_is_reported_as_expensive(), test_cancel_is_a_post_only_route(), test_cancelling_a_queued_job_settles_it_outright(), test_one_job_can_be_read_back_and_an_unknown_one_is_a_404(), test_params_are_stored_as_arguments_never_as_a_path() (+25 more)

### Community 17 - "Speaker mapping editor tests and their paid-call tripwires"
Cohesion: 0.1
Nodes (28): no_paid_calls(), Every route to OpenAI in the speakers module, closed.      ``OpenAI`` is in th, map_file(), Tests for the speaker mapping editor in the web UI (ADR-010, TASK-004).  Every, AC: every SPEAKER_xx label, with lines a person can recognise a voice by., ADR-009: human input keeps a .bak. Generated artifacts do not., The assertion this whole feature stands on.      If the editor hashes a differ, Both body shapes fail on the same input, not one silently ignoring it. (+20 more)

### Community 18 - "Transcription and diarization decisions, and the pipeline stage map"
Cohesion: 0.12
Nodes (32): ADR-001 Use faster-whisper on CUDA for transcription, ADR-001: faster-whisper (CTranslate2) on CUDA float16, ADR-002 Speaker diarization via pyannote.audio 3.1, ADR-002: pyannote.audio 3.1 speaker-diarization pipeline, ADR-006 Post-transcription vocabulary correction via vocabulary.json, Restore Anything Moved Aside but Not Rewritten, 4e. Alternative Blog, 4d. Blog (+24 more)

### Community 19 - "FastAPI app internals: request state and SSE plumbing"
Cohesion: 0.07
Nodes (29): _apply_targets(), _audio_of(), _episode_jobs(), _last_event_id(), _prompt_spec_or_404(), _public_speaker_state(), FastAPI application for the WHYcast web UI (ADR-008, TASK-002/TASK-003).  Read, The API view of :func:`_speaker_state`: no absolute paths.      ``map_path`` i (+21 more)

### Community 20 - "Editor validators: newlines, duplicate keys, rejections"
Cohesion: 0.09
Nodes (25): Exception, _EditorRejected, A save this app refuses to write, with a message for the operator.      Separa, ``object_pairs_hook`` that refuses a repeated key.      ``json.loads`` keeps t, _reject_duplicate_keys(), JsonlEventSink, An :class:`whycast.events.EventSink` that appends JSON lines to a file.      O, Close the file. Safe to call twice. (+17 more)

### Community 21 - "Golden-master tests for the monolith split"
Cohesion: 0.12
Nodes (26): check_golden(), fake_process_with_openai(), new_func(), _normalize(), old_mod(), Golden extraction tests for the ADR-008 monolith -> whycast package split.  Tw, The legacy transcribe monolith, or None if it can no longer be imported., Stable text form for snapshot files. (+18 more)

### Community 22 - "Episode scanner internals: suffix peeling and kind ranking"
Cohesion: 0.1
Nodes (25): test_parse_episode_number(), Artifact, belongs_to_base(), _derive_base(), _fmt_rank(), _kind_of_rest(), _kind_rank(), _matching_base() (+17 more)

### Community 23 - "Runner verdicts and feed jobs: finish, fail, cancel, fetch"
Cohesion: 0.11
Nodes (26): Trim an error to a UI-sized blurb, pointing at the event log for more., _short_error(), _duration(), _finish_cancelled(), _finish_failed(), _job_fetch_all(), _job_fetch_latest(), _make_step_handler() (+18 more)

### Community 24 - "Versioned backups moved aside before a run (ADR-011)"
Cohesion: 0.11
Nodes (27): Move Aside Before the Run Starts, _JOB_INPUT_KINDS, Per-Job Table of Kinds Read and Written, _JOB_OUTPUT_KINDS, webui.runner, whycast.backups, Pipeline Job Types, Single-Step Re-run (+19 more)

### Community 25 - "Speaker mapping application and staleness checks"
Cohesion: 0.1
Nodes (26): analyze_transcript_changes(), _announce_missing_mapping(), _announce_saved_mapping(), _check_mapping_covers_transcript(), _clean_mapping(), load_speaker_map_record(), _malformed_mapping(), mapping_is_stale() (+18 more)

### Community 26 - "Recognising <base>_speakers.json as an input, not an artifact"
Cohesion: 0.09
Nodes (16): _names(), ``<base>_speakers.json`` is a pipeline *input* (ADR-010).      A person edits, The web UI needs to offer an input for editing, not for download., ``episode_1_summary.json`` names a real base and a real kind.          It is s, ``random_summary.json`` must not mint an episode ``random``.          The mint, ``episode_70_speakers.json`` has no audio and no results behind it., Minting nothing must not mean detaching: ``_ts`` mints, json joins., The suffix alone does not make it one: the mapping is JSON. (+8 more)

### Community 27 - "SSE progress stream tests"
Cohesion: 0.09
Nodes (27): parse_sse(), Split an SSE body into ``(event_name, event_id, payload)`` triples., A queued job has written nothing. That is a state, not a failure., A job id becomes a path in exactly one place, and that place checks it.      T, test_the_stream_cannot_be_pointed_outside_the_jobs_directory(), test_the_stream_of_a_job_with_no_log_yet_is_not_an_error(), event_record(), Append JSONL event lines the way :class:`webui.runner.JsonlEventSink` does. (+19 more)

### Community 28 - "GPU setup: CUDA selection and audio preparation"
Cohesion: 0.09
Nodes (25): prepare_audio_for_diarization(), Ensures the audio file is in a compatible mono 16kHz mp3 format for diarization/, enable_tf32(), force_cuda_device(), get_default_device(), is_cuda_available(), maximize_gpu_utilization(), GPU setup and utilization helpers for the WHYcast pipeline (ADR-008).  Extract (+17 more)

### Community 29 - "Speaker mapping decisions: ADR-004 and ADR-010"
Cohesion: 0.13
Nodes (26): ADR-004 Hybrid AI-analysis plus programmatic speaker assignment, ADR-010 Persisted Editable Speaker Mapping as Pipeline Input, Mapping File Precedence Over Paid Analysis, <base>_speakers.json Speaker Mapping File, Stale Mapping Flag After Re-transcription, TASK-004 WebUI Phase 3 Correction Workflows, Unmapped-Label Warning, whycast.episodes (+18 more)

### Community 30 - "Index freshness and the rescan contract"
Cohesion: 0.09
Nodes (17): _dump_index(), Write ``content`` byte for byte.      Not ``Path.write_text``: on Windows that, A restart must reflect the disk, including files rewritten in place.      ``_n, Every row of the three index tables, in scan order.      Used to prove rescan, ADR-008 Must: configuration surfaces go through an allowlist., TestIndexFreshness, TestRescan, write() (+9 more)

### Community 31 - "Before/after diff and the read-only config viewer"
Cohesion: 0.1
Nodes (16): env(), _job_with_snapshot(), Phase 4 (TASK-005): the before-snapshot diff and the configuration viewer.  Tw, Create a job row, snapshot `before`, then leave `after` on disk., The allowlist must hold even when the environment is full of secrets., A denylist would fail open here; the allowlist is what makes this pass.      S, A web UI pointed at throwaway directories, never the real podcasts/., A 40 MB diff helps nobody, and the job log is not the place to grow one. (+8 more)

### Community 32 - "The rebuildable SQLite index over podcasts/"
Cohesion: 0.12
Nodes (23): _as_float(), _as_int(), _database_path(), _ensure_schema(), get_meta(), IndexConsistencyError, init_db(), _insert_episodes() (+15 more)

### Community 33 - "The web UI's browser-side JavaScript"
Cohesion: 0.21
Nodes (22): applyJobStatus(), describeError(), formatDuration(), handleCancel(), handleEnqueue(), handleSpeakerDiscard(), handleSpeakerSave(), note() (+14 more)

### Community 34 - "The three human-input editors as an HTTP contract"
Cohesion: 0.09
Nodes (21): map_path(), The three human-input editors, as an HTTP contract (TASK-004 phase 3).  ``test, ``parse_qsl`` drops blanks by default; the editors need them kept.      A blan, Every name the API offers resolves; nothing else does., Saved first, so this cannot pass by the name having been rejected., The vocabulary editor renders the file into a textarea., The validators quote what was submitted; that is user input too., Whatever a bad paste does, it must not take the server with it. (+13 more)

### Community 35 - "Prompt editor path safety and its fixtures"
Cohesion: 0.09
Nodes (14): inputs_dir(), The prompt editor's whole path-safety story: a key, never a path., Point every editable input file at ``tmp_path``., test_a_prompt_name_outside_the_allowlist_is_refused(), Tests for the vocabulary and prompt editors (ADR-008 phase 3, TASK-004).  Thes, The classic attempt, spelled several ways, against a real target file.      ``, An editable file is named relative to the repository, or by filename only., The page offers a job; the API must accept exactly that job.      This enqueue (+6 more)

### Community 36 - "Job log directories, parked verdicts and cancellation"
Cohesion: 0.14
Nodes (21): cancel_requested(), clear_pending_verdict(), ensure_job_dir(), events_path(), finish_or_record(), jobs_dir(), mark_running(), The job queue for the WHYcast web UI (ADR-008, TASK-003 phase 2).  This module (+13 more)

### Community 37 - "Job context, cancellation and the free self-test"
Cohesion: 0.11
Nodes (20): ensure_api_key(), Ensure that the OpenAI API key is available and valid.          Returns:, _Context, _job_selftest(), JobCancelled, Raised inside the runner when a cancel request is seen between steps., Everything a job handler needs, resolved once and validated once., Stop the job if a cancel has been requested. Called between steps.          As (+12 more)

### Community 38 - "The SQLite job queue: claim, finish, list"
Cohesion: 0.13
Nodes (20): active_job(), _begin(), claim_next(), _claim_next_sql(), finish(), get_job(), _job_dict(), _params_dict() (+12 more)

### Community 39 - "Before-snapshots and hostile-name safety"
Cohesion: 0.13
Nodes (19): Every file under ``root``, mapped to its bytes.      The SQLite index is skipp, ``base_name`` is an opaque index key, never joined onto a directory., test_a_hostile_base_name_writes_no_speaker_mapping(), The load-bearing property is not the status code; it is that nothing moved., Every file under ``root``, mapped to its bytes. Used to prove absence., test_a_hostile_base_name_touches_nothing(), test_a_label_that_is_not_in_the_transcript_is_refused(), load_manifest() (+11 more)

### Community 40 - "Backup call-site tests (ADR-009)"
Cohesion: 0.13
Nodes (18): ADR-009's backup decision, pinned at the call sites instead of only at the prim, The control for every test above. If this fails they prove nothing., ADR-009 Must Not: do not change the io_utils default to backup=False., The behaviour the keyword buys, checked by writing twice (ADR-010)., ``<base>_cleaned.txt`` is an artifact like any other: regenerable, no .bak., ``.env`` is the one file that must be atomic *and* must have no ``.bak``., The signature half of the same promise, checked without writing a file., Everything that is neither the artifact itself nor an artifact format. (+10 more)

### Community 41 - "Runner event sink and the episode-level job handlers"
Cohesion: 0.12
Nodes (19): Protocol, _job_force_episode(), _job_full_episode(), _job_retranscribe(), Progress implied by a ``[2/4]`` step marker in a pipeline message.      :func:, Append one event. Never raises., Diarize and transcribe again, and stop before anything paid.      The only GPU, Run the full pipeline on one episode's existing audio. (+11 more)

### Community 42 - "CUDA runtime regression tests (ADR-001)"
Cohesion: 0.16
Nodes (15): CUDA runtime regression tests (ADR-001).  The failure these guard against: CTr, Repeated calls must not keep appending to PATH., The CUDA 12 cuBLAS that CTranslate2 needs must exist somewhere on PATH., A tiny silent-ish WAV; content does not matter, only that decoding works., End-to-end GPU proof: transcribe actual audio.      Runs in a subprocess so a, test_cublas12_is_reachable(), test_cuda_libs_land_on_path_when_wheels_installed(), test_ensure_cuda_libs_is_idempotent() (+7 more)

### Community 43 - "Runner handlers for post-processing and speaker steps"
Cohesion: 0.14
Nodes (18): _float_param(), _job_postprocess(), _job_speakers(), JobInputError, The job cannot be started as specified - exit code 2.      Distinct from :clas, Read one artifact this step needs, or say precisely what is missing.      A si, Re-run cleanup, summary, blog and history from the transcript on disk.      Th, Re-run speaker identification and assignment over the existing transcript. (+10 more)

### Community 44 - "Discarded partial downloads and dropped .bak files (ADR-009)"
Cohesion: 0.17
Nodes (18): ADR-009 Discard Partial Downloads and Drop Per-Artifact Backups, Explicit backup=False at Artifact Call Sites, Content-Length Download Verification, Human Input Never Lives in a Generated Artifact, Per-Artifact .bak Backups, whycast.pipeline.feed, <base>_speaker_analysis.txt Report, ADR-011 Versioned Artifact Backups Moved Aside Before Every Run (+10 more)

### Community 45 - "find_episodes.py: rename-collision analysis script"
Cohesion: 0.15
Nodes (16): analyze_episodes(), detect_rename_collisions(), find_mp3_files(), generate_backup_strategy(), generate_rename_script(), print_collision_analysis(), print_episode_analysis(), Print formatted analysis of episodes.          Args:         episodes: List o (+8 more)

### Community 46 - "Tests against the operator's real podcasts/ directory"
Cohesion: 0.12
Nodes (5): Scans the operator's actual ``podcasts/``.      Every assertion here must surv, ``webui.db.artifacts.path`` is a PRIMARY KEY; a repeat breaks rescan., Where a ``(kind, fmt)`` slot repeats, it is two real files, resolvable., ``webui.db`` keys on ``base_name.lower()`` and refuses collisions., TestRealPodcastDirectory

### Community 47 - "Prompt editor tests"
Cohesion: 0.12
Nodes (16): prompt_path(), A prompt is free-form prose, so it is the easiest place to hide markup., The two human-input editors used to disagree about "plain text".      The spea, The control: rejecting all C0 controls would reject every prompt., ``cleanup_prompt.txt`` does not exist in this fixture. Creating it is not     a, test_a_control_character_in_a_prompt_is_refused(), test_a_new_prompt_file_needs_no_backup(), test_a_prompt_keeps_one_backup() (+8 more)

### Community 48 - "Vocabulary editor validation and its 400s"
Cohesion: 0.12
Nodes (16): The control. A cap that refuses everything proves nothing., ``RecursionError`` is neither ``ValueError`` nor ``JSONDecodeError``.      It, Valid JSON source, but no UTF-8 exists for it.      It passed every validator, The blocker: the value was used as an ``re.sub`` replacement template.      ``, The control for the four above., test_a_form_body_within_the_cap_still_works(), test_a_lone_surrogate_is_a_400_not_a_500(), test_a_rejected_vocabulary_leaves_the_file_byte_identical() (+8 more)

### Community 49 - "Paid-call tripwires in the speaker tests"
Cohesion: 0.16
Nodes (13): _NoNetworkOpenAI, Constructor always raises, so the direct client path in     attribute_unknown_s, A stand-in for the paid analysis that records whether it was wanted., Branch 1 needs no model, so no API key should be required to honour it.      R, Stands in for a paid call. Being called at all is the failure., The outcome of one patched ``speaker_assignment_step``., _Recorder, _Run (+5 more)

### Community 50 - "Episode scanner tests: prefix collisions"
Cohesion: 0.16
Nodes (10): _artifact_names(), Tests for :mod:`whycast.episodes` - the tolerant podcast directory scanner (ADR, A podcast directory holding every naming trap, and nothing else.      Returns, ``episode_1`` and ``episode_10`` both exist; neither may swallow the other., ``episode_1.foo_summary.txt`` belongs to episode_1 and mints nothing., ``episode_10_speakers.json`` is episode_10's, never episode_1's., real(), _set_mtime() (+2 more)

### Community 51 - "Loopback binding and the host allowlist (ADR-008)"
Cohesion: 0.13
Nodes (4): AC #5 / ADR-008 Must: "The web server must bind to 127.0.0.1 by default"., The flag defaults to None so the env/default answer stays the one answer., DNS rebinding: a bind address alone does not stop it., TestBindingDefaults

### Community 52 - "Application assembly and route registration"
Cohesion: 0.13
Nodes (15): create_app(), db_path_from_env(), The speaker mapping editor: one page and three API routes., Queue API, SSE stream, and the two job pages., The vocabulary and prompt editors, and their JSON API.      Every route in her, Render HTTP errors as a page for browsers, as JSON for the API.      Registere, Index database path from ``WHYCAST_WEBUI_DB``, else the repo default., Build the FastAPI application.      Arguments override the environment; both d (+7 more)

### Community 53 - "Post-processing steps: cleanup, summary, blog, history"
Cohesion: 0.2
Nodes (14): choose_appropriate_model(), Choose the appropriate model based on transcript length.          Args:, Read a prompt from file.          Args:         prompt_file: Path to the prom, read_prompt_file(), alt_blog_step(), blog_step(), cleanup_step(), history_step() (+6 more)

### Community 54 - "Request body parsing: JSON and form, with a size cap"
Cohesion: 0.19
Nodes (14): _declared_length(), _editor_payload(), _form_fields(), _is_form_request(), _job_request_payload(), Read ``{label: name}`` from a JSON or form-encoded body.      JSON: ``{"speake, ``Content-Length`` as an int, or None when absent or unparseable., True for ``application/x-www-form-urlencoded``, whatever the parameters. (+6 more)

### Community 55 - "The speaker editor's view of disk state"
Cohesion: 0.15
Nodes (14): _labels_with_context(), Import :mod:`whycast.pipeline.speakers`, or raise 503.      Imported lazily, p, Where this episode's mapping file lives, verified to be inside ``podcasts/``., Does this episode have a saved mapping? One stat, straight from disk.      Use, Every ``SPEAKER_xx`` label in ``text``, in order, with a few of its lines., Read a transcript exactly as the runner does. Returns ``(text, error)``., Everything the editor and its API need, read from disk. Blocking.      Called, Write the mapping with ``source="human"``. Blocking; run it in a thread. (+6 more)

### Community 56 - "Recursive chunked summarization (ADR-005)"
Cohesion: 0.21
Nodes (13): estimate_token_count(), process_large_text_in_chunks(), process_with_openai(), Handle very large transcripts by chunking and recursive summarization., Process text with OpenAI model using the specified prompt.          Args:, Process very large text by breaking it into chunks and reassembling the results., Estimate the number of tokens in a text.          Args:         text: The tex, summarize_large_transcript() (+5 more)

### Community 57 - "Queue robustness: terminal jobs, foreign schemas, two runners"
Cohesion: 0.21
Nodes (13): A hand-edited row must not take the whole dashboard down with it., The first verdict stands, and the attempt to overwrite it is announced., Two running jobs means two processes on one GPU. It must be visible., An older build pointed at a newer database must not 'fix' it., test_a_newer_queue_schema_is_left_untouched(), test_a_terminal_job_never_moves_again(), test_active_job_reports_the_oldest_and_warns_when_two_run(), test_cancelling_a_terminal_job_changes_nothing() (+5 more)

### Community 58 - "Artifact naming shapes and duplicate slots"
Cohesion: 0.15
Nodes (6): ``<base>_ts_speaker_assignment`` is the assignment, not a ts artifact., ``<base>.txt`` and ``<base>_transcript.txt`` are both kind=transcript., Two files in one ``(kind, fmt)`` slot: both indexed, newest served.          T, The mp3 is gone; the artifacts still form one episode, not three strays., The ffmpeg leftover belongs to episode_13 but names no kind., TestNamingShapes

### Community 59 - "The Host allowlist and DNS-rebinding guard"
Cohesion: 0.15
Nodes (7): The Host allowlist, which is what actually stops DNS rebinding.      A loopbac, ``--host ::1`` must keep working.          ``TrustedHostMiddleware`` derives t, Needed when the UI is reached by a machine name or through a proxy., The operator published the UI under names we cannot enumerate.          ``pyth, Artifact links must keep working from anywhere; only writes are gated., That is curl, not a browser: no browser omits Origin on a POST., TestHostGuard

### Community 60 - "Environment-variable configuration (ADR-007)"
Cohesion: 0.17
Nodes (12): allowed_hosts_from_env(), _config_payload(), host_from_env(), is_loopback_host(), podcast_dir_from_env(), port_from_env(), The configuration this UI is willing to show.      Reads only the names in the, Podcast directory from ``WHYCAST_PODCAST_DIR``, else the repo default. (+4 more)

### Community 61 - "Index queries: episodes and their artifacts"
Cohesion: 0.17
Nodes (12): _artifacts_by_episode(), _episode_dict(), _escape_like(), get_episode(), _iso(), list_episodes(), Return every indexed episode, in scan order, with its artifact summary.      A, Return one episode with its artifacts nested, or None if unknown.      Lookup (+4 more)

### Community 62 - "_safe_file: the only gate between a request and the filesystem"
Cohesion: 0.17
Nodes (4): ``commonpath`` raises across drives; the prefix test must not.          Live c, Direct tests of :func:`webui.app._safe_file`.      This is the only gate betwe, ``podcasts_evil`` must not pass a check meant for ``podcasts``.          A pre, TestSafeFileGuard

### Community 63 - "Model selection, prompts and the ADR index"
Cohesion: 0.21
Nodes (12): ADR-003 Per-task OpenAI model selection, ADR-003: per-task model env-vars met lengte-gebaseerde switch, ADR-005 Recursive chunked summarization for long transcripts, ADR-005: recursive map-reduce summarization met overlap, ADR-007 Environment-variable configuration and file-based prompts, ADR Index, ADR README, bin/adr-index Generator (+4 more)

### Community 64 - "Command-line entry points"
Cohesion: 0.2
Nodes (11): main(), Run the server. Returns a process exit code., ``python -m webui.runner <job_id>``. Returns the process exit code., ``python -m webui.worker``. Returns a process exit code., Main function with command line argument handling., Main function with command line interface., quick_update_all(), Update a specific episode by name. (+3 more)

### Community 65 - "Output formats: markdown, HTML and wiki"
Cohesion: 0.2
Nodes (11): convert_markdown_to_html(), convert_markdown_to_wiki(), Convert markdown text to Wiki markup.          Args:         markdown_text: T, Write markdown text to .txt, .html, and .wiki files in the specified directory., Convert markdown text to HTML.          Args:         markdown_text: The mark, write_all_format(), markdown, convert_markdown_to_html Golden Snapshot (+3 more)

### Community 66 - "Windows entry point: no console, no stdout"
Cohesion: 0.18
Nodes (10): pythonw.exe under Task Scheduler gets sys.stdout is None.      Regression for, Never replace a working stdout: that would swallow console output., test_a_process_that_has_streams_is_left_alone(), test_the_server_survives_having_no_console(), build_parser(), _ensure_output_streams(), Entry point for the WHYcast web UI: ``python -m webui``.  Starts uvicorn on th, Give the process real stdout/stderr when Windows gave it none.      ``pythonw. (+2 more)

### Community 67 - "Chunking and truncation, pinned by golden snapshots"
Cohesion: 0.22
Nodes (10): Split text into chunks of specified maximum size with overlap.          Args:, Truncate a transcript to fit within token limits.          Args:         tran, split_into_chunks(), truncate_transcript(), Speaker Mapping Parse Golden Snapshot, Golden Snapshot Pinning of Pipeline Functions, Overlapping Transcript Chunk Windows, split_into_chunks Golden Snapshot (+2 more)

### Community 68 - "The episodes API and its status matrix"
Cohesion: 0.2
Nodes (4): AC #1: the status matrix has to be right, not merely present., episode_10's blog must not show up under episode_1., ``_`` is a LIKE wildcard; base names are full of them., TestEpisodesApi

### Community 69 - "Chunked request bodies cannot bypass the size cap"
Cohesion: 0.2
Nodes (10): chunked(), An iterable body, which makes httpx send ``Transfer-Encoding: chunked``., The bypass, closed. A form body is capped exactly like a JSON one., The branch that already held, kept honest by the same reader., The consequence that outlived the request: a prompt sent to the paid API., The other consequence: the ``jobs`` table shares a file with the index., test_a_chunked_form_body_cannot_slip_past_the_cap(), test_a_chunked_json_body_cannot_slip_past_the_cap_either() (+2 more)

### Community 70 - "App lifespan: opening the index and refreshing it when stale"
Cohesion: 0.2
Nodes (10): _as_epoch(), _index_is_stale(), _lifespan(), _newest_change(), _public_meta(), Filter index metadata down to :data:`META_KEYS`.      The single point where a, Best-effort read of a stored timestamp as epoch seconds., The newest mtime in ``podcast_dir``: the directory itself and its entries. (+2 more)

### Community 71 - "Built-in fallback pages when a template is missing"
Cohesion: 0.27
Nodes (10): _escape(), _fallback_page(), _job_detail_fallback(), _job_row_html(), _jobs_fallback(), A minimal standalone page, used when a template is not installed., One job as a table row for the fallback dashboard., The built-in queue dashboard. (+2 more)

### Community 72 - "The job catalogue the UI and API share"
Cohesion: 0.2
Nodes (10): _episode_actions(), _job_spec(), job_types_payload(), The catalogue entry for the job that applies a mapping, or None.      Read fro, The job catalogue, as the API and the templates see it.      ``cost`` is the f, The enqueue actions the episode page offers, in :data:`EPISODE_ACTION_TYPES` ord, The single-step re-runs, in the order the pipeline runs them.      Kept apart, One entry of the job catalogue, with its ``cost`` and ``gpu`` flags.      Read (+2 more)

### Community 73 - "Connection and lookup helpers, and their 404s and 503s"
Cohesion: 0.2
Nodes (10): A connection with both schemas on it, closed afterwards., A connection with both schemas and the podcast directory indexed., _conn(), _get_episode_or_404(), _get_job_or_404(), _queue_or_503(), The index connection, after confirming the queue tables exist.      ``ensure_j, Look ``job_id`` up in the queue, or raise 404.      Like ``base_name``, a job (+2 more)

### Community 74 - "Reading a hand-edited mapping file"
Cohesion: 0.22
Nodes (9): load_speaker_mapping(), The saved mapping for one episode, or None when there is none.      The names-, ``{"SPEAKER_00": "Nancy"}`` is what a person types. It must load.      Requiri, What a person copying from the analysis report would paste., Absent" is an ordinary state - it is branch 3, not a problem., test_bracketed_labels_load_as_bare_ones(), test_no_mapping_file_reads_as_none_not_as_an_error(), test_the_malformed_message_shows_the_shape_that_works() (+1 more)

### Community 75 - "Awkward but legal input: wildcards, non-ASCII, zero bytes"
Cohesion: 0.22
Nodes (4): Values that are legal but awkward: wildcards, non-ASCII, empty files., ``_`` is the single-character LIKE wildcard, and base names are full         of, The pipeline wrote an empty file; that is a result, not a gap., TestAwkwardInput

### Community 76 - "transcript_formatter.py: episode header script"
Cohesion: 0.25
Nodes (8): find_episode_transcripts(), Find all episode transcript files that need speaker assignment.          Retur, Extract episode number from basename for sorting., extract_episode_number(), Extract episode number from filename using various patterns.          Args:, format_transcript_with_headers(), Add standard header and footer to the transcript with proper episode information, Extract episode number from the output basename.          Args:         outpu

### Community 77 - "Diarization and the HuggingFace token"
Cohesion: 0.29
Nodes (7): diarize_audio(), get_huggingface_token(), Speaker diarization step for the WHYcast pipeline (ADR-008).  Extracted verbat, Run speaker diarization using pyannote.audio 3.1 pipeline on GPU if available, u, Set the HuggingFace token in the .env file.          Args:         token: The, Get the HuggingFace token from environment. Optionally prompt the user if it's m, set_huggingface_token()

### Community 78 - "Timestamped transcript writing"
Cohesion: 0.25
Nodes (6): format_timestamp(), Write transcript files with and without timestamps, integrating speaker info., Format a timestamp in seconds to HH:MM:SS format.      Args:         start: S, write_transcript_files(), format_timestamp Golden Snapshot, tqdm

### Community 79 - "Every io_utils writer call site, checked by AST"
Cohesion: 0.25
Nodes (8): Every call to an io_utils writer in whycast/, as (file, line, node, func)., Option A's stated risk, guarded.      ADR-009 chose per-call-site opt-out over, The other half of the same rule, for the one file that is not an artifact., Keeps the three writer tests above honest as the package grows.      Read this, test_every_artifact_writer_says_backup_false_out_loud(), test_the_human_input_writer_says_backup_true_out_loud(), test_the_writers_this_suite_covers_are_all_of_them(), _writer_calls()

### Community 80 - "Queue concurrency: eight threads, two processes, one job each"
Cohesion: 0.25
Nodes (8): open_second_connection(), Eight threads, eight connections, one job each - never the same one twice., The real production shape: two processes, two connections, one queue.      The, Another connection to the same file, as a second process would have., ``{job_id: status}`` straight out of the table, bypassing the helpers., statuses(), test_concurrent_threads_each_get_a_different_job(), test_two_processes_each_get_a_different_job()

### Community 81 - "The operator's real input files must pass the editors' validators"
Cohesion: 0.25
Nodes (8): The operator's actual files must be saveable through this UI.      The validat, test_the_real_vocabulary_and_prompts_pass_their_own_validators(), _normalise_newlines(), CRLF and bare CR to LF.      A ``<textarea>`` posts CRLF per the HTML spec, an, Check an edited ``vocabulary.json`` and return ``(mapping, canonical)``., Check an edited prompt and return the text to write.      A prompt is free-for, _validated_prompt(), _validated_vocabulary()

### Community 82 - "The artifact diff, read safely"
Cohesion: 0.25
Nodes (8): _diff_entry(), _job_diff_payload(), Compare a job's before-snapshot with the artifacts on disk now., One artifact's before/after comparison, safe on anything unreadable., Read a text file, or None when it is missing or not text., Return ``candidate`` resolved, if it is a real file inside ``root``.      This, _read_text_or_none(), _safe_file()

### Community 83 - "Editor file state and human-input writes"
Cohesion: 0.29
Nodes (8): _display_path(), _file_state(), _prompt_overview(), ``path`` relative to the repository root, or just its filename.      What the, What the editor knows about one file on disk.      Deliberately contains no ab, Every allowlisted prompt with its file state, for the list page., Write a human-authored input file, keeping the previous version.      ``backup, _write_input_file()

### Community 84 - "Queue schema creation and migration"
Cohesion: 0.29
Nodes (8): ensure_job_schema(), JobQueueError, _migrate(), Refuse to work inside someone else's open transaction.      Not paranoia: the, Raised when the queue is asked to do something it cannot do safely.      Disti, Create or migrate the queue tables. Idempotent; safe on every startup.      Ca, Apply schema steps from ``stored_version`` up to the current one.      Empty a, _require_no_transaction()

### Community 85 - "Vocabulary correction (ADR-006)"
Cohesion: 0.29
Nodes (8): apply_vocabulary_corrections(), load_vocabulary_mappings(), process_transcript_with_vocabulary(), Process a transcript with custom vocabulary corrections.          Args:, Load vocabulary mappings from a JSON file.          Args:         vocab_file:, Apply vocabulary corrections to the transcribed text.          Args:, apply_vocabulary_corrections Golden Snapshot, load_vocabulary_mappings Golden Snapshot

### Community 86 - "Path traversal, symlinks and null bytes"
Cohesion: 0.29
Nodes (4): No request may reach a file outside the podcast directory.      Both halves of, Either the client refuses to send it or the server refuses to serve it., The index is re-checked against the podcast directory before opening., TestPathTraversal

### Community 88 - "Server bind flags versus environment"
Cohesion: 0.29
Nodes (3): AC #5 through ``python -m webui``, not just through a constant.      ``uvicorn, The flags are a convenience over ADR-007 variables, not a second store., TestServerBind

### Community 89 - "Correction workflows and the config viewer (TASK-004 and TASK-005)"
Cohesion: 0.33
Nodes (7): TASK-004 Correcties naar de invoer, nooit naar een artefact, TASK-004 <base>_speakers.json als persistente pipeline-invoer, TASK-005 Read-only config-viewer met allowlist, ADR-004: two-phase speaker mapping (LLM oordeelt, code past toe), ADR-006: vocabulary.json post-hoc replacement map, ADR-007: env-vars + config.py constants + prompts/*.txt, ADR-008: allowlist voor config-weergave (secrets nooit)

### Community 90 - "Artifact response headers and framable 404s"
Cohesion: 0.33
Nodes (6): _artifact_headers(), _artifact_problem(), _own_origin(), Artifact response headers, with this server's origin in the policy., A 404 the artifact frame can actually display.      Raising HTTPException here, The origin this request was addressed to, as a browser would spell it.

### Community 91 - "Picking the right artifact for a kind"
Cohesion: 0.33
Nodes (6): _artifacts_of(), _pick_artifact(), The transcript a speakers re-run would start from, or None.      Deliberately, The artifact rows of an episode dict., Best artifact of ``kind`` for ``episode``, or None.      Mirrors :meth:`whycas, _transcript_artifact()

### Community 92 - "The worker's environment and its logging sink"
Cohesion: 0.33
Nodes (6): output_dir_from_env(), The directory jobs read from and write artifacts to.      ``WHYCAST_PODCAST_DI, LoggingSink, An :class:`whycast.events.EventSink` that forwards to :mod:`logging`.      The, Claim and run jobs, one at a time, until interrupted.      Args:         poll, run_worker()

### Community 93 - "Index schema versioning"
Cohesion: 0.4
Nodes (3): A schema bump must rebuild the index tables, not corrupt the database., The phase-2 job queue will share this file and is NOT rebuildable., TestSchemaVersioning

### Community 94 - "A live TestClient for the SSE stream"
Cohesion: 0.4
Nodes (5): client(), The property that makes this endpoint worth having: live progress.      A real, A TestClient over a throwaway index, queue and job-log directory., test_the_stream_delivers_a_real_job_as_it_runs(), A started TestClient over a freshly built app.      The ``with`` block is what

### Community 97 - "Package docstrings: whycast, webui and the pipeline"
Cohesion: 0.5
Nodes (3): Pipeline step modules for WHYcast (ADR-008)., WHYcast web UI (ADR-008).  The FastAPI layer over the extracted ``whycast`` pi, WHYcast pipeline library (ADR-008).  Importing this package has no side effect

### Community 98 - "Writing <base>_cleaned.txt only when cleanup changed something"
Cohesion: 0.5
Nodes (4): Write ``<base>_cleaned.txt``, the text every later step actually reads.      T, _save_cleaned(), A file called *cleaned* that copies its input claims work nobody did.      ``c, test_no_cleaned_transcript_when_cleanup_changed_nothing()

### Community 99 - "Synthetic podcast directory fixtures"
Cohesion: 0.5
Nodes (4): podcast_dir(), A synthetic podcast directory the index can be built from., A synthetic podcasts directory, plus a secret file outside of it., A synthetic podcast directory: one episode with audio and a transcript.

### Community 100 - "SSE frame formatting"
Cohesion: 0.5
Nodes (4): _job_event_stream(), One SSE frame. ``event_id`` becomes the browser's ``Last-Event-ID``., Yield SSE frames for one job until it reaches a terminal status.      Shape of, _sse_frame()

### Community 101 - "Listing jobs and normalising the status filter"
Cohesion: 0.5
Nodes (4): _clean_statuses(), list_jobs(), Return jobs newest first, optionally filtered by status.      Args:         c, Normalise the ``list_jobs`` filter to a validated list of statuses.

### Community 102 - "The speaker prompts and the WHYcast co-hosts"
Cohesion: 1.0
Nodes (3): Speaker Analysis Prompt (SPEAKER_NN -> name mapping), WHYcast co-hosts Nancy and Ad, Speaker Unknown Attribution Prompt (conversational flow analysis)

### Community 103 - "Scratch queue fixtures"
Cohesion: 0.67
Nodes (3): queue_home(), A scratch database path and job-log directory, wired through the env.      ``W, A scratch queue, job-log directory and podcast directory.      ``WHYCAST_WEBUI

## Ambiguous Edges - Review These
- `split_into_chunks()` → `Prompt Editor Page`  [AMBIGUOUS]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/webui/templates/prompts.html · relation: conceptually_related_to
- `parse_speaker_mapping_from_analysis()` → `Speaker Mapping Editor Page`  [AMBIGUOUS]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/webui/templates/speakers.html · relation: references
- `Speaker Label to Name Mapping Shape` → `Speaker Editor Context (episode, state, apply_job, active_job, episode_jobs)`  [AMBIGUOUS]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/webui/templates/speakers.html · relation: shares_data_with

## Knowledge Gaps
- **852 isolated node(s):** `Find all episode transcript files that need speaker assignment.          Retur`, `Extract episode number from basename for sorting.`, `Batch update speaker assignments for all episodes.          Args:         dir`, `Main function with command line argument handling.`, `Compatibility shim (ADR-008): configuration moved to whycast/config.py.  Exist` (+847 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `split_into_chunks()` and `Prompt Editor Page`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `parse_speaker_mapping_from_analysis()` and `Speaker Mapping Editor Page`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Speaker Label to Name Mapping Shape` and `Speaker Editor Context (episode, state, apply_job, active_job, episode_jobs)`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `ConfigurationError` connect `Episode scanner guarantees on real filenames` to `Web UI HTTP layer tests and fixtures`, `Episode page artifact rendering and its acceptance criteria`, `The whycast pipeline step modules`, `Single-step re-runs of one post-processing step`, `Editor validators: newlines, duplicate keys, rejections`, `Episode scanner internals: suffix peeling and kind ranking`, `Recognising <base>_speakers.json as an input, not an artifact`, `Index freshness and the rescan contract`, `Tests against the operator's real podcasts/ directory`, `Episode scanner tests: prefix collisions`, `Loopback binding and the host allowlist (ADR-008)`, `Artifact naming shapes and duplicate slots`, `The Host allowlist and DNS-rebinding guard`, `Environment-variable configuration (ADR-007)`, `_safe_file: the only gate between a request and the filesystem`, `The episodes API and its status matrix`, `Awkward but legal input: wildcards, non-ASCII, zero bytes`, `Path traversal, symlinks and null bytes`, `Episode detail API: case, spaces, 404s`, `Server bind flags versus environment`, `Index schema versioning`, `The index is disposable and never writes to podcasts/`, `A missing podcast directory is not a crash`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `parse_speaker_mapping_from_analysis()` connect `Discarded partial downloads and dropped .bak files (ADR-009)` to `Web UI templates and their shared macros`, `Speaker mapping application and staleness checks`, `Chunking and truncation, pinned by golden snapshots`?**
  _High betweenness centrality (0.042) - this node is a cross-community bridge._
- **Why does `Speaker Mapping Editor Page` connect `Web UI templates and their shared macros` to `Discarded partial downloads and dropped .bak files (ADR-009)`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `scan_podcasts()` (e.g. with `.test_episode_10_blog_does_not_attach_to_episode_1()` and `.test_episode_1_keeps_its_own_artifacts()`) actually correct?**
  _`scan_podcasts()` has 45 INFERRED edges - model-reasoned connections that need verification._