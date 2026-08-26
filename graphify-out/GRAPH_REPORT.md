# Graph Report - .  (2026-08-26)

## Corpus Check
- 113 files · ~1,929,839 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2155 nodes · 3799 edges · 125 communities (93 shown, 32 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 470 edges (avg confidence: 0.76)
- Token cost: at least 314,711 subagent tokens for the first full extraction (chunks 1 and 2; chunk 3's count was lost to a context compaction), plus one un-metered subagent for this rebuild. The harness reports a single combined figure per subagent rather than an input/output split, so neither half can be stated. Treat 314,711 as a floor, not a total.

## Provenance: what was repaired before this graph was built

Rebuilt after a dead-code sweep removed three symbols and one Jinja macro. Only
`webui/templates/_macros.html` needed re-extraction - the other 53 documents
replayed from cache, and the Python changes are covered by the AST pass, which
costs nothing. The graph lost 3 nodes and 5 edges, which is what the removal
looks like from here.

Four corrections stand between what the tooling produced and what this graph
holds. Each is listed so a later reader can tell them apart.

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

2. **Eight duplicated ADRs, collapsed.** The first extraction ran as three
   parallel subagents, and a subagent can only draw edges inside its own chunk.
   The chunk holding `ADR-INDEX.md` minted its own nodes for ADR-001..008 to give
   its ~40 citation edges a target; the chunk holding the ADR files extracted the
   same eight decisions. Both were right locally, and together they were eight
   decisions represented twice with half the edges each. The per-file node won -
   it is where the decision actually lives. The collapse is now in the cache, so
   this rebuild had nothing left to fix; the check still runs, in case a future
   chunk reintroduces the split.

3. **Three cached edges pointing at ADR concepts that were never extracted.**
   Cache entries for `prompts/*.txt` written by an earlier run guessed what the
   ADR's node would be called, and a later chunk slugged it differently. Repaired
   in the cache itself rather than only in one build, so a later
   `graphify update` cannot quietly reintroduce them.

4. **105 nodes from `webui/static/htmx.min.js`, dropped.** Vendored, minified,
   third-party. Its mangled names (`He`, `e`, `F`) carried no meaning, and two of
   them outranked real functions in the god-node list - the graph reporting on its
   own noise.

295 edges remain unconnected on purpose: they are imports of stdlib and
third-party packages (`os`, `logging`, `pytest`, `fastapi`), which have no node
because those libraries are not part of this corpus.

### One thing this graph is not good for

Its `calls` edges resolve by name, not by module. `webui.jobs.enqueue` carries no
incoming call edge at all, while the unrelated `enqueue` helper in
`tests/test_jobs_api.py` collects fifteen. Do not use this graph to decide whether
a symbol is dead - it will tell you `create_app()` and `init_db()` are unreachable.
The dead-code sweep that preceded this rebuild used a separate name-resolving pass
over the Python AST, plus a template scan for Jinja macros, for exactly that reason.

## Graph Freshness
- Built from commit: `75bf8c4b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Web UI templates and their shared macros|Web UI templates and their shared macros]]
- [[_COMMUNITY_Atomic artifact writes and their failure-mode tests|Atomic artifact writes and their failure-mode tests]]
- [[_COMMUNITY_RSS feed fetching and download verification|RSS feed fetching and download verification]]
- [[_COMMUNITY_End-to-end worker and runner tests|End-to-end worker and runner tests]]
- [[_COMMUNITY_Web UI HTTP layer tests and fixtures|Web UI HTTP layer tests and fixtures]]
- [[_COMMUNITY_The whycast pipeline step modules|The whycast pipeline step modules]]
- [[_COMMUNITY_Job HTTP API tests|Job HTTP API tests]]
- [[_COMMUNITY_Serial worker process control and tree-kill|Serial worker process control and tree-kill]]
- [[_COMMUNITY_Speaker mapping precedence and transcript fingerprinting|Speaker mapping precedence and transcript fingerprinting]]
- [[_COMMUNITY_The speaker mapping as pipeline input (ADR-010)|The speaker mapping as pipeline input (ADR-010)]]
- [[_COMMUNITY_Awkward but legal input wildcards, non-ASCII, zero bytes|Awkward but legal input: wildcards, non-ASCII, zero bytes]]
- [[_COMMUNITY_Phase-2 regression tests|Phase-2 regression tests]]
- [[_COMMUNITY_Episode page artifact rendering and its acceptance criteria|Episode page artifact rendering and its acceptance criteria]]
- [[_COMMUNITY_Job queue unit tests|Job queue unit tests]]
- [[_COMMUNITY_LLM steps model choice, chunking and the API key|LLM steps: model choice, chunking and the API key]]
- [[_COMMUNITY_Episode scanner guarantees on real filenames|Episode scanner guarantees on real filenames]]
- [[_COMMUNITY_The Backlog plan doc-001 phases and TASK-001..005|The Backlog plan: doc-001 phases and TASK-001..005]]
- [[_COMMUNITY_Single-step re-runs of one post-processing step|Single-step re-runs of one post-processing step]]
- [[_COMMUNITY_Backup call-site tests and the cleaned transcript (ADR-009)|Backup call-site tests and the cleaned transcript (ADR-009)]]
- [[_COMMUNITY_The job runner and its per-job-type handlers|The job runner and its per-job-type handlers]]
- [[_COMMUNITY_Golden-master tests for the monolith split|Golden-master tests for the monolith split]]
- [[_COMMUNITY_FastAPI app internals request state and SSE plumbing|FastAPI app internals: request state and SSE plumbing]]
- [[_COMMUNITY_GPU setup, audio preparation and diarization|GPU setup, audio preparation and diarization]]
- [[_COMMUNITY_Episode scanner internals suffix peeling and kind ranking|Episode scanner internals: suffix peeling and kind ranking]]
- [[_COMMUNITY_The three human-input editors as an HTTP contract|The three human-input editors as an HTTP contract]]
- [[_COMMUNITY_Prompt editor tests|Prompt editor tests]]
- [[_COMMUNITY_Recognising base_speakers.json as an input, not an artifact|Recognising <base>_speakers.json as an input, not an artifact]]
- [[_COMMUNITY_SSE progress stream tests|SSE progress stream tests]]
- [[_COMMUNITY_Speaker mapping editor tests|Speaker mapping editor tests]]
- [[_COMMUNITY_Speaker analysis and programmatic assignment|Speaker analysis and programmatic assignment]]
- [[_COMMUNITY_The SQLite job queue|The SQLite job queue]]
- [[_COMMUNITY_Beforeafter diff and the read-only config viewer|Before/after diff and the read-only config viewer]]
- [[_COMMUNITY_The rebuildable SQLite index over podcasts|The rebuildable SQLite index over podcasts/]]
- [[_COMMUNITY_The web UI's browser-side JavaScript|The web UI's browser-side JavaScript]]
- [[_COMMUNITY_Runner verdicts finish, fail, cancel, restore|Runner verdicts: finish, fail, cancel, restore]]
- [[_COMMUNITY_Job context, cancellation and the free self-test|Job context, cancellation and the free self-test]]
- [[_COMMUNITY_Chunked summarization and mapping precedence decisions|Chunked summarization and mapping precedence decisions]]
- [[_COMMUNITY_Versioned backups moved aside before a run (ADR-011)|Versioned backups moved aside before a run (ADR-011)]]
- [[_COMMUNITY_Reading and validating a saved speaker mapping|Reading and validating a saved speaker mapping]]
- [[_COMMUNITY_Checking a mapping covers every label in the transcript|Checking a mapping covers every label in the transcript]]
- [[_COMMUNITY_Before-snapshots and hostile-name safety|Before-snapshots and hostile-name safety]]
- [[_COMMUNITY_Job execution event sink, lock file, exit codes|Job execution: event sink, lock file, exit codes]]
- [[_COMMUNITY_Index freshness and the artifact-serving fixtures|Index freshness and the artifact-serving fixtures]]
- [[_COMMUNITY_CUDA runtime regression tests (ADR-001)|CUDA runtime regression tests (ADR-001)]]
- [[_COMMUNITY_find_episodes.py rename-collision analysis script|find_episodes.py: rename-collision analysis script]]
- [[_COMMUNITY_Discarded partial downloads and dropped .bak files (ADR-009)|Discarded partial downloads and dropped .bak files (ADR-009)]]
- [[_COMMUNITY_Tests against the operator's real podcasts directory|Tests against the operator's real podcasts/ directory]]
- [[_COMMUNITY_Job log directories and parked verdicts|Job log directories and parked verdicts]]
- [[_COMMUNITY_Transcription and diarization decisions (ADR-001, ADR-002)|Transcription and diarization decisions (ADR-001, ADR-002)]]
- [[_COMMUNITY_Episode scanner tests prefix collisions|Episode scanner tests: prefix collisions]]
- [[_COMMUNITY_Running the queue to completion with a real worker|Running the queue to completion with a real worker]]
- [[_COMMUNITY_Application assembly and route registration|Application assembly and route registration]]
- [[_COMMUNITY_Request body parsing JSON and form, with a size cap|Request body parsing: JSON and form, with a size cap]]
- [[_COMMUNITY_The speaker editor's view of disk state|The speaker editor's view of disk state]]
- [[_COMMUNITY_Editor validators and the operator's own files|Editor validators and the operator's own files]]
- [[_COMMUNITY_The event sink contract JSONL lines and logging|The event sink contract: JSONL lines and logging]]
- [[_COMMUNITY_Artifact naming shapes and duplicate slots|Artifact naming shapes and duplicate slots]]
- [[_COMMUNITY_Backlog tasks scanner, temp sweep, correction workflows|Backlog tasks: scanner, temp sweep, correction workflows]]
- [[_COMMUNITY_Vocabulary correction (ADR-006)|Vocabulary correction (ADR-006)]]
- [[_COMMUNITY_Vocabulary editor validation and its 400s|Vocabulary editor validation and its 400s]]
- [[_COMMUNITY_Environment-variable configuration (ADR-007)|Environment-variable configuration (ADR-007)]]
- [[_COMMUNITY_Index queries episodes and their artifacts|Index queries: episodes and their artifacts]]
- [[_COMMUNITY__safe_file the only gate between a request and the filesystem|_safe_file: the only gate between a request and the filesystem]]
- [[_COMMUNITY_Command-line entry points|Command-line entry points]]
- [[_COMMUNITY_Queue robustness terminal jobs, foreign schemas, two runners|Queue robustness: terminal jobs, foreign schemas, two runners]]
- [[_COMMUNITY_Windows entry point no console, no stdout|Windows entry point: no console, no stdout]]
- [[_COMMUNITY_Per-job tables of the kinds a step reads and writes|Per-job tables of the kinds a step reads and writes]]
- [[_COMMUNITY_Chunked request bodies cannot bypass the size cap|Chunked request bodies cannot bypass the size cap]]
- [[_COMMUNITY_Connection and lookup helpers, and their 404s and 503s|Connection and lookup helpers, and their 404s and 503s]]
- [[_COMMUNITY_The job catalogue the UI and API share|The job catalogue the UI and API share]]
- [[_COMMUNITY_Built-in fallback pages when a template is missing|Built-in fallback pages when a template is missing]]
- [[_COMMUNITY_App lifespan opening the index and refreshing it when stale|App lifespan: opening the index and refreshing it when stale]]
- [[_COMMUNITY_Queue schema creation and migration|Queue schema creation and migration]]
- [[_COMMUNITY_transcript_formatter.py episode header script|transcript_formatter.py: episode header script]]
- [[_COMMUNITY_Model selection and file-based prompts (ADR-003, ADR-007)|Model selection and file-based prompts (ADR-003, ADR-007)]]
- [[_COMMUNITY_Queue concurrency eight threads, two processes, one job each|Queue concurrency: eight threads, two processes, one job each]]
- [[_COMMUNITY_The artifact diff, read safely|The artifact diff, read safely]]
- [[_COMMUNITY_Editor file state and human-input writes|Editor file state and human-input writes]]
- [[_COMMUNITY_Output formats markdown, HTML and wiki|Output formats: markdown, HTML and wiki]]
- [[_COMMUNITY_Server bind flags versus environment|Server bind flags versus environment]]
- [[_COMMUNITY_Path traversal, symlinks and null bytes|Path traversal, symlinks and null bytes]]
- [[_COMMUNITY_Episode detail API case, spaces, 404s|Episode detail API: case, spaces, 404s]]
- [[_COMMUNITY_Rescan idempotence and the metadata allowlist|Rescan idempotence and the metadata allowlist]]
- [[_COMMUNITY_Artifact response headers and framable 404s|Artifact response headers and framable 404s]]
- [[_COMMUNITY_Picking the right artifact for a kind|Picking the right artifact for a kind]]
- [[_COMMUNITY_Feed jobs fetch latest, fetch all|Feed jobs: fetch latest, fetch all]]
- [[_COMMUNITY_A live TestClient for the SSE stream|A live TestClient for the SSE stream]]
- [[_COMMUNITY_Index schema versioning|Index schema versioning]]
- [[_COMMUNITY_The index is disposable and never writes to podcasts|The index is disposable and never writes to podcasts/]]
- [[_COMMUNITY_A missing podcast directory is not a crash|A missing podcast directory is not a crash]]
- [[_COMMUNITY_Finishing a job and trimming its error|Finishing a job and trimming its error]]
- [[_COMMUNITY_Synthetic podcast directory fixtures|Synthetic podcast directory fixtures]]
- [[_COMMUNITY_SSE frame formatting|SSE frame formatting]]
- [[_COMMUNITY_The speaker prompts and the WHYcast co-hosts|The speaker prompts and the WHYcast co-hosts]]
- [[_COMMUNITY_Editable-input fixtures redirected to tmp_path|Editable-input fixtures redirected to tmp_path]]
- [[_COMMUNITY_A real uvicorn on a real socket|A real uvicorn on a real socket]]
- [[_COMMUNITY_Dumping the index tables in scan order|Dumping the index tables in scan order]]
- [[_COMMUNITY_Scratch queue fixtures|Scratch queue fixtures]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 102|Community 102]]
- [[_COMMUNITY_Community 103|Community 103]]
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
- [[_COMMUNITY_Community 118|Community 118]]
- [[_COMMUNITY_Community 119|Community 119]]
- [[_COMMUNITY_Community 120|Community 120]]
- [[_COMMUNITY_Community 121|Community 121]]
- [[_COMMUNITY_Community 122|Community 122]]
- [[_COMMUNITY_Community 123|Community 123]]
- [[_COMMUNITY_Community 124|Community 124]]

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
- `Prompt Editor Page` --conceptually_related_to--> `split_into_chunks()`  [AMBIGUOUS]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/webui/templates/prompts.html → whycast/pipeline/llm.py
- `Speaker Mapping Editor Page` --references--> `parse_speaker_mapping_from_analysis()`  [AMBIGUOUS]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/webui/templates/speakers.html → whycast/pipeline/speakers.py
- `Audio Downloads Are Not Backed Up` --conceptually_related_to--> `atomic_writer()`  [INFERRED]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/docs/adr/ADR-011-versioned-artifact-backups-moved-aside-before-every-run.md → whycast/io_utils.py
- `estimate_token_count()` --conceptually_related_to--> `ADR-005 Recursive chunked summarization for long transcripts`  [INFERRED]
  whycast/pipeline/llm.py → D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/docs/adr/ADR-005-recursive-chunked-summarization-for-long-transcripts.md
- `Overlapping Transcript Chunk Windows` --rationale_for--> `split_into_chunks()`  [INFERRED]
  D:/Users/Robert/Documents/GitHub/RvdB/WHYcast-transcribe/tests/golden/split_into_chunks.snapshot.txt → whycast/pipeline/llm.py

## Hyperedges (group relationships)
- **Fase 0: monoliet naar importeerbare library met golden-master gate** — backlog_docs_doc_001___webui_plan_whycast_transcribe_fase_0_pipeline_extractie, backlog_tasks_task_001___extract_whycast_pipeline_library_uit_transcribe_py_webui_fase_0_task_001, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_whycast_pipeline_library, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_eventsink_progress_protocol, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_transcribe_py_cli_shim, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_golden_master_test [EXTRACTED 1.00]
- **Menselijke correcties leven in bewerkbare pipeline-invoerbestanden** — docs_adr_adr_006_post_transcription_vocabulary_correction_via_vocabulary_json_vocabulary_json_replacement_map, docs_adr_adr_007_environment_variable_configuration_and_file_based_prompts_env_vars_and_file_based_prompts, backlog_tasks_task_004___webui_fase_3_correctie_workflows_sprekers_vocabulary_prompts_speakers_json_pipeline_input, backlog_tasks_task_004___webui_fase_3_correctie_workflows_sprekers_vocabulary_prompts_input_first_correctie_principe [INFERRED 0.85]
- **Episode Processing Flow (audio to blog)** — docs_pipeline_prepare_audio, docs_pipeline_diarize, docs_pipeline_transcribe, docs_pipeline_merge_speaker_lines_step, docs_pipeline_speaker_assignment, docs_pipeline_cleanup, docs_pipeline_summary, docs_pipeline_blog [EXTRACTED 1.00]
- **Eén GPU dwingt seriële job-uitvoering met crash-isolatie** — docs_adr_adr_001_use_faster_whisper_on_cuda_for_transcription_faster_whisper_on_cuda, docs_adr_adr_002_speaker_diarization_via_pyannote_audio_3_1_pyannote_speaker_diarization_pipeline, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_serial_worker_subprocess, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_sqlite_index_and_queue, docs_adr_adr_008_web_ui_as_fastapi_layer_over_extracted_pipeline_library_sse_jsonl_progress, backlog_tasks_task_003___webui_fase_2_job_queue_worker_en_live_voortgang_seriele_worker_met_lockfile [EXTRACTED 1.00]
- **Speaker Mapping Correction Loop** — docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_speakers_json, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_speaker_assignment_step, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_analyze_speakers_with_o4, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_mapping_file_precedence, tests_golden_apply_speaker_mapping_programmatically_snapshot_apply_speaker_mapping_programmatically, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_stale_mapping_flag [INFERRED 0.85]
- **Artifact Durability Regime (inputs preserved, outputs versioned)** — docs_adr_adr_009_discard_partial_downloads_and_drop_per_artifact_backups_adr_009, docs_adr_adr_010_persisted_editable_speaker_mapping_as_pipeline_input_adr_010, docs_adr_adr_011_versioned_artifact_backups_moved_aside_before_every_run_adr_011, docs_adr_adr_011_versioned_artifact_backups_moved_aside_before_every_run_backups_directory, docs_adr_adr_009_discard_partial_downloads_and_drop_per_artifact_backups_human_input_never_in_generated_artifacts, docs_pipeline_input_output_rule [INFERRED 0.85]
- **Untrusted Pipeline Output Never Enters the Live DOM** — webui_templates_base_page_skeleton, webui_templates__macros_never_safe, webui_templates_episode_artifact_tabs, webui_templates_job_detail_sse_log_replay, webui_templates_job_diff_diff_page, webui_templates_base_htmx_is_not_a_sanitiser [INFERRED 0.85]
- **Two Gates Before Any Paid API Call** — webui_templates_episode_episode_detail, webui_templates_jobs_job_dashboard, webui_templates_speakers_editor_page, webui_templates_vocabulary_editor_page, webui_templates_prompts_editor_page, webui_templates__macros_cost_badge, webui_templates_episode_two_gates [INFERRED 0.85]
- **Human-Written Pipeline Inputs Keep a .bak** — webui_templates_vocabulary_editor_page, webui_templates_prompts_editor_page, webui_templates_speakers_editor_page, webui_templates_vocabulary_input_gets_backup, webui_templates_job_diff_before_snapshot [INFERRED 0.85]
- **Absent Data Renders as a Muted Dash** — webui_templates__macros_size, webui_templates__macros_epoch, webui_templates__macros_duration, webui_templates__macros_elapsed, webui_templates__macros_episode_link [INFERRED 0.85]
- **The Widgets That Together Render One Job Row** — webui_templates__macros_job_link, webui_templates__macros_episode_link, webui_templates__macros_status_badge, webui_templates__macros_elapsed, webui_templates__macros_cost_badge, webui_templates__macros_gpu_badge, webui_templates__macros_cancel_form [INFERRED 0.85]

## Communities (125 total, 32 thin omitted)

### Community 0 - "Web UI templates and their shared macros"
Cohesion: 0.05
Nodes (95): Layered Speaker Mapping Parse Fallbacks, Speaker Label to Name Mapping Shape, abbrev() Macro, Artifact Row Shape (detail page only), cancel_form() Macro, Cancel Is a Plain POST Form That Works Without JavaScript, cell() Macro, cost_badge() Macro (+87 more)

### Community 1 - "Atomic artifact writes and their failure-mode tests"
Cohesion: 0.05
Nodes (78): Atomic Artifact Write (temp file then os.replace), whycast.io_utils, Audio Downloads Are Not Backed Up, get_huggingface_token(), Set the HuggingFace token in the .env file.          Args:         token: The, Get the HuggingFace token from environment. Optionally prompt the user if it's m, set_huggingface_token(), leftovers() (+70 more)

### Community 2 - "RSS feed fetching and download verification"
Cohesion: 0.05
Nodes (71): BaseHTTPRequestHandler, delete_episode_files(), download_all_episodes_from_rssfeed(), _download_audio(), _enclosure_length(), _header_length(), podcast_fetching_workflow(), Size the RSS enclosure claims, in bytes, or None when it does not say.      Pa (+63 more)

### Community 3 - "End-to-end worker and runner tests"
Cohesion: 0.05
Nodes (57): wait_until(), assert_queue_is_free(), enqueue_selftest(), peek_events(), End-to-end tests for :mod:`webui.worker` and :mod:`webui.runner` (ADR-008, TASK, Refuse to start a worker while the queue holds a job that costs money.      Th, One claim-run-exit cycle, after the cost guard has had its say., Start ``run_worker(once=True)`` in a thread; returns ``(thread, box)``.      ` (+49 more)

### Community 4 - "Web UI HTTP layer tests and fixtures"
Cohesion: 0.04
Nodes (35): configured_env(), Tests for the WHYcast web UI HTTP layer (ADR-008 / TASK-002 phase 1).  Everyth, Every format in the scanner's vocabulary must be servable., Point the ADR-007 environment variables at the temporary directories.      ``c, TestArtifactFormats, TestHealth, Tests for the WHYcast web UI job API and SSE stream (ADR-008 / TASK-003).  The, Every type says whether it costs money and whether it claims the GPU. (+27 more)

### Community 5 - "The whycast pipeline step modules"
Cohesion: 0.07
Nodes (35): Audio preparation step for the WHYcast pipeline (ADR-008).  Extracted verbatim, Speaker diarization step for the WHYcast pipeline (ADR-008).  Extracted verbat, RSS feed fetching and episode download for the WHYcast pipeline (ADR-008).  Fu, enable_tf32(), get_default_device(), is_cuda_available(), GPU setup and utilization helpers for the WHYcast pipeline (ADR-008).  Extract, Enable TensorFloat-32 on Ampere-or-newer GPUs (moved from module level, ADR-008) (+27 more)

### Community 6 - "Job HTTP API tests"
Cohesion: 0.05
Nodes (40): job_env(), Tests for the job HTTP API of the WHYcast web UI (ADR-008, TASK-003 phase 2)., Create a job through the API; fail loudly if it was not accepted., The request is wrong, not the URL: nothing about /api/jobs is missing., base_name is an index key. A path is simply a name the index does not have., The lookup is an equality test, not a pattern match.      Worth pinning: ``web, The page has to work with scripting off, which means a plain form post., The status becomes cancelled once the process has actually stopped. (+32 more)

### Community 7 - "Serial worker process control and tree-kill"
Cohesion: 0.06
Nodes (49): _apply_pending_verdict(), _cancel_wanted(), is_runner_for_job(), _kernel32(), _kill_hint(), kill_process_tree(), lock_path(), _log_tail() (+41 more)

### Community 8 - "Speaker mapping precedence and transcript fingerprinting"
Cohesion: 0.07
Nodes (47): fingerprint_transcript(), SHA-256 of the transcript a mapping was derived from.      Cheap and determini, Write ``<base>_speakers.json`` atomically, keeping a backup.      Args:, save_speaker_mapping(), ADR-010's precedence rule, checked where it actually costs money.  ``<base>_sp, ADR-010's first Confirmation: mapped transcript, no OpenAI call., Somebody who cannot see their correction being used will not trust it.      Th, Branch 1 needs no model, so no API key should be required to honour it.      R (+39 more)

### Community 9 - "The speaker mapping as pipeline input (ADR-010)"
Cohesion: 0.06
Nodes (47): Pipeline step modules for WHYcast (ADR-008)., Path of the persisted mapping for one episode.      Args:         output_base, speaker_map_path(), test_the_path_is_the_one_the_scanner_and_the_editor_agree_on(), The speaker mapping as a pipeline input (ADR-010, TASK-004 phase 3).  ``tests/, Write a mapping file exactly as given - a hand edit, not a save., Pins the outcome the removed gpt-4o call did not affect.      Inside :func:`sp, The decision procedure itself, on a transcript that still has labels.      Sam (+39 more)

### Community 10 - "Awkward but legal input: wildcards, non-ASCII, zero bytes"
Cohesion: 0.04
Nodes (19): Values that are legal but awkward: wildcards, non-ASCII, empty files., ``_`` is the single-character LIKE wildcard, and base names are full         of, The pipeline wrote an empty file; that is a result, not a gap., The Host allowlist, which is what actually stops DNS rebinding.      A loopbac, ``--host ::1`` must keep working.          ``TrustedHostMiddleware`` derives t, Needed when the UI is reached by a machine name or through a proxy., The operator published the UI under names we cannot enumerate.          ``pyth, Artifact links must keep working from anywhere; only writes are gated. (+11 more)

### Community 11 - "Phase-2 regression tests"
Cohesion: 0.06
Nodes (26): queue(), Regression tests for the phase-2 review findings (ADR-008, TASK-003).  One cla, Force a job row into ``running`` with a given pid, bypassing the worker., It was ``<WHYCAST_JOBS_DIR>/worker.lock`` while the queue is     ``WHYCAST_WEBU, ``claim_next`` published ``running`` with ``pid`` still NULL, and the     reape, Belt and braces for rows written by an older build., ``_settle`` called ``finish`` once and swallowed ``sqlite3.Error``. One     tra, Otherwise the rescue channel rescues the status and loses the reason. (+18 more)

### Community 12 - "Episode page artifact rendering and its acceptance criteria"
Cohesion: 0.04
Nodes (15): AC #1: the matrix must be *right*, not merely rendered.          The overview, AC #2: every artifact the episode has is reachable from the page., AC #2: the mp3 must be playable from the page., Artifacts are prompt-injectable LLM output; they load sandboxed.          A pr, AC #3: junk is visible, not silently dropped., ``list_unmatched`` yields dicts; iterating them as strings printed         the, Artifacts are LLM output: never trusted, never merged into a page., ``_speaker_assignment`` vs ``_ts_speaker_assignment``: serve the new one. (+7 more)

### Community 14 - "LLM steps: model choice, chunking and the API key"
Cohesion: 0.08
Nodes (41): choose_appropriate_model(), ensure_api_key(), estimate_token_count(), process_large_text_in_chunks(), process_with_openai(), Choose the appropriate model based on transcript length.          Args:, Split text into chunks of specified maximum size with overlap.          Args:, Handle very large transcripts by chunking and recursive summarization. (+33 more)

### Community 15 - "Episode scanner guarantees on real filenames"
Cohesion: 0.09
Nodes (18): ``number`` is not a key. ``base_name`` is., The module's headline guarantee: nothing in the directory vanishes., The scanner reads directory metadata and nothing else (ADR-008)., NTFS holds ``Episode_28.mp3`` next to ``episode_28_summary.txt``.      Treatin, The pipeline can legitimately write an empty file., ``episode32.m4a`` next to ``episode32.mp3`` is live on disk., TestCaseInsensitivity, TestEpisodeNumbers (+10 more)

### Community 16 - "The Backlog plan: doc-001 phases and TASK-001..005"
Cohesion: 0.12
Nodes (34): WHYcast-transcribe Backlog Project Config, doc-001 Fase 0: Pipeline-extractie, doc-001 Fase 1: Read-only episode-browser, doc-001 Fase 2: Jobs starten, volgen, annuleren, doc-001 Fase 3: Correctie-workflows, doc-001 Fase 4: Afwerking (optioneel), doc-001 WebUI-plan WHYcast-transcribe, TASK-001 Multi-agent extractie met adversariële review (+26 more)

### Community 17 - "Single-step re-runs of one post-processing step"
Cohesion: 0.08
Nodes (32): The one episode with this base name, or a readable failure., _Ctx, Single-step re-runs: one post-processing step at a time.  Editing one prompt a, summary not found" leaves the reader to work out the next move., An empty file is not an input; sending it to a model wastes the call., The path comes from the index, and is still re-checked against the root., Running the blog step must not regenerate the summary as a side effect., ``import webui.runner`` must work without webui.jobs being loaded first. (+24 more)

### Community 18 - "Backup call-site tests and the cleaned transcript (ADR-009)"
Cohesion: 0.08
Nodes (32): Write ``<base>_cleaned.txt``, the text every later step actually reads.      T, _save_cleaned(), ADR-009's backup decision, pinned at the call sites instead of only at the prim, No backup does not mean no write: the overwrite still has to happen., The control for every test above. If this fails they prove nothing., ADR-009 Must Not: do not change the io_utils default to backup=False., Every call to an io_utils writer in whycast/, as (file, line, node, func)., Option A's stated risk, guarded.      ADR-009 chose per-call-site opt-out over (+24 more)

### Community 19 - "The job runner and its per-job-type handlers"
Cohesion: 0.1
Nodes (31): _float_param(), _job_force_episode(), _job_full_episode(), _job_postprocess(), _job_retranscribe(), _job_speakers(), JobInputError, _make_step_handler() (+23 more)

### Community 20 - "Golden-master tests for the monolith split"
Cohesion: 0.11
Nodes (28): check_golden(), fake_process_with_openai(), new_func(), _NoNetworkOpenAI, _normalize(), old_mod(), Golden extraction tests for the ADR-008 monolith -> whycast package split.  Tw, The legacy transcribe monolith, or None if it can no longer be imported. (+20 more)

### Community 21 - "FastAPI app internals: request state and SSE plumbing"
Cohesion: 0.07
Nodes (29): _apply_targets(), _audio_of(), _episode_jobs(), _last_event_id(), _prompt_spec_or_404(), _public_speaker_state(), FastAPI application for the WHYcast web UI (ADR-008, TASK-002/TASK-003).  Read, The API view of :func:`_speaker_state`: no absolute paths.      ``map_path`` i (+21 more)

### Community 22 - "GPU setup, audio preparation and diarization"
Cohesion: 0.07
Nodes (26): prepare_audio_for_diarization(), Ensures the audio file is in a compatible mono 16kHz mp3 format for diarization/, diarize_audio(), Run speaker diarization using pyannote.audio 3.1 pipeline on GPU if available, u, force_cuda_device(), maximize_gpu_utilization(), Configure optimal GPU settings for Whisper inference.     Focus on proper batch, Aggressively try to select CUDA device for maximum GPU utilization.          T (+18 more)

### Community 23 - "Episode scanner internals: suffix peeling and kind ranking"
Cohesion: 0.1
Nodes (25): test_parse_episode_number(), Artifact, belongs_to_base(), _derive_base(), _fmt_rank(), _kind_of_rest(), _kind_rank(), _matching_base() (+17 more)

### Community 24 - "The three human-input editors as an HTTP contract"
Cohesion: 0.08
Nodes (25): map_path(), The three human-input editors, as an HTTP contract (TASK-004 phase 3).  ``test, ``parse_qsl`` drops blanks by default; the editors need them kept.      A blan, The prompt editor's whole path-safety story: a key, never a path., Every name the API offers resolves; nothing else does., Saved first, so this cannot pass by the name having been rejected., The vocabulary editor renders the file into a textarea., The validators quote what was submitted; that is user input too. (+17 more)

### Community 25 - "Prompt editor tests"
Cohesion: 0.09
Nodes (19): prompt_path(), A prompt is free-form prose, so it is the easiest place to hide markup., test_a_prompt_keeps_one_backup(), test_prompt_content_is_escaped_in_the_page(), Tests for the vocabulary and prompt editors (ADR-008 phase 3, TASK-004).  Thes, A browser posts CRLF; the file must not collect a \\r per save.      ``atomic_, The classic attempt, spelled several ways, against a real target file.      ``, An editable file is named relative to the repository, or by filename only. (+11 more)

### Community 26 - "Recognising <base>_speakers.json as an input, not an artifact"
Cohesion: 0.09
Nodes (16): _names(), ``<base>_speakers.json`` is a pipeline *input* (ADR-010).      A person edits, The web UI needs to offer an input for editing, not for download., ``episode_1_summary.json`` names a real base and a real kind.          It is s, ``random_summary.json`` must not mint an episode ``random``.          The mint, ``episode_70_speakers.json`` has no audio and no results behind it., Minting nothing must not mean detaching: ``_ts`` mints, json joins., The suffix alone does not make it one: the mapping is JSON. (+8 more)

### Community 27 - "SSE progress stream tests"
Cohesion: 0.09
Nodes (27): parse_sse(), Split an SSE body into ``(event_name, event_id, payload)`` triples., A queued job has written nothing. That is a state, not a failure., A job id becomes a path in exactly one place, and that place checks it.      T, test_the_stream_cannot_be_pointed_outside_the_jobs_directory(), test_the_stream_of_a_job_with_no_log_yet_is_not_an_error(), event_record(), Append JSONL event lines the way :class:`webui.runner.JsonlEventSink` does. (+19 more)

### Community 28 - "Speaker mapping editor tests"
Cohesion: 0.13
Nodes (23): map_file(), Tests for the speaker mapping editor in the web UI (ADR-010, TASK-004).  Every, ADR-009: human input keeps a .bak. Generated artifacts do not., The assertion this whole feature stands on.      If the editor hashes a differ, Both body shapes fail on the same input, not one silently ignoring it., A name is text. It is never markup, on any page that shows it., Saving is free. It must not start the paid job on the user's behalf., The API reports file names, never absolute paths (see _public_speaker_state). (+15 more)

### Community 29 - "Speaker analysis and programmatic assignment"
Cohesion: 0.1
Nodes (26): whycast.pipeline.feed, <base>_speaker_analysis.txt Report, Unmapped-Label Warning, analyze_speakers_with_o4(), _announce_missing_mapping(), _announce_saved_mapping(), apply_speaker_mapping_programmatically(), attribute_unknown_speakers() (+18 more)

### Community 30 - "The SQLite job queue"
Cohesion: 0.11
Nodes (25): active_job(), cancel_requested(), claim_next(), _claim_next_sql(), _clean_statuses(), get_job(), _job_dict(), list_jobs() (+17 more)

### Community 31 - "Before/after diff and the read-only config viewer"
Cohesion: 0.1
Nodes (16): env(), _job_with_snapshot(), Phase 4 (TASK-005): the before-snapshot diff and the configuration viewer.  Tw, Create a job row, snapshot `before`, then leave `after` on disk., The allowlist must hold even when the environment is full of secrets., A denylist would fail open here; the allowlist is what makes this pass.      S, A web UI pointed at throwaway directories, never the real podcasts/., A 40 MB diff helps nobody, and the job log is not the place to grow one. (+8 more)

### Community 32 - "The rebuildable SQLite index over podcasts/"
Cohesion: 0.12
Nodes (23): _as_float(), _as_int(), _database_path(), _ensure_schema(), get_meta(), IndexConsistencyError, init_db(), _insert_episodes() (+15 more)

### Community 33 - "The web UI's browser-side JavaScript"
Cohesion: 0.21
Nodes (22): applyJobStatus(), describeError(), formatDuration(), handleCancel(), handleEnqueue(), handleSpeakerDiscard(), handleSpeakerSave(), note() (+14 more)

### Community 34 - "Runner verdicts: finish, fail, cancel, restore"
Cohesion: 0.12
Nodes (23): Protocol, _duration(), _finish_cancelled(), _finish_failed(), Put back anything the run moved aside but never replaced.      Moving an artif, Record a cooperative stop. Never 'failed' - a cancel is not a failure., Record a failure: one sentence in the database, the stack in the log., Write the runner's own verdict, tolerating a database that is gone.      :func (+15 more)

### Community 35 - "Job context, cancellation and the free self-test"
Cohesion: 0.13
Nodes (20): _Context, _job_selftest(), JobCancelled, Raised inside the runner when a cancel request is seen between steps., Everything a job handler needs, resolved once and validated once., Stop the job if a cancel has been requested. Called between steps.          As, Sleep in slices, checking for a cancel between them., Emit progress events for a few seconds, then succeed (or fail on demand). (+12 more)

### Community 36 - "Chunked summarization and mapping precedence decisions"
Cohesion: 0.19
Nodes (21): ADR-005 Recursive chunked summarization for long transcripts, ADR-005: recursive map-reduce summarization met overlap, Mapping File Precedence Over Paid Analysis, Restore Anything Moved Aside but Not Rewritten, 4e. Alternative Blog, 4d. Blog, 4b. Cleanup, 4f. History Extraction (+13 more)

### Community 37 - "Versioned backups moved aside before a run (ADR-011)"
Cohesion: 0.15
Nodes (20): Copy, Not Move, for Artifacts a Job Both Reads and Writes, tests/test_artifact_backups.py, whycast.backups, backup_dir_for(), backup_root(), _base_from_filename(), copy_to_backup(), move_artifacts() (+12 more)

### Community 38 - "Reading and validating a saved speaker mapping"
Cohesion: 0.12
Nodes (20): _clean_mapping(), load_speaker_map_record(), load_speaker_mapping(), _malformed_mapping(), mapping_is_stale(), Was this mapping made for a different transcript than the one at hand?      Ar, Read ``<base>_speakers.json``, whole, as a normalised record.      The record, The saved mapping for one episode, or None when there is none.      The names- (+12 more)

### Community 39 - "Checking a mapping covers every label in the transcript"
Cohesion: 0.13
Nodes (18): _check_mapping_covers_transcript(), The numbered speaker labels a transcript actually contains, bare.      ``"[SPE, Refuse a saved mapping that names none of this transcript's speakers.      Arg, transcript_speaker_labels(), no_paid_calls(), A stand-in for the paid analysis that records whether it was wanted., Stands in for a paid call. Being called at all is the failure., The outcome of one patched ``speaker_assignment_step``. (+10 more)

### Community 40 - "Before-snapshots and hostile-name safety"
Cohesion: 0.13
Nodes (19): Every file under ``root``, mapped to its bytes.      The SQLite index is skipp, ``base_name`` is an opaque index key, never joined onto a directory., test_a_hostile_base_name_writes_no_speaker_mapping(), The load-bearing property is not the status code; it is that nothing moved., Every file under ``root``, mapped to its bytes. Used to prove absence., test_a_hostile_base_name_touches_nothing(), test_a_label_that_is_not_in_the_transcript_is_refused(), load_manifest() (+11 more)

### Community 41 - "Job execution: event sink, lock file, exit codes"
Cohesion: 0.14
Nodes (15): JsonlEventSink, An :class:`whycast.events.EventSink` that appends JSON lines to a file.      O, Close the file. Safe to call twice., Run one job to completion and return the process exit code.      Everything -, The body of :func:`run_job`, with the event sink already installed., run_job(), _run_with_sink(), Another worker process holds the lock file.      Raised by :func:`run_worker`, (+7 more)

### Community 42 - "Index freshness and the artifact-serving fixtures"
Cohesion: 0.11
Nodes (17): A typo in a hand-edited file must be visible.      Falling back to the model h, awkward_client(), client_with_all_formats(), Write ``content`` byte for byte.      Not ``Path.write_text``: on Windows that, A restart must reflect the disk, including files rewritten in place.      ``_n, TestIndexFreshness, write(), The podcast directory, plus a sibling directory that must stay untouched. (+9 more)

### Community 43 - "CUDA runtime regression tests (ADR-001)"
Cohesion: 0.16
Nodes (15): CUDA runtime regression tests (ADR-001).  The failure these guard against: CTr, Repeated calls must not keep appending to PATH., The CUDA 12 cuBLAS that CTranslate2 needs must exist somewhere on PATH., A tiny silent-ish WAV; content does not matter, only that decoding works., End-to-end GPU proof: transcribe actual audio.      Runs in a subprocess so a, test_cublas12_is_reachable(), test_cuda_libs_land_on_path_when_wheels_installed(), test_ensure_cuda_libs_is_idempotent() (+7 more)

### Community 44 - "find_episodes.py: rename-collision analysis script"
Cohesion: 0.15
Nodes (16): analyze_episodes(), detect_rename_collisions(), find_mp3_files(), generate_backup_strategy(), generate_rename_script(), print_collision_analysis(), print_episode_analysis(), Print formatted analysis of episodes.          Args:         episodes: List o (+8 more)

### Community 45 - "Discarded partial downloads and dropped .bak files (ADR-009)"
Cohesion: 0.28
Nodes (16): ADR-004 Hybrid AI-analysis plus programmatic speaker assignment, ADR-009 Discard Partial Downloads and Drop Per-Artifact Backups, Explicit backup=False at Artifact Call Sites, Human Input Never Lives in a Generated Artifact, Per-Artifact .bak Backups, ADR-010 Persisted Editable Speaker Mapping as Pipeline Input, <base>_speakers.json Speaker Mapping File, Stale Mapping Flag After Re-transcription (+8 more)

### Community 46 - "Tests against the operator's real podcasts/ directory"
Cohesion: 0.12
Nodes (5): Scans the operator's actual ``podcasts/``.      Every assertion here must surv, ``webui.db.artifacts.path`` is a PRIMARY KEY; a repeat breaks rescan., Where a ``(kind, fmt)`` slot repeats, it is two real files, resolvable., ``webui.db`` keys on ``base_name.lower()`` and refuses collisions., TestRealPodcastDirectory

### Community 47 - "Job log directories and parked verdicts"
Cohesion: 0.16
Nodes (16): clear_pending_verdict(), ensure_job_dir(), events_path(), finish_or_record(), jobs_dir(), Root directory for per-job event logs.      ``WHYCAST_JOBS_DIR`` if set, else, Path of a job's JSONL event log: ``<jobs_dir>/<job_id>/events.jsonl``.      Cr, Create (if needed) and return a job's log directory.      The lazy half of :fu (+8 more)

### Community 48 - "Transcription and diarization decisions (ADR-001, ADR-002)"
Cohesion: 0.18
Nodes (15): ADR-001 Use faster-whisper on CUDA for transcription, ADR-001: faster-whisper (CTranslate2) on CUDA float16, ADR-002 Speaker diarization via pyannote.audio 3.1, ADR-002: pyannote.audio 3.1 speaker-diarization pipeline, Content-Length Download Verification, 2. Diarize (pyannote 3.1, GPU), 1. Prepare Audio, RSS Feed Download (+7 more)

### Community 49 - "Episode scanner tests: prefix collisions"
Cohesion: 0.16
Nodes (10): _artifact_names(), Tests for :mod:`whycast.episodes` - the tolerant podcast directory scanner (ADR, A podcast directory holding every naming trap, and nothing else.      Returns, ``episode_1`` and ``episode_10`` both exist; neither may swallow the other., ``episode_1.foo_summary.txt`` belongs to episode_1 and mints nothing., ``episode_10_speakers.json`` is episode_10's, never episode_1's., real(), _set_mtime() (+2 more)

### Community 50 - "Running the queue to completion with a real worker"
Cohesion: 0.13
Nodes (15): finished_selftest(), Run the queue to completion with a real worker, after the cost guard.      The, Enqueue a self-test, run it with a real worker, and return the row., A finished job's events arrive in full, in order, and the stream closes., ``Last-Event-ID: 3`` means "everything after 3", not the whole log again., Wrong in the safe direction: replay something twice, never skip it., Not the key, not the token, not the name of the variable holding them., The error column is a sentence for a human; the stack is in the log. (+7 more)

### Community 51 - "Application assembly and route registration"
Cohesion: 0.13
Nodes (15): create_app(), db_path_from_env(), The speaker mapping editor: one page and three API routes., Queue API, SSE stream, and the two job pages., The vocabulary and prompt editors, and their JSON API.      Every route in her, Render HTTP errors as a page for browsers, as JSON for the API.      Registere, Index database path from ``WHYCAST_WEBUI_DB``, else the repo default., Build the FastAPI application.      Arguments override the environment; both d (+7 more)

### Community 52 - "Request body parsing: JSON and form, with a size cap"
Cohesion: 0.19
Nodes (14): _declared_length(), _editor_payload(), _form_fields(), _is_form_request(), _job_request_payload(), Read ``{label: name}`` from a JSON or form-encoded body.      JSON: ``{"speake, ``Content-Length`` as an int, or None when absent or unparseable., True for ``application/x-www-form-urlencoded``, whatever the parameters. (+6 more)

### Community 53 - "The speaker editor's view of disk state"
Cohesion: 0.15
Nodes (14): _labels_with_context(), Import :mod:`whycast.pipeline.speakers`, or raise 503.      Imported lazily, p, Where this episode's mapping file lives, verified to be inside ``podcasts/``., Does this episode have a saved mapping? One stat, straight from disk.      Use, Every ``SPEAKER_xx`` label in ``text``, in order, with a few of its lines., Read a transcript exactly as the runner does. Returns ``(text, error)``., Everything the editor and its API need, read from disk. Blocking.      Called, Write the mapping with ``source="human"``. Blocking; run it in a thread. (+6 more)

### Community 54 - "Editor validators and the operator's own files"
Cohesion: 0.15
Nodes (14): Exception, The operator's actual files must be saveable through this UI.      The validat, test_the_real_vocabulary_and_prompts_pass_their_own_validators(), _EditorRejected, _normalise_newlines(), A save this app refuses to write, with a message for the operator.      Separa, CRLF and bare CR to LF.      A ``<textarea>`` posts CRLF per the HTML spec, an, ``object_pairs_hook`` that refuses a repeated key.      ``json.loads`` keeps t (+6 more)

### Community 55 - "The event sink contract: JSONL lines and logging"
Cohesion: 0.14
Nodes (14): A JSONL reader splits on \\n; CRLF here would be a Windows-only bug., A logging problem must not take down a pipeline step., test_a_pipeline_supplied_progress_is_never_overwritten(), test_the_sink_never_raises_at_the_caller(), test_the_sink_numbers_events_from_one_and_appends(), test_the_sink_writes_one_lf_terminated_line_per_event(), output_dir_from_env(), The directory jobs read from and write artifacts to.      ``WHYCAST_PODCAST_DI (+6 more)

### Community 56 - "Artifact naming shapes and duplicate slots"
Cohesion: 0.15
Nodes (6): ``<base>_ts_speaker_assignment`` is the assignment, not a ts artifact., ``<base>.txt`` and ``<base>_transcript.txt`` are both kind=transcript., Two files in one ``(kind, fmt)`` slot: both indexed, newest served.          T, The mp3 is gone; the artifacts still form one episode, not three strays., The ffmpeg leftover belongs to episode_13 but names no kind., TestNamingShapes

### Community 57 - "Backlog tasks: scanner, temp sweep, correction workflows"
Cohesion: 0.18
Nodes (12): TASK-002 Tolerante episode-scanner (longest-base-match), TASK-003 sweep_stale_temps opruiming, TASK-004 Correcties naar de invoer, nooit naar een artefact, TASK-004 <base>_speakers.json als persistente pipeline-invoer, TASK-005 Read-only config-viewer met allowlist, ADR-004: two-phase speaker mapping (LLM oordeelt, code past toe), ADR-006 Post-transcription vocabulary correction via vocabulary.json, ADR-006: vocabulary.json post-hoc replacement map (+4 more)

### Community 58 - "Vocabulary correction (ADR-006)"
Cohesion: 0.18
Nodes (12): apply_vocabulary_corrections(), load_vocabulary_mappings(), process_transcript_with_vocabulary(), Process a transcript with custom vocabulary corrections.          Args:, Load vocabulary mappings from a JSON file.          Args:         vocab_file:, Apply vocabulary corrections to the transcribed text.          Args:, apply_vocabulary_corrections Golden Snapshot, load_vocabulary_mappings Golden Snapshot (+4 more)

### Community 59 - "Vocabulary editor validation and its 400s"
Cohesion: 0.17
Nodes (12): The control. A cap that refuses everything proves nothing., ``RecursionError`` is neither ``ValueError`` nor ``JSONDecodeError``.      It, Valid JSON source, but no UTF-8 exists for it.      It passed every validator, test_a_form_body_within_the_cap_still_works(), test_a_lone_surrogate_is_a_400_not_a_500(), test_a_rejected_vocabulary_leaves_the_file_byte_identical(), test_deeply_nested_json_is_a_400_not_a_500(), test_the_vocabulary_keeps_one_backup() (+4 more)

### Community 60 - "Environment-variable configuration (ADR-007)"
Cohesion: 0.17
Nodes (12): allowed_hosts_from_env(), _config_payload(), host_from_env(), is_loopback_host(), podcast_dir_from_env(), port_from_env(), The configuration this UI is willing to show.      Reads only the names in the, Podcast directory from ``WHYCAST_PODCAST_DIR``, else the repo default. (+4 more)

### Community 61 - "Index queries: episodes and their artifacts"
Cohesion: 0.17
Nodes (12): _artifacts_by_episode(), _episode_dict(), _escape_like(), get_episode(), _iso(), list_episodes(), Return every indexed episode, in scan order, with its artifact summary.      A, Return one episode with its artifacts nested, or None if unknown.      Lookup (+4 more)

### Community 62 - "_safe_file: the only gate between a request and the filesystem"
Cohesion: 0.17
Nodes (4): ``commonpath`` raises across drives; the prefix test must not.          Live c, Direct tests of :func:`webui.app._safe_file`.      This is the only gate betwe, ``podcasts_evil`` must not pass a check meant for ``podcasts``.          A pre, TestSafeFileGuard

### Community 63 - "Command-line entry points"
Cohesion: 0.2
Nodes (11): main(), Run the server. Returns a process exit code., ``python -m webui.runner <job_id>``. Returns the process exit code., ``python -m webui.worker``. Returns a process exit code., Main function with command line argument handling., Main function with command line interface., quick_update_all(), Update a specific episode by name. (+3 more)

### Community 64 - "Queue robustness: terminal jobs, foreign schemas, two runners"
Cohesion: 0.18
Nodes (11): A hand-edited row must not take the whole dashboard down with it., The first verdict stands, and the attempt to overwrite it is announced., Two running jobs means two processes on one GPU. It must be visible., An older build pointed at a newer database must not 'fix' it., test_a_newer_queue_schema_is_left_untouched(), test_a_terminal_job_never_moves_again(), test_active_job_reports_the_oldest_and_warns_when_two_run(), test_cancelling_a_terminal_job_changes_nothing() (+3 more)

### Community 65 - "Windows entry point: no console, no stdout"
Cohesion: 0.18
Nodes (10): pythonw.exe under Task Scheduler gets sys.stdout is None.      Regression for, Never replace a working stdout: that would swallow console output., test_a_process_that_has_streams_is_left_alone(), test_the_server_survives_having_no_console(), build_parser(), _ensure_output_streams(), Entry point for the WHYcast web UI: ``python -m webui``.  Starts uvicorn on th, Give the process real stdout/stderr when Windows gave it none.      ``pythonw. (+2 more)

### Community 66 - "Per-job tables of the kinds a step reads and writes"
Cohesion: 0.27
Nodes (10): Move Aside Before the Run Starts, _JOB_INPUT_KINDS, Per-Job Table of Kinds Read and Written, _JOB_OUTPUT_KINDS, webui.runner, Pipeline Job Types, retranscribe Job, Single-Step Re-run (+2 more)

### Community 67 - "Chunked request bodies cannot bypass the size cap"
Cohesion: 0.2
Nodes (10): chunked(), An iterable body, which makes httpx send ``Transfer-Encoding: chunked``., The bypass, closed. A form body is capped exactly like a JSON one., The branch that already held, kept honest by the same reader., The consequence that outlived the request: a prompt sent to the paid API., The other consequence: the ``jobs`` table shares a file with the index., test_a_chunked_form_body_cannot_slip_past_the_cap(), test_a_chunked_json_body_cannot_slip_past_the_cap_either() (+2 more)

### Community 68 - "Connection and lookup helpers, and their 404s and 503s"
Cohesion: 0.2
Nodes (10): A connection with both schemas on it, closed afterwards., A connection with both schemas and the podcast directory indexed., _conn(), _get_episode_or_404(), _get_job_or_404(), _queue_or_503(), The index connection, after confirming the queue tables exist.      ``ensure_j, Look ``job_id`` up in the queue, or raise 404.      Like ``base_name``, a job (+2 more)

### Community 69 - "The job catalogue the UI and API share"
Cohesion: 0.2
Nodes (10): _episode_actions(), _job_spec(), job_types_payload(), The catalogue entry for the job that applies a mapping, or None.      Read fro, The job catalogue, as the API and the templates see it.      ``cost`` is the f, The enqueue actions the episode page offers, in :data:`EPISODE_ACTION_TYPES` ord, The single-step re-runs, in the order the pipeline runs them.      Kept apart, One entry of the job catalogue, with its ``cost`` and ``gpu`` flags.      Read (+2 more)

### Community 70 - "Built-in fallback pages when a template is missing"
Cohesion: 0.27
Nodes (10): _escape(), _fallback_page(), _job_detail_fallback(), _job_row_html(), _jobs_fallback(), A minimal standalone page, used when a template is not installed., One job as a table row for the fallback dashboard., The built-in queue dashboard. (+2 more)

### Community 71 - "App lifespan: opening the index and refreshing it when stale"
Cohesion: 0.2
Nodes (10): _as_epoch(), _index_is_stale(), _lifespan(), _newest_change(), _public_meta(), Filter index metadata down to :data:`META_KEYS`.      The single point where a, Best-effort read of a stored timestamp as epoch seconds., The newest mtime in ``podcast_dir``: the directory itself and its entries. (+2 more)

### Community 72 - "Queue schema creation and migration"
Cohesion: 0.24
Nodes (10): _begin(), ensure_job_schema(), JobQueueError, _migrate(), Refuse to work inside someone else's open transaction.      Not paranoia: the, Open a write transaction, refusing to nest.      ``BEGIN IMMEDIATE`` takes the, Raised when the queue is asked to do something it cannot do safely.      Disti, Create or migrate the queue tables. Idempotent; safe on every startup.      Ca (+2 more)

### Community 73 - "transcript_formatter.py: episode header script"
Cohesion: 0.25
Nodes (8): find_episode_transcripts(), Find all episode transcript files that need speaker assignment.          Retur, Extract episode number from basename for sorting., extract_episode_number(), Extract episode number from filename using various patterns.          Args:, format_transcript_with_headers(), Add standard header and footer to the transcript with proper episode information, Extract episode number from the output basename.          Args:         outpu

### Community 74 - "Model selection and file-based prompts (ADR-003, ADR-007)"
Cohesion: 0.32
Nodes (8): ADR-003 Per-task OpenAI model selection, ADR-003: per-task model env-vars met lengte-gebaseerde switch, ADR-007 Environment-variable configuration and file-based prompts, ADR Index, ADR README, bin/adr-index Generator, openai, python-dotenv

### Community 75 - "Queue concurrency: eight threads, two processes, one job each"
Cohesion: 0.25
Nodes (8): open_second_connection(), Eight threads, eight connections, one job each - never the same one twice., The real production shape: two processes, two connections, one queue.      The, Another connection to the same file, as a second process would have., ``{job_id: status}`` straight out of the table, bypassing the helpers., statuses(), test_concurrent_threads_each_get_a_different_job(), test_two_processes_each_get_a_different_job()

### Community 76 - "The artifact diff, read safely"
Cohesion: 0.25
Nodes (8): _diff_entry(), _job_diff_payload(), Compare a job's before-snapshot with the artifacts on disk now., One artifact's before/after comparison, safe on anything unreadable., Read a text file, or None when it is missing or not text., Return ``candidate`` resolved, if it is a real file inside ``root``.      This, _read_text_or_none(), _safe_file()

### Community 77 - "Editor file state and human-input writes"
Cohesion: 0.29
Nodes (8): _display_path(), _file_state(), _prompt_overview(), ``path`` relative to the repository root, or just its filename.      What the, What the editor knows about one file on disk.      Deliberately contains no ab, Every allowlisted prompt with its file state, for the list page., Write a human-authored input file, keeping the previous version.      ``backup, _write_input_file()

### Community 78 - "Output formats: markdown, HTML and wiki"
Cohesion: 0.29
Nodes (7): convert_markdown_to_html(), convert_markdown_to_wiki(), Convert markdown text to Wiki markup.          Args:         markdown_text: T, Convert markdown text to HTML.          Args:         markdown_text: The mark, markdown, convert_markdown_to_html Golden Snapshot, convert_markdown_to_wiki Golden Snapshot

### Community 79 - "Server bind flags versus environment"
Cohesion: 0.29
Nodes (3): AC #5 through ``python -m webui``, not just through a constant.      ``uvicorn, The flags are a convenience over ADR-007 variables, not a second store., TestServerBind

### Community 80 - "Path traversal, symlinks and null bytes"
Cohesion: 0.29
Nodes (4): No request may reach a file outside the podcast directory.      Both halves of, Either the client refuses to send it or the server refuses to serve it., The index is re-checked against the podcast directory before opening., TestPathTraversal

### Community 83 - "Artifact response headers and framable 404s"
Cohesion: 0.33
Nodes (6): _artifact_headers(), _artifact_problem(), _own_origin(), Artifact response headers, with this server's origin in the policy., A 404 the artifact frame can actually display.      Raising HTTPException here, The origin this request was addressed to, as a browser would spell it.

### Community 84 - "Picking the right artifact for a kind"
Cohesion: 0.33
Nodes (6): _artifacts_of(), _pick_artifact(), The transcript a speakers re-run would start from, or None.      Deliberately, The artifact rows of an episode dict., Best artifact of ``kind`` for ``episode``, or None.      Mirrors :meth:`whycas, _transcript_artifact()

### Community 85 - "Feed jobs: fetch latest, fetch all"
Cohesion: 0.33
Nodes (6): _job_fetch_all(), _job_fetch_latest(), Podcast feed from ``WHYCAST_RSSFEED``, else :data:`DEFAULT_RSSFEED`.      Deli, Download every feed episode that is not on disk yet. No GPU, no cost., Fetch the newest feed episode and run the full pipeline on it., rssfeed_from_env()

### Community 86 - "A live TestClient for the SSE stream"
Cohesion: 0.4
Nodes (5): client(), The property that makes this endpoint worth having: live progress.      A real, A TestClient over a throwaway index, queue and job-log directory., test_the_stream_delivers_a_real_job_as_it_runs(), A started TestClient over a freshly built app.      The ``with`` block is what

### Community 87 - "Index schema versioning"
Cohesion: 0.4
Nodes (3): A schema bump must rebuild the index tables, not corrupt the database., The phase-2 job queue will share this file and is NOT rebuildable., TestSchemaVersioning

### Community 90 - "Finishing a job and trimming its error"
Cohesion: 0.4
Nodes (5): finish(), Trim an error to a UI-sized blurb, pointing at the event log for more., Move a job to a terminal status and stamp its outcome.      Only ``queued`` an, _short_error(), One sentence for the database: type and message, no stack, no newlines.

### Community 91 - "Synthetic podcast directory fixtures"
Cohesion: 0.5
Nodes (4): podcast_dir(), A synthetic podcast directory the index can be built from., A synthetic podcasts directory, plus a secret file outside of it., A synthetic podcast directory: one episode with audio and a transcript.

### Community 92 - "SSE frame formatting"
Cohesion: 0.5
Nodes (4): _job_event_stream(), One SSE frame. ``event_id`` becomes the browser's ``Last-Event-ID``., Yield SSE frames for one job until it reaches a terminal status.      Shape of, _sse_frame()

### Community 93 - "The speaker prompts and the WHYcast co-hosts"
Cohesion: 1.0
Nodes (3): Speaker Analysis Prompt (SPEAKER_NN -> name mapping), WHYcast co-hosts Nancy and Ad, Speaker Unknown Attribution Prompt (conversational flow analysis)

### Community 94 - "Editable-input fixtures redirected to tmp_path"
Cohesion: 0.67
Nodes (3): inputs_dir(), Point every editable input file at ``tmp_path``., Redirect every editable input file into ``tmp_path``.      Returns the directo

### Community 95 - "A real uvicorn on a real socket"
Cohesion: 0.67
Nodes (3): live_server(), A real uvicorn on a real socket, for the one test that needs live bytes., A real uvicorn on a real loopback socket, in a background thread.      Needed

### Community 97 - "Scratch queue fixtures"
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
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `split_into_chunks()` and `Prompt Editor Page`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `parse_speaker_mapping_from_analysis()` and `Speaker Mapping Editor Page`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `Speaker Label to Name Mapping Shape` and `Speaker Editor Context (episode, state, apply_job, active_job, episode_jobs)`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `ConfigurationError` connect `Episode scanner guarantees on real filenames` to `Web UI HTTP layer tests and fixtures`, `The whycast pipeline step modules`, `Awkward but legal input: wildcards, non-ASCII, zero bytes`, `Episode page artifact rendering and its acceptance criteria`, `Single-step re-runs of one post-processing step`, `Episode scanner internals: suffix peeling and kind ranking`, `Recognising <base>_speakers.json as an input, not an artifact`, `Job context, cancellation and the free self-test`, `Index freshness and the artifact-serving fixtures`, `Tests against the operator's real podcasts/ directory`, `Episode scanner tests: prefix collisions`, `Editor validators and the operator's own files`, `Artifact naming shapes and duplicate slots`, `Environment-variable configuration (ADR-007)`, `_safe_file: the only gate between a request and the filesystem`, `Server bind flags versus environment`, `Path traversal, symlinks and null bytes`, `Episode detail API: case, spaces, 404s`, `Rescan idempotence and the metadata allowlist`, `Index schema versioning`, `The index is disposable and never writes to podcasts/`, `A missing podcast directory is not a crash`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `parse_speaker_mapping_from_analysis()` connect `Speaker analysis and programmatic assignment` to `Web UI templates and their shared macros`, `The whycast pipeline step modules`, `LLM steps: model choice, chunking and the API key`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `Speaker Mapping Editor Page` connect `Web UI templates and their shared macros` to `Speaker analysis and programmatic assignment`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `scan_podcasts()` (e.g. with `.test_episode_10_blog_does_not_attach_to_episode_1()` and `.test_episode_1_keeps_its_own_artifacts()`) actually correct?**
  _`scan_podcasts()` has 45 INFERRED edges - model-reasoned connections that need verification._