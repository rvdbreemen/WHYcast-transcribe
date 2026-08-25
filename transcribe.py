#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WHYcast Transcribe - v0.4.0 - CLI shim (ADR-008)

This module is now a thin CLI shim over the whycast package. All pipeline
logic (transcription, diarization, speaker assignment, summarization, blog
generation, RSS fetching) lives in whycast/ and whycast/pipeline/; this file
only parses arguments, installs the console event sink, initializes logging
and TF32, translates pipeline exceptions into exit codes, and re-exports the
public functions under their historical names so existing scripts that do
"from transcribe import ..." keep working (see docs/adr/ ADR-008).
"""

import argparse
import logging
import os

# ---------------------------------------------------------------------------
# Compatibility re-exports (ADR-008): every public name that used to live in
# this file is imported from its new whycast home and re-exported unchanged,
# so batch scripts (batch_update_speaker_assignments.py, quick_batch_update.py,
# find_episodes.py) that import from transcribe keep working.
# ---------------------------------------------------------------------------

# Config constants (VERSION, MODEL_SIZE, OPENAI_MODEL, prompt paths, ...)
from whycast.config import *  # noqa: F401,F403
from whycast.config import VERSION  # explicit: used below and by importers

# Errors
from whycast.errors import (  # noqa: F401
    EpisodeNotFoundError,
    PipelineError,
    SecurityError,
)

# Logging
from whycast.logging_setup import setup_logging  # noqa: F401

# Validation
from whycast.validation import (  # noqa: F401
    check_file_size,
    validate_directory_path,
    validate_file_path,
)

# Audio preparation
from whycast.pipeline.audio import prepare_audio_for_diarization  # noqa: F401

# Diarization
from whycast.pipeline.diarization import (  # noqa: F401
    diarize_audio,
    get_huggingface_token,
    set_huggingface_token,
)

# RSS feed / episode management
from whycast.pipeline.feed import (  # noqa: F401
    delete_episode_files,
    download_all_episodes_from_rssfeed,
    podcast_fetching_workflow,
)

# GPU helpers
from whycast.pipeline.gpu import (  # noqa: F401
    enable_tf32,
    force_cuda_device,
    get_default_device,
    is_cuda_available,
    maximize_gpu_utilization,
    verify_gpu_setup,
)

# OpenAI / LLM helpers
from whycast.pipeline.llm import (  # noqa: F401
    choose_appropriate_model,
    ensure_api_key,
    estimate_token_count,
    process_large_text_in_chunks,
    process_with_openai,
    read_prompt_file,
    split_into_chunks,
    summarize_large_transcript,
    truncate_transcript,
)

# Output formats
from whycast.pipeline.outputs import (  # noqa: F401
    convert_markdown_to_html,
    convert_markdown_to_wiki,
    write_all_format,
)

# Transcript post-processing steps
from whycast.pipeline.postprocess import (  # noqa: F401
    alt_blog_step,
    blog_step,
    cleanup_step,
    history_step,
    process_transcript_workflow,
    summary_step,
)

# Speaker assignment
from whycast.pipeline.speakers import (  # noqa: F401
    analyze_speakers_with_o4,
    analyze_transcript_changes,
    apply_speaker_mapping_programmatically,
    attribute_unknown_speakers_with_ai,
    handle_unknown_speakers,
    merge_speaker_lines,
    parse_speaker_mapping_from_analysis,
    speaker_assignment_fallback,
    speaker_assignment_programmatic,
    speaker_assignment_step,
    write_merged_transcript,
)

# Transcription
from whycast.pipeline.transcription import (  # noqa: F401
    format_timestamp,
    setup_model,
    transcribe_audio,
    write_transcript_files,
)

# Vocabulary corrections
from whycast.pipeline.vocabulary import (  # noqa: F401
    apply_vocabulary_corrections,
    load_vocabulary_mappings,
    process_transcript_with_vocabulary,
)

# Full workflow
from whycast.pipeline.workflow import full_workflow  # noqa: F401

# Event sink for console output
from whycast.events import ConsoleSink, set_sink


# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f'WHYcast Transcribe v{VERSION} - Transcribe audio files and generate summaries')
    parser.add_argument('input', nargs='?', help='Path to the input audio file, directory, or glob pattern')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files', default='./podcasts')
    parser.add_argument('--rssfeed', '-r', help='RSS feed URL to fetch latest episodes from', default=os.environ.get('WHYCAST_RSSFEED', 'https://whycast.podcast.audio/@whycast/feed.xml'))
    parser.add_argument('--force', action='store_true', help='Force re-download and re-transcribe the latest episode, deleting all related files first')
    parser.add_argument('--version', action='version', version=f'WHYcast Transcribe v{VERSION}')
    parser.add_argument('--fetch-all', action='store_true', help='Download all mp3 episodes from the RSS feed to the output directory and exit')

    args = parser.parse_args()

    # CLI-side initialization (used to happen at import time / inside the old
    # monolith): console sink for pipeline progress events, logging, TF32.
    set_sink(ConsoleSink())
    setup_logging()
    enable_tf32()
    logging.info(f"WHYcast Transcribe {VERSION} starting up")

    try:
        # Handle --fetch-all: download all mp3s and exit before any other logic
        if args.fetch_all:
            rssfeed = args.rssfeed or os.environ.get("WHYCAST_RSSFEED", "https://whycast.podcast.audio/@whycast/feed.xml")
            output_dir = args.output_dir or os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
            download_all_episodes_from_rssfeed(rssfeed, output_dir)
            exit(0)

        # Handle --force: fetch latest, delete all related files, then run workflow
        if args.force:
            rssfeed = args.rssfeed or os.environ.get("WHYCAST_RSSFEED", "https://whycast.podcast.audio/@whycast/feed.xml")
            output_dir = args.output_dir or os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
            if args.input:
                base_name = os.path.splitext(os.path.basename(str(args.input)))[0] if args.input else None
                input_path = os.path.abspath(str(args.input))
                if base_name:
                    print(f"--force: Deleting all files for episode base name '{base_name}' in {output_dir} (excluding input file)")
                    delete_episode_files(base_name, output_dir, exclude_files=[input_path])
                full_workflow(audio_file=args.input, output_dir=output_dir, rssfeed=rssfeed, force=False)
                exit(0)
            else:
                result = podcast_fetching_workflow(rssfeed, output_dir, return_base_name=True)
                if not result or not isinstance(result, tuple) or not result[0] or not result[1]:
                    print("No episode could be fetched from the feed.")
                    exit(1)
                audio_file, base_name = result
                input_path = os.path.abspath(audio_file) if audio_file else None
                if base_name:
                    print(f"--force: Deleting all files for episode base name '{base_name}' in {output_dir} (excluding input file)")
                    delete_episode_files(base_name, output_dir, exclude_files=[input_path] if input_path else None)
                # Re-fetch after deletion to ensure mp3 is present
                result = podcast_fetching_workflow(rssfeed, output_dir, return_base_name=True)
                if not result or not isinstance(result, tuple) or not result[0]:
                    print("Failed to download episode after deletion.")
                    exit(1)
                audio_file, base_name = result
                full_workflow(audio_file=audio_file, output_dir=output_dir, rssfeed=rssfeed, force=False)
                exit(0)
        # If no parameters, check if latest mp3 is already downloaded and transcribed
        if not args.input:
            rssfeed = args.rssfeed or os.environ.get("WHYCAST_RSSFEED", "https://whycast.podcast.audio/@whycast/feed.xml")
            output_dir = args.output_dir or os.environ.get("WHYCAST_OUTPUT_DIR", "./podcasts")
            result = podcast_fetching_workflow(rssfeed, output_dir, return_base_name=True)
            if result and isinstance(result, tuple) and result[0] and result[1]:
                audio_file, base_name = result
                transcript_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
                if os.path.exists(transcript_path):
                    print(f"Transcript for latest episode '{base_name}' already exists at {transcript_path}. Exiting.")
                    exit(0)
            # If not found, proceed as normal
            audio_file = result[0] if result and isinstance(result, tuple) else result
            full_workflow(audio_file=audio_file, output_dir=output_dir, rssfeed=rssfeed, force=False)
            exit(0)
        # Always call the main workflow with parsed arguments if input is provided
        full_workflow(
            audio_file=args.input,
            output_dir=args.output_dir,
            rssfeed=args.rssfeed,
            force=False
        )
    except EpisodeNotFoundError as e:
        # Library code raises instead of print+exit (ADR-008); translate here.
        print(str(e) or "No episode could be fetched from the feed.")
        exit(1)
    except PipelineError as e:
        print(str(e))
        exit(1)
