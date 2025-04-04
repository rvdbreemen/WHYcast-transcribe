#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for the WHYcast Transcribe tool.
"""

import os
import sys
import logging
import argparse
import glob
from typing import Optional

# Import configuration variables
try:
    from config import VERSION
except ImportError as e:
    VERSION = "0.0.6"  # Default version if config import fails

# Import from other modules
from logging_utils import setup_logging
from transcription_utils import setup_model, transcribe_audio, cleanup_resources
from file_utils import (create_output_paths, write_transcript_files, 
                      transcription_exists)
from format_utils import convert_existing_blogs
from vocabulary_utils import process_transcript_with_vocabulary
from summarization_utils import process_summary
from workflow_utils import (regenerate_summary,
                          regenerate_all_summaries, regenerate_all_blogs,
                          regenerate_cleaned_transcript, regenerate_all_cleaned,
                          regenerate_full_workflow, regenerate_all_full_workflow,
                          regenerate_blogs_from_cleaned)
from podcast_utils import download_latest_episode, process_all_episodes

logger = setup_logging()

def main(input_file: str, model_size: Optional[str] = None, output_dir: Optional[str] = None, 
         skip_summary: bool = False, force: bool = False, is_batch_mode: bool = False,
         regenerate_summary_only: bool = False, skip_vocabulary: bool = False,
         model=None) -> None:
    """
    Main function to process an audio file.
    
    Args:
        input_file: Path to the input audio file
        model_size: Override default model size if provided
        output_dir: Directory to save output files (defaults to input file directory)
        skip_summary: Flag to skip summary generation
        force: Flag to force regeneration of transcription even if it exists
        is_batch_mode: Flag indicating if running as part of batch processing
        regenerate_summary_only: Flag to only regenerate summary from existing transcript file
        skip_vocabulary: Flag to skip vocabulary corrections
        model: Optional pre-initialized WhisperModel to use (to avoid reloading)
    """
    try:
        # Process for summary regeneration
        if regenerate_summary_only:
            if input_file.endswith('.txt') and 'summary' not in input_file:
                # Input is already a transcript file
                transcript_file = input_file
                logging.info(f"Regenerating summary from transcript: {transcript_file}")
                success = regenerate_summary(transcript_file)
                if not success and not is_batch_mode:
                    sys.exit(1)
                return
            else:
                # Input is an audio file, construct the transcript filename
                transcript_file = os.path.splitext(input_file)[0] + ".txt"
                if not os.path.exists(transcript_file):
                    logging.error(f"Cannot regenerate summary: transcript file {transcript_file} does not exist")
                    if not is_batch_mode:
                        sys.exit(1)
                    return
                logging.info(f"Regenerating summary from transcript: {transcript_file}")
                success = regenerate_summary(transcript_file)
                if not success and not is_batch_mode:
                    sys.exit(1)
                return
        
        # Check input file
        if not os.path.exists(input_file):
            logging.error(f"Input file does not exist: {input_file}")
            if not is_batch_mode:
                sys.exit(1)
            return
        
        # Skip if transcription already exists and not forcing
        if not force and transcription_exists(input_file):
            logging.info(f"Skipping {input_file} - transcription already exists (use --force to override)")
            return
        
        # Setup output directory and create if necessary
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(os.path.splitext(input_file)[0])
            output_base = os.path.join(output_dir, base_name)
        else:
            output_base = os.path.splitext(input_file)[0]
        
        output_file, output_file_timestamped, output_summary_file = create_output_paths(output_base)
        
        # Setup model if not provided
        if model is None:
            model = setup_model(model_size)
        
        # Transcribe
        result = transcribe_audio(model, input_file)
        segments, info = result
        
        logging.info(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        # Apply vocabulary corrections if enabled
        apply_vocab = None if skip_vocabulary else process_transcript_with_vocabulary
        
        # Write transcript files
        full_transcript = write_transcript_files(segments, output_file, output_file_timestamped, apply_vocab)
        
        # Generate and save summary (if not skipped)
        if not skip_summary:
            process_summary(full_transcript, output_summary_file)
    except Exception as e:
        logging.error(f"An error occurred processing {input_file}: {str(e)}")
        if not is_batch_mode:
            sys.exit(1)
        # In batch mode, we continue to the next file

def process_batch(input_pattern: str, **kwargs) -> None:
    """
    Process multiple files matching the given pattern.
    
    Args:
        input_pattern: Glob pattern for input files
        **kwargs: Additional arguments to pass to main()
    """
    files = glob.glob(input_pattern)
    if not files:
        logging.warning(f"No files found matching pattern: {input_pattern}")
        return
    
    logging.info(f"Found {len(files)} files to process")
    model = setup_model(kwargs.get('model_size'))
    for file in files:
        logging.info(f"Processing file: {file}")
        # Pass the model instance to avoid reloading for each file
        kwargs_with_model = dict(kwargs)
        kwargs_with_model['model'] = model
        main(file, is_batch_mode=True, **kwargs_with_model)
        
    # Clean up after processing all files
    cleanup_resources(model)

def process_all_mp3s(directory: str, **kwargs) -> None:
    """
    Process all MP3 files in the given directory.
    
    Args:
        directory: Directory containing MP3 files
        **kwargs: Additional arguments to pass to main()
    """
    if not os.path.isdir(directory):
        directory = os.path.dirname(directory) or '.'
    mp3_pattern = os.path.join(directory, "*.mp3")
    process_batch(mp3_pattern, **kwargs)

def process_episode(file_path: str, **kwargs):
    """
    Process a downloaded podcast episode.
    
    Args:
        file_path: Path to the downloaded episode file
        **kwargs: Additional arguments to pass to main()
    """
    main(file_path, is_batch_mode=True, **kwargs)

def run_cli():
    """
    Parse command-line arguments and execute the appropriate action.
    """
    parser = argparse.ArgumentParser(description=f'WHYcast Transcribe v{VERSION} - Transcribe audio files and generate summaries')
    parser.add_argument('input', nargs='?', help='Path to the input audio file, directory, or glob pattern (default: current directory for regeneration options)')
    parser.add_argument('--batch', '-b', action='store_true', help='Process multiple files matching pattern')
    parser.add_argument('--all-mp3s', '-a', action='store_true', help='Process all MP3 files in directory')
    parser.add_argument('--model', '-m', help='Model size (e.g., "large-v3", "medium", "small")')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files')
    parser.add_argument('--skip-summary', '-s', action='store_true', help='Skip summary generation')
    parser.add_argument('--force', '-f', action='store_true', help='Force regeneration of transcriptions even if they exist')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--version', action='version', version=f'WHYcast Transcribe v{VERSION}')
    parser.add_argument('--regenerate-summary', '-r', action='store_true', help='Regenerate summary and blog from existing transcript')
    parser.add_argument('--regenerate-all-summaries', '-R', action='store_true', 
                         help='Regenerate summaries for all transcripts in directory (uses current dir if no input)')
    parser.add_argument('--regenerate-all-blogs', '-B', action='store_true', 
                         help='Regenerate only blog posts for all transcripts in directory (uses current dir if no input)')
    parser.add_argument('--regenerate-cleaned', '-rc', action='store_true', 
                         help='Regenerate cleaned version from existing transcript')
    parser.add_argument('--skip-vocabulary', action='store_true', help='Skip custom vocabulary corrections')
    parser.add_argument('--regenerate-all-cleaned', action='store_true',
                        help='Regenerate cleaned transcripts for all transcript files in directory (uses input dir or download-dir if not provided)')
    parser.add_argument('--regenerate-full-workflow', action='store_true',
                        help='Run a single workflow to generate cleaned, summary, blog, and blog_alt1 from existing transcript')
    parser.add_argument('--regenerate-blogs-from-cleaned', action='store_true',
                       help='Regenerate blog posts using only cleaned transcripts')
    
    # Add podcast feed arguments
    parser.add_argument('--feed', '-F', default="https://whycast.podcast.audio/@whycast/feed.xml", 
                       help='RSS feed URL to download latest episode (default: WHYcast feed)')
    parser.add_argument('--download-dir', '-D', default='podcasts', 
                       help='Directory to save downloaded episodes (default: podcasts)')
    parser.add_argument('--no-download', '-N', action='store_true', 
                       help='Disable automatic podcast download')
    
    # Add new argument for processing all episodes
    parser.add_argument('--all-episodes', '-A', action='store_true', 
                       help='Process all episodes from the podcast feed instead of just the latest')
    
    # Add new argument for converting existing blogs to HTML and Wiki formats
    parser.add_argument('--convert-blogs', '-C', action='store_true',
                       help='Convert existing blog text files to HTML and Wiki formats')
    
    args = parser.parse_args()
    
    logging.info(f"WHYcast Transcribe {VERSION} starting up")
    
    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if we should convert blogs
    if args.convert_blogs:
        directory = args.input if args.input else args.download_dir
        logging.info(f"Converting all blog files in directory: {directory}")
        convert_existing_blogs(directory)
        sys.exit(0)
    
    # Check if we should process all episodes
    should_process_all_episodes = (args.all_episodes and 
                                not args.no_download and
                                not args.input and 
                                not args.regenerate_all_summaries and
                                not args.regenerate_all_blogs and
                                not args.regenerate_blogs_from_cleaned and
                                not args.regenerate_summary)
    
    if should_process_all_episodes:
        feed_url = args.feed
        download_dir = args.download_dir
        
        logging.info(f"Processing all episodes from {feed_url}")
        process_all_episodes(
            feed_url, 
            download_dir, 
            lambda file_path: process_episode(
                file_path,
                model_size=args.model,
                output_dir=args.output_dir,
                skip_summary=args.skip_summary,
                force=args.force,
                skip_vocabulary=args.skip_vocabulary
            )
        )
        sys.exit(0)
    
    # Determine if we should check the podcast feed for latest episode (original behavior)
    should_check_feed = (not args.no_download and 
                         not args.all_episodes and
                         not args.input and 
                         not args.regenerate_all_summaries and
                         not args.regenerate_all_blogs and
                         not args.regenerate_blogs_from_cleaned and
                         not args.regenerate_summary)
    
    if should_check_feed:
        feed_url = args.feed
        download_dir = args.download_dir
        
        logging.info(f"Checking for the latest episode from {feed_url}")
        episode_file = download_latest_episode(feed_url, download_dir)
        
        if episode_file:
            logging.info(f"Processing newly downloaded episode: {episode_file}")
            main(episode_file, model_size=args.model, output_dir=args.output_dir, 
                 skip_summary=args.skip_summary, force=args.force,
                 skip_vocabulary=args.skip_vocabulary)
            sys.exit(0)
        else:
            logging.info("No new episode to download or process")
            sys.exit(0)
    
    # Continue with existing functionality if input is provided or special mode is requested
    if args.regenerate_blogs_from_cleaned:
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating blogs from cleaned transcripts in directory: {directory}")
        regenerate_blogs_from_cleaned(directory)
    elif args.regenerate_all_blogs:
        # Allow regenerate_all_blogs without an input by using podcasts directory
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating all blogs in directory: {directory}")
        regenerate_all_blogs(directory)
    elif args.regenerate_all_summaries:
        # Allow regenerate_all_summaries without an input by using podcasts directory
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating all summaries in directory: {directory}")
        regenerate_all_summaries(directory)
    elif args.regenerate_all_cleaned:
        directory = args.input if args.input else args.download_dir
        logging.info(f"Regenerating all cleaned transcripts in directory: {directory}")
        regenerate_all_cleaned(directory)
    elif not args.input:
        parser.print_help()
        sys.exit(1)
    elif args.regenerate_cleaned:
        if os.path.isdir(args.input):
            logging.error("Please specify a transcript file when using --regenerate-cleaned, not a directory")
            sys.exit(1)
        elif not os.path.isfile(args.input):
            logging.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        else:
            success = regenerate_cleaned_transcript(args.input)
            if not success:
                sys.exit(1)
    elif args.regenerate_summary:
        if os.path.isdir(args.input):
            logging.error("Please specify a transcript file when using --regenerate-summary, not a directory")
            sys.exit(1)
        elif not os.path.isfile(args.input):
            logging.error(f"Input file does not exist: {args.input}")
            sys.exit(1)
        else:
            main(args.input, regenerate_summary_only=True, skip_vocabulary=args.skip_vocabulary)
    elif args.regenerate_full_workflow:
        if not args.input:
            logging.error("You must specify an input file or directory with --regenerate-full-workflow.")
            sys.exit(1)
        
        if os.path.isdir(args.input):
            # Process all transcripts in directory
            logging.info(f"Running full workflow for all transcripts in directory: {args.input}")
            regenerate_all_full_workflow(args.input)
        else:
            # Process single file
            regenerate_full_workflow(args.input)
        sys.exit(0)
    elif args.all_mp3s:
        process_all_mp3s(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
                         force=args.force, skip_vocabulary=args.skip_vocabulary)
    elif args.batch:
        process_batch(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
                      force=args.force, skip_vocabulary=args.skip_vocabulary)
    else:
        main(args.input, model_size=args.model, output_dir=args.output_dir, skip_summary=args.skip_summary,
             force=args.force, regenerate_summary_only=args.regenerate_summary,
             skip_vocabulary=args.skip_vocabulary)

if __name__ == "__main__":
    run_cli()