#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import torchaudio # Import torchaudio
from typing import Optional # Add this import
from whycast_transcribe.config import (
    VERSION, MODEL_SIZE, USE_SPEAKER_DIARIZATION,
    DIARIZATION_MIN_SPEAKERS, DIARIZATION_MAX_SPEAKERS,
    DIARIZATION_MODEL
)
from whycast_transcribe.model_manager import setup_models
from whycast_transcribe.transcribe import transcribe_audio
from whycast_transcribe.utils.file_helpers import write_transcript_files
from whycast_transcribe.postprocess import run_postprocessing, write_workflow_outputs


def get_audio_duration(audio_path: str) -> Optional[float]:
    """Calculate the duration of an audio file in seconds using torchaudio."""
    try:
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        return duration
    except Exception as e:
        logging.error(f"Could not get duration for {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=f"WHYcast Transcribe v{VERSION}: Transcribe audio and generate artifacts"
    )
    parser.add_argument('input', nargs='?', help='Path to audio file, transcript, directory, or glob')
    parser.add_argument('--batch', '-b', action='store_true', help='Process multiple files matching pattern')
    parser.add_argument('--all-mp3s', '-a', action='store_true', help='Process all MP3 files in directory')
    parser.add_argument('--model', '-m', help='Model size override (e.g. large-v3, medium)')
    parser.add_argument('--output-dir', '-o', help='Directory to save output files')
    parser.add_argument('--skip-summary', '-s', action='store_true', help='Skip summary and blog generation')
    parser.add_argument('--force', '-f', action='store_true', help='Force regeneration of outputs')
    parser.add_argument('--regenerate-summary', '-r', action='store_true', help='Regenerate summary and blog')
    parser.add_argument('--regenerate-cleaned', '-rc', action='store_true', help='Regenerate cleaned transcript')
    parser.add_argument('--generate-history', '-H', action='store_true', help='Generate history lesson extraction')
    parser.add_argument('--regenerate-all-summaries', '-R', action='store_true', help='Regenerate summaries for all transcripts')
    parser.add_argument('--regenerate-all-blogs', '-B', action='store_true', help='Regenerate blog posts for all transcripts')
    parser.add_argument('--regenerate-all-history', action='store_true', help='Regenerate history extractions for all transcripts')
    parser.add_argument('--regenerate-all-cleaned', action='store_true', help='Regenerate cleaned transcripts for all transcripts')
    parser.add_argument('--regenerate-full-workflow', action='store_true', help='Run full workflow on existing transcripts')
    parser.add_argument('--convert-blogs', '-C', action='store_true', help='Convert existing blog text files to HTML/Wiki')
    parser.add_argument('--diarize', action='store_true', help='Enable speaker diarization')
    parser.add_argument('--no-diarize', action='store_true', help='Disable speaker diarization')
    parser.add_argument('--min-speakers', type=int, help=f'Min number of speakers (default {DIARIZATION_MIN_SPEAKERS})')
    parser.add_argument('--max-speakers', type=int, help=f'Max number of speakers (default {DIARIZATION_MAX_SPEAKERS})')
    parser.add_argument('--diarization-model', default=DIARIZATION_MODEL, help='Diarization model to use')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--version', action='version', version=f"WHYcast Transcribe v{VERSION}")
    args = parser.parse_args()

    # Configure logging to console and file
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = os.path.join(os.getcwd(), 'transcribe.log')
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    logging.info("WHYcast Transcribe started")

    # Determine if diarization should be enabled based on command line args
    use_di = args.diarize if args.diarize else (not args.no_diarize and USE_SPEAKER_DIARIZATION)
    
    # Setup models - both Whisper and diarization
    model, diarization_pipeline = setup_models(args.model or MODEL_SIZE, use_di)

    # Handle special regeneration and conversion commands
    if args.convert_blogs:
        from whycast_transcribe.postprocess.formatter import convert_existing_blogs
        convert_existing_blogs(args.input or '.')
        sys.exit(0)
    if args.regenerate_all_summaries:
        from whycast_transcribe.transcribe import regenerate_all_summaries
        regenerate_all_summaries(args.input or '.')
        sys.exit(0)
    if args.regenerate_all_cleaned:
        from whycast_transcribe.transcribe import regenerate_all_cleaned
        regenerate_all_cleaned(args.input or '.')
        sys.exit(0)
    if args.regenerate_all_blogs:
        from whycast_transcribe.transcribe import regenerate_all_blogs
        regenerate_all_blogs(args.input or '.')
        sys.exit(0)
    if args.regenerate_all_history:
        from whycast_transcribe.transcribe import regenerate_all_history_extractions
        regenerate_all_history_extractions(args.input or '.', force=args.force)
        sys.exit(0)
    if args.regenerate_full_workflow:
        from whycast_transcribe.transcribe import regenerate_all_full_workflow
        regenerate_all_full_workflow(args.input)
        sys.exit(0)

    # Input required for core processing
    if not args.input:
        parser.print_help()
        sys.exit(1)

    # Main transcription flow or specific commands
    if args.regenerate_cleaned:
        from whycast_transcribe.transcribe import regenerate_cleaned_transcript
        success = regenerate_cleaned_transcript(args.input)
        sys.exit(0 if success else 1)
    if args.regenerate_summary:
        from whycast_transcribe.transcribe import regenerate_summary
        success = regenerate_summary(args.input)
        sys.exit(0 if success else 1)
    if args.generate_history:
        from whycast_transcribe.postprocess.history_generator import generate_history
        from whycast_transcribe.postprocess.formatter import write_workflow_outputs
        
        # Determine the transcript file path
        if args.input.lower().endswith('.txt'):
            transcript_file = args.input
        else:
            # Input is an audio file, determine corresponding transcript file
            base = os.path.splitext(args.input)[0]
            transcript_file = f"{base}.txt"
        
        # Check if the transcript file exists
        if not os.path.exists(transcript_file):
            logging.error(f"Transcript file not found: {transcript_file}")
            print(f"Error: Transcript file does not exist: {transcript_file}")
            print("Please transcribe the file first before generating history.")
            sys.exit(1)
            
        # Load the transcript text from file
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Error reading transcript file: {e}")
            print(f"Error reading transcript file: {e}")
            sys.exit(1)
        
        # Generate history from transcript
        history = generate_history(text)
        base_filename = os.path.splitext(transcript_file)[0]
        
        # Use the formatter to write all file formats (txt, html, wiki)
        write_workflow_outputs(base_filename, history=history, force=args.force)
        print(f"History extraction generated: {base_filename}_history.txt (with HTML and Wiki versions)")
        
        sys.exit(0)

    # Batch or all-mp3s
    if args.batch:
        from whycast_transcribe.transcribe import process_batch
        process_batch(args.input, model=model)
        sys.exit(0)
    if args.all_mp3s:
        from whycast_transcribe.transcribe import process_all_mp3s
        process_all_mp3s(args.input, model=model)
        sys.exit(0)

    # Single-file transcription - directly use the updated transcribe_audio function
    segments, info, speaker_segments = transcribe_audio(
        model, 
        args.input, 
        diarization_pipeline,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )
    base = os.path.splitext(args.input)[0]
    write_transcript_files(segments, f"{base}.txt", f"{base}_ts.txt", speaker_segments)

    # Calculate and log audio duration
    duration = get_audio_duration(args.input)
    if (duration):
        logging.info(f"Audio duration: {duration:.2f} seconds")
    else:
        logging.warning("Could not determine audio duration.")

    # Post-processing
    if not args.skip_summary:
        run_postprocessing(segments, args.input)

    logging.info("WHYcast Transcribe finished")
    return 0

if __name__ == '__main__':
    sys.exit(main())
