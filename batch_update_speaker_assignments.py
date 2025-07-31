#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch Speaker Assignment Update Script

This script regenerates speaker assignment files for all episodes with:
- New header/footer integration 
- Improved speaker identification using o4 model
- Content preservation monitoring
- Progress tracking and error handling
"""

import os
import sys
import glob
import logging
import time
from datetime import datetime
from pathlib import Path

# Add the main directory to Python path to import transcribe module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from transcribe import (
        speaker_assignment_step, 
        analyze_transcript_changes,
        setup_logging
    )
except ImportError as e:
    print(f"❌ Error importing transcribe module: {e}")
    print("Make sure you're running this script from the WHYcast-transcribe directory")
    sys.exit(1)

def find_episode_transcripts(directory: str = "podcasts") -> list:
    """
    Find all episode transcript files that need speaker assignment.
    
    Returns:
        List of tuples: (transcript_file, output_basename, episode_number)
    """
    transcript_files = []
    
    # Look for merged transcript files first (preferred), then regular transcripts
    patterns = [
        f"{directory}/*_merged.txt",
        f"{directory}/*_transcript.txt"
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file_path in files:
            basename = os.path.splitext(os.path.basename(file_path))[0]
            
            # Remove _merged or _transcript suffix to get base name
            if basename.endswith('_merged'):
                output_basename = basename[:-7]  # Remove '_merged'
            elif basename.endswith('_transcript'):
                output_basename = basename[:-11]  # Remove '_transcript'
            else:
                output_basename = basename
            
            # Extract episode number for sorting
            episode_num = extract_episode_number(output_basename)
            
            transcript_files.append((file_path, output_basename, episode_num))
    
    # Sort by episode number
    transcript_files.sort(key=lambda x: x[2] if x[2] else 0)
    
    return transcript_files

def extract_episode_number(basename: str) -> int:
    """Extract episode number from basename for sorting."""
    import re
    
    # Try various patterns: episode_45, ep45, Episode_35, etc.
    patterns = [
        r'episode[_\s]*(\d+)',
        r'ep[_\s]*(\d+)', 
        r'(\d+)',  # Just numbers
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename.lower())
        if match:
            return int(match.group(1))
    
    return 0  # Default for non-numeric episodes

def batch_update_speaker_assignments(
    directory: str = "podcasts",
    force_update: bool = False,
    dry_run: bool = False,
    specific_episodes: list = None
):
    """
    Batch update speaker assignments for all episodes.
    
    Args:
        directory: Directory containing transcript files
        force_update: Update even if speaker assignment file already exists
        dry_run: Show what would be processed without actually doing it
        specific_episodes: List of specific episode basenames to process
    """
    
    # Setup logging
    log_file = f"batch_speaker_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    try:
        setup_logging(log_file)
    except:
        # Fallback logging setup if the function doesn't exist
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    print("🎯 WHYcast Batch Speaker Assignment Update")
    print("=" * 50)
    
    # Find all transcript files
    transcript_files = find_episode_transcripts(directory)
    
    if not transcript_files:
        print(f"❌ No transcript files found in {directory}")
        return
    
    # Filter for specific episodes if requested
    if specific_episodes:
        transcript_files = [
            (file_path, basename, ep_num) 
            for file_path, basename, ep_num in transcript_files
            if basename in specific_episodes
        ]
    
    print(f"📁 Found {len(transcript_files)} transcript files")
    
    if dry_run:
        print("\n🔍 DRY RUN - Would process these files:")
        for file_path, basename, ep_num in transcript_files:
            assignment_file = os.path.join(directory, f"{basename}_speaker_assignment.txt")
            exists = "✅ EXISTS" if os.path.exists(assignment_file) else "❌ MISSING"
            print(f"  Episode {ep_num:2d}: {basename} - {exists}")
        return
    
    # Process each transcript
    results = {
        'success': [],
        'failed': [],
        'skipped': [],
        'errors': []
    }
    
    for i, (file_path, basename, ep_num) in enumerate(transcript_files, 1):
        print(f"\n📊 Processing {i}/{len(transcript_files)}: Episode {ep_num} ({basename})")
        
        # Check if speaker assignment already exists
        assignment_file = os.path.join(directory, f"{basename}_speaker_assignment.txt")
        
        if os.path.exists(assignment_file) and not force_update:
            print(f"⏭️  Skipping - speaker assignment already exists (use --force to update)")
            results['skipped'].append(basename)
            continue
        
        try:
            # Read transcript file
            print(f"📖 Reading: {os.path.basename(file_path)}")
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_content = f.read()
            
            if not transcript_content.strip():
                print(f"⚠️  Empty transcript file")
                results['failed'].append(f"{basename} - Empty file")
                continue
            
            # Check if transcript has speaker labels
            if '[SPEAKER_' not in transcript_content:
                print(f"⚠️  No speaker labels found - skipping")
                results['skipped'].append(f"{basename} - No speaker labels")
                continue
            
            print(f"🔄 Running speaker assignment...")
            
            # Run speaker assignment
            result = speaker_assignment_step(
                transcript_content,
                output_basename=basename,
                output_dir=directory
            )
            
            if result:
                # Try to analyze changes if function exists
                try:
                    changes = analyze_transcript_changes(transcript_content, result)
                    retention = changes['word_retention']
                    print(f"✅ Success! Content retention: {retention:.1f}%")
                    
                    if retention < 90:
                        print(f"⚠️  Warning: Lower than expected retention")
                    
                    results['success'].append(f"{basename} ({retention:.1f}% retention)")
                except:
                    # Fallback if analyze_transcript_changes doesn't exist
                    print(f"✅ Success!")
                    results['success'].append(basename)
                
                # Add small delay to avoid overwhelming the API
                time.sleep(2)
                
            else:
                print(f"❌ Failed - no result returned")
                results['failed'].append(f"{basename} - No result")
        
        except Exception as e:
            error_msg = f"{basename} - {str(e)}"
            print(f"❌ Error: {str(e)}")
            results['errors'].append(error_msg)
            logging.error(f"Error processing {basename}: {str(e)}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📈 BATCH UPDATE SUMMARY")
    print("=" * 50)
    
    print(f"✅ Successfully processed: {len(results['success'])}")
    for item in results['success']:
        print(f"   • {item}")
    
    print(f"\n⏭️  Skipped: {len(results['skipped'])}")
    for item in results['skipped']:
        print(f"   • {item}")
    
    print(f"\n❌ Failed: {len(results['failed'])}")
    for item in results['failed']:
        print(f"   • {item}")
    
    print(f"\n🚨 Errors: {len(results['errors'])}")
    for item in results['errors']:
        print(f"   • {item}")
    
    print(f"\n📋 Total processed: {len(transcript_files)}")
    print(f"📋 Success rate: {len(results['success'])}/{len(transcript_files)} ({len(results['success'])/len(transcript_files)*100:.1f}%)")
    
    print(f"\n📁 Log file: {log_file}")

def main():
    """Main function with command line argument handling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch update speaker assignments for WHYcast episodes"
    )
    
    parser.add_argument(
        '--directory', '-d',
        default='podcasts',
        help='Directory containing transcript files (default: podcasts)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force update even if speaker assignment files already exist'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be processed without actually doing it'
    )
    
    parser.add_argument(
        '--episodes', '-e',
        nargs='+',
        help='Process only specific episodes (e.g., episode_45 ep29)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available episodes and their status'
    )
    
    args = parser.parse_args()
    
    if args.list:
        # List all episodes and their status
        transcript_files = find_episode_transcripts(args.directory)
        print(f"📁 Episodes in {args.directory}:")
        print("-" * 60)
        
        for file_path, basename, ep_num in transcript_files:
            assignment_file = os.path.join(args.directory, f"{basename}_speaker_assignment.txt")
            status = "✅ HAS ASSIGNMENT" if os.path.exists(assignment_file) else "❌ NEEDS ASSIGNMENT"
            print(f"Episode {ep_num:2d}: {basename:25s} {status}")
        
        return
    
    # Run batch update
    batch_update_speaker_assignments(
        directory=args.directory,
        force_update=args.force,
        dry_run=args.dry_run,
        specific_episodes=args.episodes
    )

if __name__ == "__main__":
    main()
