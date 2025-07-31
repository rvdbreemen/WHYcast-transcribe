#!/usr/bin/env python3

"""
Quick Batch Update Script - Simple version for immediate use

This script quickly updates all episodes that need speaker assignment
with minimal configuration and maximum simplicity.
"""

import os
import glob
import sys
import time

# Add the main directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_update_all():
    """Quick update all episodes that need speaker assignment."""
    
    try:
        from transcribe import speaker_assignment_step
    except ImportError:
        print("❌ Error: Cannot import transcribe module")
        print("Make sure you're running this script from the WHYcast-transcribe directory")
        return
    
    print("🎯 Quick Batch Speaker Assignment Update")
    print("=" * 40)
    
    # Find transcript files
    transcript_files = []
    patterns = ['podcasts/*_merged.txt', 'podcasts/*_transcript.txt', 'podcasts/episode_*.txt']
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file_path in files:
            basename = os.path.splitext(os.path.basename(file_path))[0]
            
            # Clean up basename
            if basename.endswith('_merged'):
                output_basename = basename[:-7]
            elif basename.endswith('_transcript'):
                output_basename = basename[:-11]
            else:
                output_basename = basename
            
            # Skip if already processed or if it's already a speaker assignment file
            if '_speaker_assignment' in basename:
                continue
                
            # Check if assignment file already exists
            assignment_file = f"podcasts/{output_basename}_speaker_assignment.txt"
            if not os.path.exists(assignment_file):
                transcript_files.append((file_path, output_basename))
    
    # Remove duplicates (in case same episode found in multiple patterns)
    seen = set()
    unique_files = []
    for file_path, basename in transcript_files:
        if basename not in seen:
            seen.add(basename)
            unique_files.append((file_path, basename))
    
    transcript_files = unique_files
    
    if not transcript_files:
        print("✅ All episodes already have speaker assignments!")
        return
    
    print(f"📁 Found {len(transcript_files)} episodes needing speaker assignment:")
    for _, basename in transcript_files:
        print(f"   • {basename}")
    
    print(f"\n🚀 Starting batch processing...")
    
    success_count = 0
    failed_count = 0
    
    for i, (file_path, basename) in enumerate(transcript_files, 1):
        print(f"\n📊 Processing {i}/{len(transcript_files)}: {basename}")
        
        try:
            # Read transcript content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it has speaker labels
            if '[SPEAKER_' not in content:
                print(f"⏭️  No speaker labels found - skipping")
                continue
            
            print(f"🔄 Running speaker assignment...")
            
            # Run speaker assignment
            result = speaker_assignment_step(content, basename, "podcasts")
            
            if result:
                print(f"✅ Success: {basename}")
                success_count += 1
                
                # Small delay to be nice to the API
                time.sleep(1)
            else:
                print(f"❌ Failed: {basename}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Error processing {basename}: {e}")
            failed_count += 1
    
    # Summary
    print(f"\n" + "=" * 40)
    print(f"📈 SUMMARY")
    print(f"=" * 40)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📋 Total: {len(transcript_files)}")
    
    if success_count > 0:
        print(f"\n🎉 Successfully updated {success_count} episodes with:")
        print("   • Clean speaker labels (Nancy:, Host:, etc.)")
        print("   • Disclaimer headers")
        print("   • WikiMedia footers")
        print("   • Proper episode numbering")

def update_specific_episode(episode_name):
    """Update a specific episode by name."""
    
    try:
        from transcribe import speaker_assignment_step
    except ImportError:
        print("❌ Error: Cannot import transcribe module")
        return
    
    # Find the transcript file
    possible_files = [
        f"podcasts/{episode_name}_merged.txt",
        f"podcasts/{episode_name}_transcript.txt", 
        f"podcasts/{episode_name}.txt"
    ]
    
    transcript_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            transcript_file = file_path
            break
    
    if not transcript_file:
        print(f"❌ Could not find transcript file for {episode_name}")
        print(f"Looked for: {', '.join(possible_files)}")
        return
    
    print(f"🎯 Updating episode: {episode_name}")
    print(f"📖 Using transcript: {transcript_file}")
    
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '[SPEAKER_' not in content:
            print(f"⚠️  No speaker labels found in {episode_name}")
            return
        
        print(f"🔄 Running speaker assignment...")
        result = speaker_assignment_step(content, episode_name, "podcasts")
        
        if result:
            print(f"✅ Successfully updated {episode_name}!")
            print("   • Check podcasts/{episode_name}_speaker_assignment.txt")
        else:
            print(f"❌ Failed to update {episode_name}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function with simple command line handling."""
    
    if len(sys.argv) > 1:
        # Update specific episode
        episode_name = sys.argv[1]
        update_specific_episode(episode_name)
    else:
        # Update all episodes
        quick_update_all()

if __name__ == "__main__":
    main()
