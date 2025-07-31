#!/usr/bin/env python3
"""
Test script to run speaker assignment on episode 45 with header/footer integration
"""

import os
import sys
from transcribe import speaker_assignment_step

def test_ep45():
    # Read the episode 45 transcript
    transcript_file = "podcasts/episode_45_transcript.txt"
    
    print(f"Reading transcript from: {transcript_file}")
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_content = f.read()
    
    print(f"Transcript length: {len(transcript_content)} characters")
    print(f"First 200 characters: {transcript_content[:200]}")
    
    # Run speaker assignment with header/footer integration
    print("\nRunning speaker assignment with header/footer integration...")
    result = speaker_assignment_step(
        transcript=transcript_content,
        output_basename="ep45_test_headers",
        output_dir="podcasts"
    )
    
    if result:
        print(f"\n✅ Success! Generated files with base name: ep45_test_headers")
        print(f"Result length: {len(result)} characters")
        
        # Check if the files were created and show first part of output
        output_files = [
            "podcasts/ep45_test_headers_speaker_assignment.txt",
            "podcasts/ep45_test_headers_speaker_assignment.html",
            "podcasts/ep45_test_headers_speaker_assignment.wiki"
        ]
        
        for file_path in output_files:
            if os.path.exists(file_path):
                print(f"\n📄 Created: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"   Length: {len(content)} characters")
                    print(f"   First 300 characters:")
                    print(f"   {content[:300]}...")
            else:
                print(f"\n❌ Missing: {file_path}")
    else:
        print("\n❌ Speaker assignment failed!")

if __name__ == "__main__":
    test_ep45()
