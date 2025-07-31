#!/usr/bin/env python3
"""
Test script to test the header/footer integration on episode 45 transcript
"""

import os
from transcript_formatter import extract_episode_number, format_transcript_with_headers

def test_ep45_headers():
    # Read the episode 45 transcript
    transcript_file = "podcasts/episode_45_transcript.txt"
    
    print(f"Reading transcript from: {transcript_file}")
    
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript_content = f.read()
    
    print(f"Transcript length: {len(transcript_content)} characters")
    
    # Test episode extraction
    episode_num = extract_episode_number(transcript_file)
    print(f"Extracted episode number: {episode_num}")
    
    # Create a simple mock transcript with speaker names instead of SPEAKER_XX
    # Just to simulate what would happen after speaker assignment
    mock_assigned = transcript_content.replace("[SPEAKER_01]", "Nancy:").replace("[SPEAKER_04]", "Ad:").replace("[SPEAKER_UNKNOWN]", "Host:")
    
    print(f"\nFirst 300 characters of mock assigned transcript:")
    print(mock_assigned[:300])
    
    # Test header/footer formatting
    print(f"\n🔄 Applying header/footer formatting...")
    formatted_transcript = format_transcript_with_headers(mock_assigned, episode_num)
    
    # Save the result
    output_file = "podcasts/ep45_test_formatted.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_transcript)
    
    print(f"✅ Saved formatted transcript to: {output_file}")
    print(f"Formatted transcript length: {len(formatted_transcript)} characters")
    
    # Show the first part with headers
    print(f"\n📄 First 500 characters of formatted output:")
    print(formatted_transcript[:500])
    
    # Show the last part with footer  
    print(f"\n📄 Last 300 characters of formatted output:")
    print(formatted_transcript[-300:])

if __name__ == "__main__":
    test_ep45_headers()
