#!/usr/bin/env python3
"""
Test script for the new programmatic speaker assignment functionality
"""

import os
import sys

# Add the main directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_programmatic_assignment():
    """Test the programmatic speaker assignment functionality."""
    
    try:
        from transcribe import (
            apply_speaker_mapping_programmatically, 
            handle_unknown_speakers,
            speaker_assignment_programmatic
        )
    except ImportError as e:
        print(f"❌ Error importing functions: {e}")
        return
    
    # Test transcript
    test_transcript = """[SPEAKER_00] Hi and welcome to the WHYcast episode 45. I'm Nancy.

[SPEAKER_01] Thanks for having me on the show today.

[SPEAKER_00] Today we're discussing AI developments and what they mean for the future.

[SPEAKER_UNKNOWN] This is an unknown speaker comment.

[SPEAKER_01] That sounds fascinating. Can you tell us more about the technical details?

[SPEAKER_02] As an expert in this field, I can share some insights.

[SPEAKER_00] Perfect! Let's dive deeper into that topic."""

    # Test mapping
    test_mapping = {
        "[SPEAKER_00]": "Nancy",
        "[SPEAKER_01]": "Guest",
        "[SPEAKER_02]": "Expert"
    }
    
    print("🧪 Testing Programmatic Speaker Assignment")
    print("=" * 60)
    print("Input transcript:")
    print(test_transcript)
    print(f"\nMapping: {test_mapping}")
    
    # Test basic mapping
    print("\n🔧 Testing basic speaker mapping...")
    mapped_text = apply_speaker_mapping_programmatically(test_transcript, test_mapping)
    print("Result after mapping:")
    print(mapped_text)
    
    # Test unknown speaker handling
    print("\n🔧 Testing unknown speaker handling...")
    final_text = handle_unknown_speakers(mapped_text)
    print("Result after handling unknowns:")
    print(final_text)
    
    # Test full programmatic assignment
    print("\n🔧 Testing full programmatic assignment...")
    try:
        result = speaker_assignment_programmatic(test_transcript, test_mapping, "test_episode", "podcasts")
        if result:
            print("\n✅ Full programmatic assignment successful!")
            print("First 300 characters of result:")
            print(result[:300] + "...")
            
            # Check if files were created
            output_files = [
                "podcasts/test_episode_speaker_assignment.txt",
                "podcasts/test_episode_speaker_assignment.html",
                "podcasts/test_episode_speaker_assignment.wiki"
            ]
            
            print("\n📁 Checking created files:")
            for file_path in output_files:
                if os.path.exists(file_path):
                    print(f"   ✅ {file_path}")
                else:
                    print(f"   ❌ {file_path}")
        else:
            print("❌ Full programmatic assignment failed")
            
    except Exception as e:
        print(f"❌ Error in full assignment: {e}")
    
    print("\n🎯 Test completed!")

def test_episode_45_assignment():
    """Test speaker assignment on actual episode 45 data."""
    
    try:
        from transcribe import speaker_assignment_step
    except ImportError as e:
        print(f"❌ Error importing speaker_assignment_step: {e}")
        return
    
    # Check if episode 45 transcript exists
    transcript_file = "podcasts/episode_45_transcript.txt"
    if not os.path.exists(transcript_file):
        print(f"❌ {transcript_file} not found - skipping Episode 45 test")
        return
    
    print("\n🎯 Testing Episode 45 with New Programmatic Assignment")
    print("=" * 60)
    
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_content = f.read()
        
        print(f"📖 Loaded Episode 45 transcript: {len(transcript_content)} characters")
        
        # Run speaker assignment
        result = speaker_assignment_step(
            transcript_content,
            output_basename="episode_45_programmatic_test",
            output_dir="podcasts"
        )
        
        if result:
            print("✅ Episode 45 programmatic assignment successful!")
            print(f"Result length: {len(result)} characters")
            
            # Check if it has headers and footers
            if result.startswith("== Disclaimer =="):
                print("✅ Header found")
            else:
                print("❌ Header missing")
                
            if "[[Category:WHYcast]]" in result:
                print("✅ Footer found")
            else:
                print("❌ Footer missing")
                
            # Check speaker labels
            if "[SPEAKER_" not in result:
                print("✅ All speaker tags replaced")
            else:
                remaining = result.count("[SPEAKER_")
                print(f"⚠️ {remaining} speaker tags still remain")
                
        else:
            print("❌ Episode 45 assignment failed")
            
    except Exception as e:
        print(f"❌ Error testing Episode 45: {e}")

if __name__ == "__main__":
    # Run basic tests
    test_programmatic_assignment()
    
    # Test on real data if available
    test_episode_45_assignment()
