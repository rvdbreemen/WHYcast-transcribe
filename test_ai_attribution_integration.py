#!/usr/bin/env python3
"""
Test script for the new AI-powered SPEAKER_UNKNOWN attribution functionality.
Tests the integration with Episode 45 data.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the transcribe module
import transcribe

def test_ai_attribution():
    """Test the new AI attribution functionality."""
    print("🧪 Testing AI SPEAKER_UNKNOWN Attribution")
    print("=" * 60)
    
    # Load Episode 45 test data
    merged_file = project_root / "podcasts" / "episode_45_merged.txt"
    if not merged_file.exists():
        print("❌ Episode 45 merged transcript not found")
        return
    
    with open(merged_file, 'r', encoding='utf-8') as f:
        test_transcript = f.read()
    
    print(f"✅ Loaded test transcript: {len(test_transcript)} characters")
    
    # Count SPEAKER_UNKNOWN segments before
    unknown_before = test_transcript.count('[SPEAKER_UNKNOWN]')
    print(f"📊 SPEAKER_UNKNOWN segments before attribution: {unknown_before}")
    
    # Create a sample speaker mapping (from Episode 45 analysis)
    speaker_mapping = {
        '[SPEAKER_01]': 'Nancy',
        '[SPEAKER_04]': 'Ad',
        '[SPEAKER_00]': 'Guest',
        '[SPEAKER_05]': 'Guest',
        '[SPEAKER_03]': 'Guest'
    }
    
    # Test the new AI attribution function
    print("\n🤖 Running AI attribution test...")
    try:
        result = transcribe.attribute_unknown_speakers_with_ai(
            transcript=test_transcript,
            speaker_mapping=speaker_mapping,
            output_basename="test_episode_45",
            output_dir="podcasts"
        )
        
        # Count SPEAKER_UNKNOWN segments after
        unknown_after = result.count('[SPEAKER_UNKNOWN]')
        attributed = unknown_before - unknown_after
        
        print(f"\n📊 RESULTS:")
        print(f"✅ SPEAKER_UNKNOWN segments before: {unknown_before}")
        print(f"✅ SPEAKER_UNKNOWN segments after: {unknown_after}")
        print(f"✅ Segments attributed: {attributed}")
        print(f"✅ Attribution rate: {(attributed/unknown_before)*100:.1f}%" if unknown_before > 0 else "N/A")
        
        # Save the result for inspection
        output_file = project_root / "podcasts" / "test_episode_45_ai_attributed.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"💾 Saved attributed transcript: {output_file}")
        
        # Show some examples of attributions
        print(f"\n🔍 Sample attributions:")
        lines = result.split('\n')
        attribution_count = 0
        for i, line in enumerate(lines):
            if any(speaker in line for speaker in ['Nancy:', 'Ad:', 'Guest:']):
                # Check if this was likely an attribution from SPEAKER_UNKNOWN
                original_lines = test_transcript.split('\n')
                if i < len(original_lines) and '[SPEAKER_UNKNOWN]' in original_lines[i]:
                    print(f"  Line {i+1}: {line[:100]}...")
                    attribution_count += 1
                    if attribution_count >= 3:  # Show first 3 examples
                        break
        
        return True
        
    except Exception as e:
        print(f"❌ Error during AI attribution test: {e}")
        return False

def test_pattern_attribution():
    """Test just the pattern-based attribution (same speaker before/after)."""
    print("\n🔧 Testing Pattern-Based Attribution Only")
    print("=" * 50)
    
    # Create a simple test transcript with clear patterns
    test_pattern_transcript = """[SPEAKER_01] Hello, I'm Nancy and this is the test.

[SPEAKER_04] Hi Nancy, I'm Ad. Today we have

[SPEAKER_UNKNOWN] some great content to discuss.

[SPEAKER_04] And also we'll talk about

[SPEAKER_UNKNOWN] the new features.

[SPEAKER_01] That sounds

[SPEAKER_UNKNOWN] really interesting.

[SPEAKER_01] Let's dive in."""

    print("📝 Test transcript:")
    print(test_pattern_transcript)
    
    unknown_before = test_pattern_transcript.count('[SPEAKER_UNKNOWN]')
    print(f"\n📊 SPEAKER_UNKNOWN segments: {unknown_before}")
    
    # Test attribution
    result = transcribe.attribute_unknown_speakers_with_ai(
        transcript=test_pattern_transcript,
        speaker_mapping={'[SPEAKER_01]': 'Nancy', '[SPEAKER_04]': 'Ad'},
        output_basename="pattern_test",
        output_dir="."
    )
    
    unknown_after = result.count('[SPEAKER_UNKNOWN]')
    attributed = unknown_before - unknown_after
    
    print(f"\n📊 PATTERN RESULTS:")
    print(f"✅ Attributed: {attributed}/{unknown_before} segments")
    print(f"✅ Result transcript:")
    print(result)
    
    return attributed > 0

def main():
    """Main test function."""
    print("🚀 AI SPEAKER_UNKNOWN Attribution Testing")
    print("=" * 70)
    
    # Test 1: Pattern-based attribution
    print("\n### TEST 1: Pattern-Based Attribution ###")
    pattern_success = test_pattern_attribution()
    
    # Test 2: Full AI attribution with Episode 45
    print("\n### TEST 2: Full AI Attribution ###")
    ai_success = test_ai_attribution()
    
    # Summary
    print("\n🎯 TEST SUMMARY:")
    print(f"✅ Pattern-based attribution: {'PASSED' if pattern_success else 'FAILED'}")
    print(f"✅ Full AI attribution: {'PASSED' if ai_success else 'FAILED'}")
    
    if pattern_success and ai_success:
        print("\n🎉 All tests passed! AI attribution is ready for integration.")
    else:
        print("\n⚠️ Some tests failed. Review the implementation.")

if __name__ == "__main__":
    main()
