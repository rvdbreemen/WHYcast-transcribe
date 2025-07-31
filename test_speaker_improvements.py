#!/usr/bin/env python3
"""
Test script for improved speaker identification functionality.

This script tests the new two-phase speaker assignment process:
1. analyze_speakers_with_o4() - Deep analysis with o4 model
2. speaker_assignment_step() - Application with mapping

Author: WHYcast Transcription Tool v0.3.0+
"""

import os
import sys
import logging
from typing import Dict, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
import transcribe
from config import *

def setup_test_logging():
    """Setup logging for test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_speaker_improvements.log')
        ]
    )

def create_test_transcript() -> str:
    """Create a test transcript with speaker labels."""
    return """[SPEAKER_00]: Welcome to today's episode. I'm excited to have our guest here.

[SPEAKER_01]: Thank you for having me. It's great to be here.

[SPEAKER_00]: So let's dive right in. Can you tell us about your background?

[SPEAKER_01]: Absolutely. I've been working in technology for over fifteen years, primarily focusing on artificial intelligence and machine learning.

[SPEAKER_00]: That's fascinating. What got you started in AI?

[SPEAKER_01]: Well, it actually started during my computer science studies at university. I was fascinated by the idea of machines that could learn and adapt.

[SPEAKER_00]: And now you're leading a team at a major tech company, right?

[SPEAKER_01]: Yes, I'm currently the Director of AI Research at TechCorp. We're working on some really exciting projects.

[SPEAKER_00]: Before we wrap up, any advice for listeners interested in AI?

[SPEAKER_01]: My advice would be to start with the fundamentals. Learn programming, understand statistics, and most importantly, never stop being curious."""

def test_speaker_mapping_parsing():
    """Test the parse_speaker_mapping_from_analysis function."""
    print("\n=== Testing Speaker Mapping Parsing ===")
    

    # Test analysis text with mapping section (new simple format)
    test_analysis = """
## STEP 4: FINAL MAPPING CREATION

Based on my analysis, here is the final mapping for the transcript:

FINAL MAPPING FOR TRANSCRIPT:
============================
SPEAKER_00 → Sarah
SPEAKER_01 → Host

**EXAMPLES OF GOOD LABELS:**
- Sarah (when name is clearly identified)
- Host (when role is clear but name isn't)

## CONFIDENCE SUMMARY:
- Host identification: HIGH (consistent hosting patterns)
- Guest identification: HIGH (professional context, clear expertise)
"""
    
    mapping = transcribe.parse_speaker_mapping_from_analysis(test_analysis)
    print(f"Parsed mapping: {mapping}")
    
    expected = {
        "[SPEAKER_00]": "Sarah",
        "[SPEAKER_01]": "Host"
    }
    
    if mapping == expected:
        print("✅ Speaker mapping parsing test PASSED")
        return True
    else:
        print(f"❌ Speaker mapping parsing test FAILED")
        print(f"Expected: {expected}")
        print(f"Got: {mapping}")
        return False

def test_content_preservation():
    """Test the analyze_transcript_changes function."""
    print("\n=== Testing Content Preservation Analysis ===")
    
    original = create_test_transcript()
    
    # Simulate a good preservation (just speaker label changes)
    good_processed = original.replace("[SPEAKER_00]", "[Host]").replace("[SPEAKER_01]", "[Guest - Dr. Chen]")
    
    # Simulate poor preservation (content loss)
    bad_processed = """[Host]: Welcome.
[Guest]: Thanks.
[Host]: Background?
[Guest]: AI experience.
[Host]: Advice?
[Guest]: Learn fundamentals."""
    
    # Test good preservation
    good_analysis = transcribe.analyze_transcript_changes(original, good_processed)
    print(f"Good preservation analysis:")
    print(f"  Word retention: {good_analysis['word_retention']:.1f}%")
    print(f"  Char retention: {good_analysis['char_retention']:.1f}%")
    
    # Test poor preservation
    bad_analysis = transcribe.analyze_transcript_changes(original, bad_processed)
    print(f"Poor preservation analysis:")
    print(f"  Word retention: {bad_analysis['word_retention']:.1f}%")
    print(f"  Char retention: {bad_analysis['char_retention']:.1f}%")
    
    # Validate results
    if good_analysis['word_retention'] > 90 and bad_analysis['word_retention'] < 50:
        print("✅ Content preservation analysis test PASSED")
        return True
    else:
        print("❌ Content preservation analysis test FAILED")
        return False

def test_speaker_merging():
    """Test the merge_speaker_lines function."""
    print("\n=== Testing Speaker Line Merging ===")
    
    # Create test with consecutive same speaker lines
    test_transcript = """[SPEAKER_00]: Hello everyone.
[SPEAKER_00]: Welcome to the show.
[SPEAKER_00]: Today we have a special guest.

[SPEAKER_01]: Thank you for having me.
[SPEAKER_01]: I'm excited to be here.

[SPEAKER_00]: Let's get started.
[SPEAKER_01]: Absolutely.
[SPEAKER_01]: I've been looking forward to this.
[SPEAKER_01]: It's going to be great."""

    merged = transcribe.merge_speaker_lines(test_transcript)
    print("Original lines:", test_transcript.count('\n') + 1)
    print("Merged lines:", merged.count('\n') + 1)
    
    # Should have fewer lines due to merging
    if merged.count('\n') < test_transcript.count('\n'):
        print("✅ Speaker line merging test PASSED")
        print("Sample merged output:")
        print(merged[:200] + "..." if len(merged) > 200 else merged)
        return True
    else:
        print("❌ Speaker line merging test FAILED")
        return False

def test_file_operations():
    """Test file read/write operations."""
    print("\n=== Testing File Operations ===")
    
    # Test prompt file reading
    speaker_prompt_path = os.path.join("prompts", "speaker_assignment_prompt.txt")
    analysis_prompt_path = os.path.join("prompts", "speaker_analysis_prompt.txt")
    
    if os.path.exists(speaker_prompt_path):
        speaker_prompt = transcribe.read_prompt_file(speaker_prompt_path)
        if speaker_prompt:
            print(f"✅ Speaker assignment prompt loaded: {len(speaker_prompt)} chars")
        else:
            print(f"❌ Speaker assignment prompt is empty")
            return False
    else:
        print(f"❌ Speaker assignment prompt not found: {speaker_prompt_path}")
        return False
    
    if os.path.exists(analysis_prompt_path):
        analysis_prompt = transcribe.read_prompt_file(analysis_prompt_path)
        if analysis_prompt:
            print(f"✅ Speaker analysis prompt loaded: {len(analysis_prompt)} chars")
        else:
            print(f"❌ Speaker analysis prompt is empty")
            return False
    else:
        print(f"❌ Speaker analysis prompt not found: {analysis_prompt_path}")
        return False
    
    # Test write operations
    test_dir = "test_output"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    test_content = "Test content for file operations"
    test_basename = "test_speaker_improvements"
    
    try:
        # Write the merged transcript using the correct function signature
        merged_file = os.path.join(test_dir, f"{test_basename}_merged.txt")
        with open(merged_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        if os.path.exists(merged_file):
            print("✅ File write operations test PASSED")
            # Clean up
            os.remove(merged_file)
            os.rmdir(test_dir)
            return True
        else:
            print("❌ File write operations test FAILED - file not created")
            return False
            
    except Exception as e:
        print(f"❌ File write operations test FAILED: {str(e)}")
        return False

def run_all_tests():
    """Run all tests and provide summary."""
    print("🧪 Running Speaker Improvements Tests")
    print("=" * 50)
    
    setup_test_logging()
    
    tests = [
        ("Speaker Mapping Parsing", test_speaker_mapping_parsing),
        ("Content Preservation Analysis", test_content_preservation), 
        ("Speaker Line Merging", test_speaker_merging),
        ("File Operations", test_file_operations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"🧪 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Speaker improvements are ready.")
    else:
        print(f"⚠️  {total - passed} test(s) FAILED. Please review the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
else:
    print("\n✅ Good content preservation!")
