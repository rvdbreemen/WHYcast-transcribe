#!/usr/bin/env python3
"""
Test script to verify the speaker assignment fix works correctly.
This tests that the output is clean without extra headers/footers.
"""

from transcribe import speaker_assignment_step

def test_clean_speaker_assignment():
    print("=== TESTING CLEAN SPEAKER ASSIGNMENT ===")
    
    test_content = '''[SPEAKER_00] Welcome back to the show everyone.
[SPEAKER_01] Thanks for having me on.
[SPEAKER_UNKNOWN] I think this is fascinating.'''

    print("Input transcript:")
    print(test_content)
    print(f"Input stats: {len(test_content.split())} words, {len(test_content)} chars")
    print()

    result = speaker_assignment_step(test_content, 'clean_test', 'podcasts')
    
    if result:
        print("=== RESULT ===")
        print(result)
        print(f"Output stats: {len(result.split())} words, {len(result)} chars")
        print()
        
        # Check if result contains unwanted content
        unwanted_patterns = [
            "== Disclaimer ==",
            "[[Category:WHYcast]]",
            "This is the full transcript generated",
            "== Transcript"
        ]
        
        has_unwanted = any(pattern in result for pattern in unwanted_patterns)
        
        if has_unwanted:
            print("❌ FAILED: Result contains unwanted headers/footers")
            for pattern in unwanted_patterns:
                if pattern in result:
                    print(f"   Found: {pattern}")
        else:
            print("✅ SUCCESS: Clean result without extra content")
            
            # Check content retention
            input_words = len(test_content.split())
            output_words = len(result.split())
            retention = (output_words / input_words) * 100 if input_words > 0 else 100
            
            if 95 <= retention <= 105:  # Allow 5% variance for speaker label changes
                print(f"✅ Good word retention: {retention:.1f}%")
            else:
                print(f"⚠️ Unusual word retention: {retention:.1f}%")
        
    else:
        print("❌ Speaker assignment failed")

if __name__ == "__main__":
    test_clean_speaker_assignment()
