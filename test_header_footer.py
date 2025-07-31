#!/usr/bin/env python3
"""
Test script to verify the header/footer integration works correctly.
"""

import sys
import os

# Add current directory to path to import modules
sys.path.insert(0, '.')

try:
    from transcript_formatter import format_transcript_with_headers
    from transcribe import speaker_assignment_step
    
    def test_header_footer_integration():
        print("=== TESTING HEADER/FOOTER INTEGRATION ===")
        
        # Simple test content
        test_content = '''[SPEAKER_00] Welcome back to episode 29.
[SPEAKER_01] Thanks for having me on.
[SPEAKER_UNKNOWN] This is really interesting.'''

        print("Input transcript:")
        print(test_content)
        print()

        # Test the complete pipeline
        result = speaker_assignment_step(test_content, 'ep29_test', 'podcasts')
        
        if result:
            print("=== SUCCESS ===")
            print("First 300 characters:")
            print(result[:300])
            print("...")
            print("Last 200 characters:")  
            print(result[-200:])
            
            # Check for expected elements
            has_disclaimer = "== Disclaimer ==" in result
            has_transcript_header = "== Transcript 29 ==" in result
            has_category = "[[Category:WHYcast]]" in result
            has_episode_tag = "[[Episode number::29|" in result
            
            print(f"\\n=== VALIDATION ===")
            print(f"Has disclaimer header: {has_disclaimer}")
            print(f"Has transcript header: {has_transcript_header}")
            print(f"Has category footer: {has_category}")
            print(f"Has episode number: {has_episode_tag}")
            
            if all([has_disclaimer, has_transcript_header, has_category, has_episode_tag]):
                print("✅ ALL CHECKS PASSED - Headers and footers working correctly!")
            else:
                print("❌ Some checks failed")
                
        else:
            print("❌ Speaker assignment failed")

    if __name__ == "__main__":
        test_header_footer_integration()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
