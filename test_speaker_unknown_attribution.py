#!/usr/bin/env python3
"""
Test script for SPEAKER_UNKNOWN attribution using conversational flow analysis.
This tests the new AI-based approach before integration.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_test_data():
    """Load Episode 45 test data for analysis."""
    print("Loading Episode 45 test data...")
    
    # Load the merged transcript with SPEAKER_UNKNOWN segments
    merged_file = project_root / "podcasts" / "episode_45_merged.txt"
    if merged_file.exists():
        with open(merged_file, 'r', encoding='utf-8') as f:
            merged_content = f.read()
        print(f"✅ Loaded merged transcript: {len(merged_content)} characters")
    else:
        print("❌ Episode 45 merged transcript not found")
        return None
    
    # Load the speaker analysis for reference
    analysis_file = project_root / "podcasts" / "episode_45_speaker_analysis.txt"
    if analysis_file.exists():
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_content = f.read()
        print(f"✅ Loaded speaker analysis: {len(analysis_content)} characters")
    else:
        print("❌ Episode 45 speaker analysis not found")
        analysis_content = None
    
    return {
        'merged_transcript': merged_content,
        'speaker_analysis': analysis_content
    }

def extract_speaker_unknown_segments(transcript):
    """Extract all SPEAKER_UNKNOWN segments with context."""
    print("\nExtracting SPEAKER_UNKNOWN segments...")
    
    lines = transcript.split('\n')
    unknown_segments = []
    
    for i, line in enumerate(lines):
        # Look for both SPEAKER_UNKNOWN: and [SPEAKER_UNKNOWN] patterns
        if 'SPEAKER_UNKNOWN' in line and ('SPEAKER_UNKNOWN:' in line or '[SPEAKER_UNKNOWN]' in line):
            # Get context: previous 3 lines and next 3 lines
            start_idx = max(0, i - 3)
            end_idx = min(len(lines), i + 4)
            context_lines = lines[start_idx:end_idx]
            
            segment = {
                'line_number': i + 1,
                'content': line.strip(),
                'context': '\n'.join(context_lines),
                'previous_speaker': None,
                'next_speaker': None
            }
            
            # Find previous speaker (look for [SPEAKER_XX] or SPEAKER_XX: patterns)
            for j in range(i - 1, -1, -1):
                if ('[SPEAKER_' in lines[j] or 'SPEAKER_' in lines[j]) and 'SPEAKER_UNKNOWN' not in lines[j]:
                    # Extract speaker label
                    if '[SPEAKER_' in lines[j]:
                        segment['previous_speaker'] = lines[j].split(']')[0] + ']'
                    elif ':' in lines[j]:
                        segment['previous_speaker'] = lines[j].split(':')[0].strip()
                    break
            
            # Find next speaker
            for j in range(i + 1, len(lines)):
                if ('[SPEAKER_' in lines[j] or 'SPEAKER_' in lines[j]) and 'SPEAKER_UNKNOWN' not in lines[j]:
                    # Extract speaker label
                    if '[SPEAKER_' in lines[j]:
                        segment['next_speaker'] = lines[j].split(']')[0] + ']'
                    elif ':' in lines[j]:
                        segment['next_speaker'] = lines[j].split(':')[0].strip()
                    break
            
            unknown_segments.append(segment)
    
    print(f"✅ Found {len(unknown_segments)} SPEAKER_UNKNOWN segments")
    return unknown_segments

def analyze_segment_patterns(segments):
    """Analyze patterns in SPEAKER_UNKNOWN segments."""
    print("\nAnalyzing SPEAKER_UNKNOWN patterns...")
    
    # Count by position
    between_speakers = {}
    for segment in segments:
        prev = segment['previous_speaker'] or 'None'
        next_speaker = segment['next_speaker'] or 'None'
        pattern = f"{prev} → UNKNOWN → {next_speaker}"
        
        if pattern not in between_speakers:
            between_speakers[pattern] = []
        between_speakers[pattern].append(segment)
    
    print("\n📊 SPEAKER_UNKNOWN Position Patterns:")
    for pattern, segs in sorted(between_speakers.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  {pattern}: {len(segs)} occurrences")
    
    return between_speakers

def test_attribution_prompt(segments, prompt_file):
    """Test the attribution prompt on selected segments."""
    print(f"\nTesting attribution prompt from {prompt_file}...")
    
    # Load the prompt
    if not os.path.exists(prompt_file):
        print(f"❌ Prompt file not found: {prompt_file}")
        return
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read()
    
    print(f"✅ Loaded prompt: {len(prompt_content)} characters")
    
    # Test on first 3 segments as examples
    test_segments = segments[:3]
    print(f"\n🧪 Testing on {len(test_segments)} sample segments:")
    
    for i, segment in enumerate(test_segments, 1):
        print(f"\n--- Test Segment {i} ---")
        print(f"Line {segment['line_number']}: {segment['content']}")
        print(f"Previous: {segment['previous_speaker']}")
        print(f"Next: {segment['next_speaker']}")
        print(f"Context Preview:\n{segment['context'][:200]}...")
        
        # This would be where we call the AI API
        print("📝 Ready for AI analysis with attribution prompt")
    
    return test_segments

def generate_test_report(test_data, segments, patterns):
    """Generate a comprehensive test report."""
    print("\n📋 Generating test report...")
    
    report = f"""
SPEAKER_UNKNOWN ATTRIBUTION TEST REPORT
======================================
Generated: {__file__}
Episode: 45 (Test Data)

DATA SUMMARY:
- Total transcript length: {len(test_data['merged_transcript'])} characters
- SPEAKER_UNKNOWN segments found: {len(segments)}
- Unique position patterns: {len(patterns)}

TOP PATTERNS (by frequency):
"""
    
    for pattern, segs in sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        report += f"- {pattern}: {len(segs)} occurrences\n"
    
    report += f"""

SAMPLE SEGMENTS FOR AI TESTING:
"""
    
    for i, segment in enumerate(segments[:5], 1):
        report += f"""
Segment {i} (Line {segment['line_number']}):
Content: {segment['content']}
Previous: {segment['previous_speaker']} → Next: {segment['next_speaker']}
Context: {segment['context'][:150]}...

"""
    
    report += """
NEXT STEPS:
1. Use the attribution prompt with AI (GPT-4o) on these segments
2. Validate attribution accuracy against conversational flow
3. Refine prompt based on results
4. Integrate successful approach into transcribe.py

PROMPT FILE: prompts/speaker_unknown_attribution_prompt.txt
"""
    
    # Save report
    report_file = project_root / "test_speaker_unknown_attribution_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Test report saved: {report_file}")
    return report

def main():
    """Main test function."""
    print("🧪 SPEAKER_UNKNOWN Attribution Test")
    print("=" * 50)
    
    # Load test data
    test_data = load_test_data()
    if not test_data or not test_data['merged_transcript']:
        print("❌ Cannot proceed without test data")
        return
    
    # Extract SPEAKER_UNKNOWN segments
    segments = extract_speaker_unknown_segments(test_data['merged_transcript'])
    if not segments:
        print("❌ No SPEAKER_UNKNOWN segments found")
        return
    
    # Analyze patterns
    patterns = analyze_segment_patterns(segments)
    
    # Test the attribution prompt
    prompt_file = project_root / "prompts" / "speaker_unknown_attribution_prompt.txt"
    test_segments = test_attribution_prompt(segments, prompt_file)
    
    # Generate comprehensive report
    report = generate_test_report(test_data, segments, patterns)
    
    print("\n🎯 TEST SUMMARY:")
    print(f"✅ Found {len(segments)} SPEAKER_UNKNOWN segments to analyze")
    print(f"✅ Identified {len(patterns)} different conversational patterns")
    print(f"✅ Created attribution prompt ready for AI testing")
    print(f"✅ Generated test report with sample data")
    print("\n📋 Next: Use GPT-4o with the attribution prompt on sample segments")

if __name__ == "__main__":
    main()
