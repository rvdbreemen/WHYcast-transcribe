"""
Manual AI Attribution Test for SPEAKER_UNKNOWN Segments
Using the attribution prompt to analyze specific segments from Episode 45
"""

# Sample segments from Episode 45 for AI analysis
test_segments = [
    {
        "segment_id": 1,
        "line": 5,
        "content": "[SPEAKER_UNKNOWN] We have a vacancy of the week and we have listener questions. It sounds like a format.",
        "previous_speaker": "[SPEAKER_04]",
        "next_speaker": "[SPEAKER_01]",
        "context": """[SPEAKER_01] Hi and welcome to the WHYcast episode 45. I'm Nancy. I'm Ad. And we are the hosts of the only podcast about a hacker camp in the universe.

[SPEAKER_04] to 12. So Ad, what are we talking about today? Well, after 45 episodes, well actually 46 because we started counting at zero. We have the news. We have an interview. We have a where to hack.

[SPEAKER_UNKNOWN] We have a vacancy of the week and we have listener questions. It sounds like a format.

[SPEAKER_01] Yes, it's like we planned this. It's like we have an actual idea what we're doing."""
    },
    {
        "segment_id": 2,
        "line": 17,
        "content": "[SPEAKER_UNKNOWN] an angel, right? Yes.",
        "previous_speaker": "[SPEAKER_01]",
        "next_speaker": "[SPEAKER_01]",
        "context": """[SPEAKER_01] news, I think. And you also signed up as

[SPEAKER_UNKNOWN] an angel, right? Yes.

[SPEAKER_01] Of course you did. Of course. You are probably at the info desk. Wild guess."""
    },
    {
        "segment_id": 3,
        "line": 23,
        "content": "[SPEAKER_UNKNOWN] One",
        "previous_speaker": "[SPEAKER_04]",
        "next_speaker": "[SPEAKER_04]",
        "context": """[SPEAKER_04] Yes.

[SPEAKER_UNKNOWN] One

[SPEAKER_04] of my tasks will be to also facilitate angel shifts at the info desk and help the angels get started."""
    }
]

# Reference: From the episode introduction we know:
# SPEAKER_01 = Nancy
# SPEAKER_04 = Ad (based on "I'm Nancy. I'm Ad.")

print("🧪 AI ATTRIBUTION TEST SEGMENTS")
print("=" * 50)
print("Reference: SPEAKER_01 = Nancy, SPEAKER_04 = Ad")
print()

for segment in test_segments:
    print(f"SEGMENT {segment['segment_id']} (Line {segment['line']}):")
    print(f"Content: {segment['content']}")
    print(f"Previous: {segment['previous_speaker']} → Next: {segment['next_speaker']}")
    print(f"Context:\n{segment['context']}")
    print()
    print("ANALYSIS QUESTIONS:")
    print("1. Does this SPEAKER_UNKNOWN continue a thought from previous speaker?")
    print("2. Is this a response to the previous speaker?")
    print("3. Does the content/style match Nancy or Ad?")
    print("4. What's the conversational flow pattern?")
    print()
    print("EXPECTED ATTRIBUTION:")
    
    # Manual analysis for testing
    if segment['segment_id'] == 1:
        print("- This appears to be Ad (SPEAKER_04) continuing the list of show segments")
        print("- Conversational flow: Ad was listing show elements, this continues that list")
        print("- The 'It sounds like a format' is likely Ad commenting on their own structure")
        print("- ATTRIBUTION: [SPEAKER_UNKNOWN] → [SPEAKER_04] (Ad)")
    
    elif segment['segment_id'] == 2:
        print("- This appears to be Ad (SPEAKER_04) responding to Nancy's question")
        print("- Nancy said 'And you also signed up as' - this completes her sentence")
        print("- The 'Yes' response would logically come from Ad")
        print("- ATTRIBUTION: [SPEAKER_UNKNOWN] → [SPEAKER_04] (Ad)")
    
    elif segment['segment_id'] == 3:
        print("- This appears to be Ad (SPEAKER_04) continuing his own thought")
        print("- Previous SPEAKER_04 said 'Yes' and next SPEAKER_04 continues with task description")
        print("- 'One' is clearly a sentence fragment completing his response")
        print("- ATTRIBUTION: [SPEAKER_UNKNOWN] → [SPEAKER_04] (Ad)")
    
    print()
    print("-" * 50)
    print()

print("KEY PATTERNS OBSERVED:")
print("✅ SPEAKER_UNKNOWN often appears mid-sentence or as sentence fragments")
print("✅ Many segments are continuations of the same speaker (same → UNKNOWN → same)")
print("✅ Context strongly suggests which speaker is continuing their thought")
print("✅ Conversational flow provides clear attribution clues")
print()
print("NEXT STEPS:")
print("1. Use these patterns to design AI prompt test")
print("2. Validate with GPT-4o using the attribution prompt")
print("3. Implement successful approach in transcribe.py")
