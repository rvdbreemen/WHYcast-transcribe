#!/usr/bin/env python3
"""
Simple test of header/footer functionality on episode 45
"""

# Simple episode number extraction
def extract_episode_45():
    filename = "episode_45_transcript.txt"
    if "episode_45" in filename:
        return "45"
    return None

# Simple header/footer formatting
def format_with_headers(transcript, episode_num):
    header = f"""== Disclaimer ==
This transcript was generated from the WHYcast podcast audio using automated speech recognition and speaker diarization. While we strive for accuracy, there may be errors in transcription or speaker identification. The content reflects the views and opinions of the podcast participants and not necessarily those of the transcription service.

== Transcript {episode_num} ==

"""
    
    footer = f"""

[[Category:WHYcast]]
[[Category:transcription]]
[[Episode number::{episode_num}| ]]"""
    
    return header + transcript + footer

# Test with episode 45
episode_num = extract_episode_45()
print(f"Extracted episode number: {episode_num}")

# Read a small sample of the transcript
with open("podcasts/episode_45_transcript.txt", 'r', encoding='utf-8') as f:
    sample = f.read(1000)  # First 1000 characters

print(f"\nSample transcript (first 1000 chars):")
print(sample)

# Format with headers/footers
formatted = format_with_headers(sample, episode_num)

print(f"\n🎯 FORMATTED OUTPUT WITH HEADERS/FOOTERS:")
print("=" * 60)
print(formatted)
print("=" * 60)

# Save to file
output_file = "podcasts/ep45_simple_test.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(formatted)

print(f"\n✅ Saved to: {output_file}")
