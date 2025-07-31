#!/usr/bin/env python3
"""
Manual demonstration of header/footer integration on episode 45
"""

from transcript_formatter import extract_episode_number, format_transcript_with_headers

def manually_format_ep45():
    # Read the existing episode 45 speaker assignment (without headers/footers)
    original_file = "podcasts/episode_45_speaker_assignment.txt"
    
    print(f"📖 Reading existing file: {original_file}")
    
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Original content length: {len(content)} characters")
    print(f"First 100 characters: {content[:100]}...")
    print(f"Last 100 characters: ...{content[-100:]}")
    
    # Extract episode number from filename 
    episode_num = extract_episode_number("episode_45")
    print(f"📊 Extracted episode number: {episode_num}")
    
    # Apply header/footer formatting
    print(f"🔄 Applying header/footer formatting...")
    formatted_content = format_transcript_with_headers(content, episode_num)
    
    # Save the formatted version
    output_file = "podcasts/episode_45_WITH_HEADERS.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_content)
    
    print(f"✅ Saved formatted version to: {output_file}")
    print(f"Formatted content length: {len(formatted_content)} characters")
    
    # Show the difference
    print(f"\n🎯 BEFORE vs AFTER COMPARISON:")
    print("=" * 60)
    print("🔴 ORIGINAL (first 200 chars):")
    print(content[:200])
    print("\n🟢 FORMATTED (first 400 chars):")
    print(formatted_content[:400])
    print("\n🔴 ORIGINAL (last 100 chars):")
    print("..." + content[-100:])
    print("\n🟢 FORMATTED (last 200 chars):")
    print("..." + formatted_content[-200:])

if __name__ == "__main__":
    manually_format_ep45()
