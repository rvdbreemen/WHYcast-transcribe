import re
from typing import Optional

def extract_episode_number(output_basename: str) -> Optional[str]:
    """
    Extract episode number from the output basename.
    
    Args:
        output_basename: The basename like 'ep29', 'episode_42', 'Episode_35', etc.
        
    Returns:
        Episode number as string or None if not found
    """
    if not output_basename:
        return None
    
    # Common episode patterns
    patterns = [
        r'^ep(\d+)',           # ep29
        r'^episode[_\s]*(\d+)', # episode_42, episode 42
        r'^Episode[_\s]*(\d+)', # Episode_35, Episode 35
        r'^E(\d+)',            # E28
        r'episode[-_](\d+)',   # episode-1, episode_1
        r'whycast.*?(\d+)',    # whycast_episode_4, Whycast33
        r'(\d+)$',             # Just a number at the end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_basename, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def format_transcript_with_headers(transcript: str, output_basename: Optional[str] = None) -> str:
    """
    Add standard header and footer to the transcript with proper episode information.
    
    Args:
        transcript: The processed transcript content
        output_basename: The basename to extract episode info from
        
    Returns:
        Formatted transcript with header and footer
    """
    episode_number = extract_episode_number(output_basename) if output_basename else None
    episode_display = episode_number if episode_number else "<<episode number>>"
    
    # Header
    header = f"""== Disclaimer ==
This is the full transcript generated using with AI tools and some human oversight. This transcript was generated using local running Whisper and Diarization, and some prompting to generate a readable transcript. The transcript was carefully transcribed, however these models do make mistakes (just like humans do). Before publishing it on the wiki read the transcribed episode to correct the obvious errors. But just as AI models, mistakes are not always correct before publishing. So you are more than welcome to correct the transcript based on the WHYcast episode out there. Please feel free to help out to make the content transcription even more accessible.

== Transcript {episode_display} ==
"""

    # Footer  
    footer = f"""
[[Category:WHYcast]] [[Category:transcription]] [[Episode number::{episode_display}| ]]"""

    return header + transcript + footer

if __name__ == "__main__":
    # Test the episode extraction
    test_cases = [
        "ep29",
        "episode_42", 
        "Episode_35",
        "E28",
        "episode-1",
        "Whycast33",
        "whycast_episode_4_-_fire_",
        "episode32_ts",
        "Episode_44",
        "fix_test"  # Should return None
    ]
    
    print("Testing episode number extraction:")
    for test in test_cases:
        result = extract_episode_number(test)
        print(f"  {test:25} → {result}")
    
    print("\nTesting full formatting:")
    sample_transcript = """Host: Welcome back to the show everyone.
Guest: Thanks for having me on.
Expert: This is really interesting stuff."""
    
    formatted = format_transcript_with_headers(sample_transcript, "ep29")
    print(formatted)
