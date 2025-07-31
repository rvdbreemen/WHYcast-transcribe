#!/usr/bin/env python3
"""
Episode MP3 File Discovery and Standardization Tool
WHYcast Transcribe v0.3.0

This script finds MP3 files, extracts episode numbers, and suggests standardized filenames.
"""

import os
import re
import sys
from typing import List, Tuple, Optional
from pathlib import Path

def extract_episode_number(filename: str) -> Optional[int]:
    """
    Extract episode number from filename using various patterns.
    
    Args:
        filename: The filename to analyze
        
    Returns:
        Episode number as integer, or None if not found
    """
    # Common patterns for episode numbers (more specific patterns first)
    patterns = [
        r'[Ee]pisode[_\s]*(\d+)',          # Episode 45, episode_45, Episode_45
        r'[Ee]p[_\s]*(\d+)',               # Ep45, ep_45, EP 45
        r'[Ww]hycast[_\s]*(\d+)',          # WHYcast 45, whycast_45, Whycast33
        r'[Ee](\d+)(?:[_\s]|$)',           # E45, e45 (followed by separator or end)
        r'^(\d+)(?:[_\s-]|$)',             # 45_, 45-, 45 (at start)
        r'episode-(\d+)',                  # episode-45 (with dash)
        r'whycast-episode-(\d+)',          # whycast-episode-13
        r'[_\s-](\d+)(?:[_\s-]|$)',        # _45_, -45-, 45 (surrounded by separators)
    ]
    
    # Remove file extension for analysis
    name_without_ext = os.path.splitext(filename)[0]
    
    for pattern in patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            try:
                episode_num = int(match.group(1))
                # Reasonable episode number range (0-999)
                if 0 <= episode_num <= 999:
                    return episode_num
            except ValueError:
                continue
    
    return None

def suggest_standard_filename(episode_num: int, original_filename: str) -> str:
    """
    Suggest a standardized filename for the episode.
    
    Args:
        episode_num: The episode number
        original_filename: The original filename
        
    Returns:
        Suggested standardized filename
    """
    # Standard format: episode_XX.mp3
    return f"episode_{episode_num:02d}.mp3"

def find_mp3_files(directory: str = ".") -> List[str]:
    """
    Find all MP3 files in the specified directory and subdirectories.
    
    Args:
        directory: Directory to search (default: current directory)
        
    Returns:
        List of MP3 file paths
    """
    mp3_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp3'):
                full_path = os.path.join(root, file)
                mp3_files.append(full_path)
    
    return mp3_files

def analyze_episodes(search_dirs: List[str] = None) -> List[Tuple[int, str, str, str]]:
    """
    Analyze MP3 files and extract episode information.
    
    Args:
        search_dirs: List of directories to search (default: current directory and podcasts/)
        
    Returns:
        List of tuples: (episode_number, relative_path, filename, suggested_name)
    """
    if search_dirs is None:
        search_dirs = [".", "podcasts"]
    
    episodes = []
    unknown_files = []
    seen_files = set()  # Track seen files to avoid duplicates
    
    print("🔍 Searching for MP3 files...")
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            print(f"⚠️  Directory not found: {search_dir}")
            continue
            
        print(f"📁 Searching in: {search_dir}")
        mp3_files = find_mp3_files(search_dir)
        
        for mp3_path in mp3_files:
            # Normalize path to avoid duplicates
            normalized_path = os.path.normpath(mp3_path)
            if normalized_path in seen_files:
                continue
            seen_files.add(normalized_path)
            
            filename = os.path.basename(mp3_path)
            relative_path = os.path.relpath(mp3_path)
            
            episode_num = extract_episode_number(filename)
            
            if episode_num is not None:
                suggested_name = suggest_standard_filename(episode_num, filename)
                episodes.append((episode_num, relative_path, filename, suggested_name))
            else:
                unknown_files.append((relative_path, filename))
    
    # Sort by episode number, then by filename for stable ordering
    episodes.sort(key=lambda x: (x[0], x[2]))
    
    # Report unknown files
    if unknown_files:
        print(f"\n⚠️  Found {len(unknown_files)} MP3 files without recognizable episode numbers:")
        for rel_path, filename in unknown_files:
            print(f"   📄 {rel_path}")
    
    return episodes

def print_episode_analysis(episodes: List[Tuple[int, str, str, str]]):
    """
    Print formatted analysis of episodes.
    
    Args:
        episodes: List of episode tuples from analyze_episodes()
    """
    if not episodes:
        print("❌ No episodes found!")
        return
    
    print(f"\n📊 Found {len(episodes)} episodes with recognizable numbers:")
    print("=" * 80)
    print(f"{'Ep#':<4} {'Current Filename':<40} {'Suggested Filename':<20}")
    print("-" * 80)
    
    for episode_num, relative_path, filename, suggested_name in episodes:
        # Show if rename is needed
        rename_indicator = "✅" if filename == suggested_name else "📝"
        print(f"{episode_num:<4} {filename:<40} {suggested_name:<20} {rename_indicator}")
    
    print("-" * 80)
    
    # Summary statistics
    total_episodes = len(episodes)
    episodes_needing_rename = sum(1 for _, _, filename, suggested in episodes if filename != suggested)
    
    print(f"\n📈 Summary:")
    print(f"   Total episodes found: {total_episodes}")
    print(f"   Episodes needing rename: {episodes_needing_rename}")
    print(f"   Already standardized: {total_episodes - episodes_needing_rename}")
    
    # Episode range analysis
    if episodes:
        min_ep = min(ep[0] for ep in episodes)
        max_ep = max(ep[0] for ep in episodes)
        expected_range = list(range(min_ep, max_ep + 1))
        missing_episodes = [ep for ep in expected_range if ep not in [e[0] for e in episodes]]
        
        print(f"   Episode range: {min_ep} - {max_ep}")
        if missing_episodes:
            print(f"   Missing episodes: {missing_episodes}")

def generate_rename_script(episodes: List[Tuple[int, str, str, str]], output_file: str = "rename_episodes.bat"):
    """
    Generate a batch script to rename files to standard format.
    
    Args:
        episodes: List of episode tuples
        output_file: Output script filename
    """
    renames_needed = [(rel_path, filename, suggested) 
                     for _, rel_path, filename, suggested in episodes 
                     if filename != suggested]
    
    if not renames_needed:
        print("✅ All files already have standard names!")
        return
    
    script_lines = [
        "@echo off",
        "REM Auto-generated episode rename script",
        "REM WHYcast Transcribe v0.3.0",
        "echo Renaming episodes to standard format...",
        ""
    ]
    
    for rel_path, old_name, new_name in renames_needed:
        directory = os.path.dirname(rel_path)
        if directory:
            old_path = f'"{directory}\\{old_name}"'
            new_path = f'"{directory}\\{new_name}"'
        else:
            old_path = f'"{old_name}"'
            new_path = f'"{new_name}"'
        
        script_lines.extend([
            f"echo Renaming: {old_name} -> {new_name}",
            f"rename {old_path} {new_path}",
            ""
        ])
    
    script_lines.extend([
        "echo Done!",
        "pause"
    ])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))
    
    print(f"\n📜 Rename script generated: {output_file}")
    print(f"   Contains {len(renames_needed)} rename operations")
    print(f"   Run: {output_file}")

def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find and analyze WHYcast episode MP3 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_episodes.py                    # Search current dir and podcasts/
  python find_episodes.py --dir /path/mp3s   # Search specific directory
  python find_episodes.py --generate-script  # Generate rename script
  python find_episodes.py --quiet            # Minimal output
        """
    )
    
    parser.add_argument(
        '--dir', '-d', 
        action='append',
        help='Directory to search for MP3 files (can be used multiple times)'
    )
    
    parser.add_argument(
        '--generate-script', '-g',
        action='store_true',
        help='Generate a batch script to rename files to standard format'
    )
    
    parser.add_argument(
        '--output-script', '-o',
        default='rename_episodes.bat',
        help='Output filename for rename script (default: rename_episodes.bat)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output (just the episode list)'
    )
    
    args = parser.parse_args()
    
    # Determine search directories
    search_dirs = args.dir if args.dir else None
    
    if not args.quiet:
        print("🎙️  WHYcast Episode MP3 Discovery Tool")
        print("=" * 50)
    
    # Analyze episodes
    episodes = analyze_episodes(search_dirs)
    
    if args.quiet:
        # Just print the episode list
        for episode_num, relative_path, filename, suggested_name in episodes:
            print(f"{episode_num:02d}\t{filename}\t{suggested_name}")
    else:
        # Full analysis output
        print_episode_analysis(episodes)
        
        # Generate rename script if requested
        if args.generate_script:
            generate_rename_script(episodes, args.output_script)

if __name__ == "__main__":
    main()
