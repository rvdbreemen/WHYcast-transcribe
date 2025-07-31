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

def detect_rename_collisions(episodes: List[Tuple[int, str, str, str]]) -> Tuple[List, List, List]:
    """
    Detect potential rename collisions and categorize renames.
    
    Args:
        episodes: List of episode tuples
        
    Returns:
        Tuple of (safe_renames, collisions, duplicates)
    """
    renames_needed = [(episode_num, rel_path, filename, suggested) 
                     for episode_num, rel_path, filename, suggested in episodes 
                     if filename != suggested]
    
    if not renames_needed:
        return [], [], []
    
    # Group by directory and analyze
    by_directory = {}
    for episode_num, rel_path, old_name, new_name in renames_needed:
        directory = os.path.dirname(rel_path)
        if directory not in by_directory:
            by_directory[directory] = []
        by_directory[directory].append((episode_num, rel_path, old_name, new_name))
    
    safe_renames = []
    collisions = []
    duplicates = []
    
    for directory, dir_renames in by_directory.items():
        # Check for target filename collisions within this directory
        target_names = {}
        existing_files = set()
        
        # Get existing files in directory
        if directory and os.path.exists(directory):
            existing_files = {f.lower() for f in os.listdir(directory) if f.lower().endswith('.mp3')}
        elif not directory:  # Current directory
            existing_files = {f.lower() for f in os.listdir('.') if f.lower().endswith('.mp3')}
        
        for episode_num, rel_path, old_name, new_name in dir_renames:
            new_name_lower = new_name.lower()
            
            # Check if multiple files want the same target name
            if new_name_lower in target_names:
                duplicates.append((episode_num, rel_path, old_name, new_name, target_names[new_name_lower]))
            else:
                target_names[new_name_lower] = (episode_num, rel_path, old_name, new_name)
                
                # Check if target already exists (and it's not the source file itself)
                if new_name_lower in existing_files and old_name.lower() != new_name_lower:
                    collisions.append((episode_num, rel_path, old_name, new_name))
                else:
                    safe_renames.append((episode_num, rel_path, old_name, new_name))
    
    return safe_renames, collisions, duplicates

def print_collision_analysis(safe_renames, collisions, duplicates):
    """Print detailed collision analysis."""
    print("\n🔍 Collision Analysis Report")
    print("=" * 50)
    
    print(f"✅ Safe renames: {len(safe_renames)}")
    print(f"💥 Collisions: {len(collisions)}")
    print(f"🔄 Duplicates: {len(duplicates)}")
    
    if collisions:
        print(f"\n💥 COLLISIONS - Target files already exist:")
        for episode_num, rel_path, old_name, new_name in collisions:
            target_path = os.path.join(os.path.dirname(rel_path), new_name)
            print(f"   Episode {episode_num:02d}: {old_name} → {new_name}")
            print(f"   ⚠️  Target exists: {target_path}")
    
    if duplicates:
        print(f"\n🔄 DUPLICATES - Multiple files want same target name:")
        for episode_num, rel_path, old_name, new_name, other in duplicates:
            print(f"   Episode {episode_num:02d}: {old_name} → {new_name}")
            print(f"   Episode {other[0]:02d}: {other[2]} → {other[3]}")
            print(f"   Both want: {new_name}")
    
    if collisions or duplicates:
        print(f"\n💡 Recommendations:")
        print(f"   1. Check if existing files are duplicates you can delete")
        print(f"   2. Use --backup-conflicted to create backup strategy") 
        print(f"   3. Manually resolve conflicts before running rename script")
    else:
        print(f"\n🎉 No conflicts detected! Safe to run rename script.")

def generate_backup_strategy(episodes):
    """Generate backup and manual resolution strategy for conflicted files."""
    safe_renames, collisions, duplicates = detect_rename_collisions(episodes)
    
    if not collisions and not duplicates:
        print("✅ No conflicts detected - no backup strategy needed!")
        return
    
    backup_script = []
    manual_steps = []
    
    backup_script.extend([
        "@echo off",
        "REM Backup strategy for conflicted episode files",
        "REM WHYcast Transcribe v0.3.0",
        "echo Creating backups for conflicted files...",
        "echo.",
        ""
    ])
    
    if collisions:
        manual_steps.append("COLLISION RESOLUTION:")
        manual_steps.append("=" * 30)
        
        for episode_num, rel_path, old_name, new_name in collisions:
            directory = os.path.dirname(rel_path) or "."
            target_path = os.path.join(directory, new_name)
            backup_name = f"{new_name}.backup.{episode_num:02d}"
            
            backup_script.extend([
                f"echo Backing up existing {new_name}...",
                f'copy "{target_path}" "{os.path.join(directory, backup_name)}"',
                f"echo Created backup: {backup_name}",
                "echo.",
                ""
            ])
            
            manual_steps.append(f"Episode {episode_num:02d}: {old_name}")
            manual_steps.append(f"  Wants to become: {new_name}")
            manual_steps.append(f"  But {new_name} already exists")
            manual_steps.append(f"  Backup created: {backup_name}")
            manual_steps.append(f"  ACTION: Compare files and decide which to keep")
            manual_steps.append("")
    
    if duplicates:
        manual_steps.append("DUPLICATE RESOLUTION:")
        manual_steps.append("=" * 30)
        
        seen_conflicts = set()
        for episode_num, rel_path, old_name, new_name, other in duplicates:
            conflict_key = tuple(sorted([episode_num, other[0]]))
            if conflict_key in seen_conflicts:
                continue
            seen_conflicts.add(conflict_key)
            
            manual_steps.append(f"Target filename: {new_name}")
            manual_steps.append(f"  Episode {episode_num:02d}: {old_name}")
            manual_steps.append(f"  Episode {other[0]:02d}: {other[2]}")
            manual_steps.append(f"  ACTION: Decide which episode number is correct")
            manual_steps.append("")
    
    # Write backup script
    backup_script.extend([
        "echo Backup complete!",
        "echo Now manually resolve conflicts as described in manual_resolution.txt",
        "pause"
    ])
    
    with open("backup_conflicted.bat", 'w', encoding='utf-8') as f:
        f.write('\n'.join(backup_script))
    
    # Write manual resolution guide
    with open("manual_resolution.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(manual_steps))
    
    print(f"\n📋 Backup strategy created:")
    print(f"   backup_conflicted.bat - Creates backups of existing target files")
    print(f"   manual_resolution.txt - Step-by-step resolution guide")
    print(f"\n🔧 Resolution Process:")
    print(f"   1. Run: backup_conflicted.bat")
    print(f"   2. Follow: manual_resolution.txt")
    print(f"   3. Then run: python find_episodes.py --generate-script")

def generate_rename_script(episodes: List[Tuple[int, str, str, str]], output_file: str = "rename_episodes.bat"):
    """
    Generate a batch script to rename files to standard format with collision detection.
    
    Args:
        episodes: List of episode tuples
        output_file: Output script filename
    """
    safe_renames, collisions, duplicates = detect_rename_collisions(episodes)
    
    if not safe_renames and not collisions and not duplicates:
        print("✅ All files already have standard names!")
        return
    
    # Report issues first
    if collisions:
        print(f"\n⚠️  Found {len(collisions)} COLLISION(S) - target files already exist:")
        for episode_num, rel_path, old_name, new_name in collisions:
            target_path = os.path.join(os.path.dirname(rel_path), new_name)
            print(f"   💥 {old_name} → {new_name} (target exists: {target_path})")
    
    if duplicates:
        print(f"\n⚠️  Found {len(duplicates)} DUPLICATE target name(s):")
        for episode_num, rel_path, old_name, new_name, other in duplicates:
            print(f"   🔄 Episode {episode_num}: {old_name} → {new_name}")
            print(f"      Conflicts with Episode {other[0]}: {other[2]} → {other[3]}")
    
    if not safe_renames:
        print("\n❌ No safe renames possible due to collisions and duplicates!")
        print("   Please resolve conflicts manually before running batch rename.")
        return
    
    print(f"\n✅ {len(safe_renames)} safe renames identified")
    
    # Generate script with safety checks
    script_lines = [
        "@echo off",
        "REM Auto-generated episode rename script with collision detection",
        "REM WHYcast Transcribe v0.3.0",
        "echo Renaming episodes to standard format...",
        "echo ⚠️  This script includes collision detection",
        "echo.",
        ""
    ]
    
    # Add collision warnings to script
    if collisions or duplicates:
        script_lines.extend([
            "echo ⚠️  WARNING: Some files could not be renamed due to conflicts:",
            ""
        ])
        
        for episode_num, rel_path, old_name, new_name in collisions:
            script_lines.append(f'echo    COLLISION: {old_name} -X-> {new_name} (target exists)')
        
        for episode_num, rel_path, old_name, new_name, other in duplicates:
            script_lines.append(f'echo    DUPLICATE: {old_name} and {other[2]} both want {new_name}')
        
        script_lines.extend([
            "echo.",
            "echo Please resolve these conflicts manually.",
            "echo.",
            ""
        ])
    
    # Add safe renames
    script_lines.extend([
        f"echo Proceeding with {len(safe_renames)} safe renames...",
        "echo.",
        ""
    ])
    
    for episode_num, rel_path, old_name, new_name in safe_renames:
        directory = os.path.dirname(rel_path)
        if directory:
            old_path = f'"{directory}\\{old_name}"'
            new_path = f'"{directory}\\{new_name}"'
        else:
            old_path = f'"{old_name}"'
            new_path = f'"{new_name}"'
        
        script_lines.extend([
            f"echo Checking: {old_name} -> {new_name}",
            f"if exist {new_path} (",
            f"    echo ERROR: Target already exists: {new_name}",
            f"    echo Skipping rename of {old_name}",
            f") else (",
            f"    echo Renaming: {old_name} -> {new_name}",
            f"    rename {old_path} {new_name}",
            f"    if errorlevel 1 (",
            f"        echo ERROR: Failed to rename {old_name}",
            f"    ) else (",
            f"        echo ✅ Success: {old_name} -> {new_name}",
            f"    )",
            f")",
            "echo.",
            ""
        ])
    
    script_lines.extend([
        "echo.",
        "echo Rename operation completed!",
        "echo Check above output for any errors.",
        "pause"
    ])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))
    
    print(f"\n📜 Safe rename script generated: {output_file}")
    print(f"   Contains {len(safe_renames)} safe rename operations")
    if collisions or duplicates:
        print(f"   ⚠️  {len(collisions + duplicates)} problematic renames excluded")
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
        '--check-collisions', '-c',
        action='store_true',
        help='Check for rename collisions without generating script'
    )
    
    parser.add_argument(
        '--backup-conflicted', '-b',
        action='store_true',
        help='Generate backup strategy for conflicted files'
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
    
    if args.check_collisions:
        # Just check for collisions
        safe_renames, collisions, duplicates = detect_rename_collisions(episodes)
        print_collision_analysis(safe_renames, collisions, duplicates)
        return
    
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
        elif args.backup_conflicted:
            generate_backup_strategy(episodes)

if __name__ == "__main__":
    main()
