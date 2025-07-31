# 🎙️ Episode MP3 Discovery Tool - Usage Guide

## Overview

The `find_episodes.py` script helps you discover, analyze, and standardize WHYcast episode MP3 files. It can find episode files with various naming patterns and suggest standardized filenames.

## Features

✅ **Smart Episode Detection**: Recognizes various naming patterns
✅ **Duplicate Elimination**: Avoids showing the same file twice  
✅ **Standard Naming**: Suggests `episode_XX.mp3` format
✅ **Batch Rename Script**: Generates Windows batch script for renaming
✅ **Multiple Output Modes**: Detailed analysis or simple list

## Quick Usage Examples

### 1. Basic Discovery (Full Analysis)
```bash
python find_episodes.py
```
Shows complete analysis with episode numbers, current names, suggested names, and summary statistics.

### 2. Simple List (Tab-Separated)
```bash
python find_episodes.py --quiet
```
Outputs: `Episode# [TAB] Current_Name [TAB] Suggested_Name`

### 3. Generate Rename Script
```bash
python find_episodes.py --generate-script
```
Creates `rename_episodes.bat` with all needed rename operations.

### 4. Search Specific Directory
```bash
python find_episodes.py --dir /path/to/mp3s
```

## Recognized Patterns

The script recognizes these episode naming patterns:
- `episode_45.mp3`, `Episode_45.mp3`, `Episode 45.mp3`
- `ep45.mp3`, `ep_45.mp3`, `Ep 45.mp3`
- `episode-45.mp3`, `episode-1.mp3`
- `whycast-episode-13.mp3`, `Whycast33.mp3`
- `whycast_episode_4_-_fire_.mp3`

## Standard Format

All episodes are standardized to: `episode_XX.mp3`
- Zero-padded episode numbers (01, 02, ..., 45)
- Lowercase `episode_`
- `.mp3` extension

## Sample Output

```
📊 Found 62 episodes with recognizable numbers:
================================================================================
Ep#  Current Filename                         Suggested Filename
--------------------------------------------------------------------------------
45   Episode_45.mp3                           episode_45.mp3       📝
44   episode_44.mp3                           episode_44.mp3       ✅
43   episode43.mp3                            episode_43.mp3       📝
```

Legend:
- ✅ Already standardized
- 📝 Needs renaming

## Integration with Batch Processing

After standardizing your MP3 filenames, you can use the batch processing scripts:

```bash
# Update speaker assignments for all episodes
python quick_batch_update.py

# Update specific range
python batch_update_speaker_assignments.py --episodes 1-45
```

---
*WHYcast Transcribe v0.3.0*
