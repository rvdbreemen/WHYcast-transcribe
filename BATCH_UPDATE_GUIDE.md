# 🎯 Batch Speaker Assignment Update Guide

This guide shows you how to use the batch update scripts to regenerate speaker assignment files with header/footer integration.

## 📁 Scripts Created

1. **`batch_update_speaker_assignments.py`** - Full-featured script with all options
2. **`quick_batch_update.py`** - Simple script for immediate use

## 🚀 Quick Start

### Update All Episodes (Simple)
```bash
python quick_batch_update.py
```

### Update Specific Episode
```bash
python quick_batch_update.py episode_45
```

## 📋 Full Script Usage

### List All Episodes and Status
```bash
python batch_update_speaker_assignments.py --list
```
Shows which episodes have speaker assignments and which need them.

### Dry Run (See What Would Be Processed)
```bash
python batch_update_speaker_assignments.py --dry-run
```

### Update All Missing Speaker Assignments
```bash
python batch_update_speaker_assignments.py
```

### Force Update All Episodes (Including Existing)
```bash
python batch_update_speaker_assignments.py --force
```

### Update Specific Episodes
```bash
python batch_update_speaker_assignments.py --episodes episode_45 episode_29 ep30
```

### Force Update Episode 45 (Your Request)
```bash
python batch_update_speaker_assignments.py --episodes episode_45 --force
```

## 🎯 What Gets Updated

Each processed episode will get:

✅ **Clean Speaker Labels**
- Nancy: (instead of [SPEAKER_01])
- Host: (when name unclear)
- Expert: (for technical guests)

✅ **Disclaimer Header**
```
== Disclaimer ==
This transcript was generated from the WHYcast podcast audio using automated speech recognition and speaker diarization...

== Transcript 45 ==
```

✅ **WikiMedia Footer**
```
[[Category:WHYcast]]
[[Category:transcription]]
[[Episode number::45| ]]
```

## 📊 Output Files Created

For each episode (e.g., episode_45), the script creates:
- `episode_45_speaker_analysis.txt` - Analysis from o4 model
- `episode_45_speaker_assignment.txt` - Final transcript with headers/footers
- `episode_45_speaker_assignment.html` - HTML version
- `episode_45_speaker_assignment.wiki` - Wiki format
- `episode_45_speaker_assignment_metadata.txt` - Processing metadata

## 🛡️ Safety Features

- **Won't overwrite existing files** unless you use `--force`
- **Validates content** before processing
- **Logs all operations** with timestamps
- **Shows progress** and success rates
- **Handles errors gracefully** and continues with other episodes

## 🔧 Troubleshooting

### "Cannot import transcribe module"
Make sure you're running the script from the WHYcast-transcribe directory:
```bash
cd d:\Users\Robert\Documents\GitHub\RvdB\WHYcast-transcribe
python quick_batch_update.py
```

### "No transcript files found"
Check that your podcast files are in the `podcasts/` directory and named like:
- `episode_45_transcript.txt`
- `episode_45_merged.txt`
- `ep29_transcript.txt`

### API Rate Limits
The scripts include delays between episodes to avoid overwhelming the OpenAI API.

## 🎯 Your Immediate Use Case

To fix episode 45 with headers/footers:

```bash
# Quick way
python quick_batch_update.py episode_45

# Or with full script
python batch_update_speaker_assignments.py --episodes episode_45 --force
```

This will create `podcasts/episode_45_speaker_assignment.txt` with:
- Proper disclaimer header
- "== Transcript 45 ==" section
- Clean speaker labels
- WikiMedia footer with episode number

## 📈 Progress Tracking

The scripts show real-time progress:
```
🎯 WHYcast Batch Speaker Assignment Update
==================================================
📁 Found 12 transcript files

📊 Processing 1/12: Episode 45 (episode_45)
📖 Reading: episode_45_transcript.txt
🔄 Running speaker assignment...
✅ Success! Content retention: 98.5%

📈 BATCH UPDATE SUMMARY
==================================================
✅ Successfully processed: 1
   • episode_45 (98.5% retention)
```

Ready to use! 🚀
