# 🚨 Handling Rename Collisions - Safety Guide

## Overview

When standardizing episode filenames, you might encounter **rename collisions** where the target filename already exists. The `find_episodes.py` tool provides comprehensive collision detection and safe resolution strategies.

## Types of Conflicts

### 💥 **Collisions**
Target filename already exists:
```
episode-12.mp3 → episode_12.mp3  (But episode_12.mp3 already exists!)
```

### 🔄 **Duplicates** 
Multiple files want the same target name:
```
episode_1.mp3  → episode_01.mp3
episode-1.mp3  → episode_01.mp3  (Both want episode_01.mp3!)
```

## Safety Commands

### 1. Check for Conflicts First
```bash
python find_episodes.py --check-collisions
```
**Output:**
- Lists all safe renames
- Identifies collisions and duplicates
- Provides recommendations

### 2. Generate Backup Strategy
```bash
python find_episodes.py --backup-conflicted
```
**Creates:**
- `backup_conflicted.bat` - Backup existing files
- `manual_resolution.txt` - Step-by-step guide

### 3. Safe Rename Script
```bash
python find_episodes.py --generate-script
```
**Features:**
- Only includes safe renames
- Excludes conflicted files
- Includes collision detection in batch script

## Resolution Process

### Step 1: Identify Conflicts
```bash
python find_episodes.py --check-collisions
```

### Step 2: Create Backups
```bash
python find_episodes.py --backup-conflicted
```

### Step 3: Run Backup Script
```bash
backup_conflicted.bat
```
This creates `.backup.XX` files for existing targets.

### Step 4: Manual Resolution
Follow `manual_resolution.txt`:

**For Collisions:**
1. Compare files (same content? different quality?)
2. Decide which to keep
3. Delete unwanted duplicates

**For Duplicates:**
1. Check which episode number is correct
2. Rename or delete incorrect files
3. Verify episode content matches number

### Step 5: Clean Rename
```bash
python find_episodes.py --generate-script
```
Now generates a collision-free script.

## Safe Rename Script Features

The generated `rename_episodes.bat` includes:

```batch
echo Checking: episode-12.mp3 -> episode_12.mp3
if exist "podcasts\episode_12.mp3" (
    echo ERROR: Target already exists: episode_12.mp3
    echo Skipping rename of episode-12.mp3
) else (
    echo Renaming: episode-12.mp3 -> episode_12.mp3
    rename "podcasts\episode-12.mp3" episode_12.mp3
    if errorlevel 1 (
        echo ERROR: Failed to rename episode-12.mp3
    ) else (
        echo ✅ Success: episode-12.mp3 -> episode_12.mp3
    )
)
```

## Common Collision Scenarios

### Scenario 1: Different Formats of Same Episode
```
episode-12.mp3   (hyphen version)
episode_12.mp3   (underscore version)
```
**Resolution:** Compare files, keep better quality, delete duplicate.

### Scenario 2: Multiple Downloads
```
episode_1.mp3    (original)
episode-1.mp3    (re-download)
```
**Resolution:** Check file sizes/dates, keep newer/better version.

### Scenario 3: Wrong Episode Numbers
```
episode_5.mp3    (actually episode 05)
episode-5.mp3    (actually episode 50)
```
**Resolution:** Listen to content, correct episode number manually.

## Prevention Tips

1. **Always check collisions before bulk renaming**
2. **Use backup strategy for complex situations**
3. **Verify episode content matches number**
4. **Clean up obvious duplicates first**

## Emergency Recovery

If something goes wrong:

1. **Restore from backups:**
   ```bash
   copy episode_12.mp3.backup.12 episode_12.mp3
   ```

2. **Re-run collision detection:**
   ```bash
   python find_episodes.py --check-collisions
   ```

3. **Start resolution process again**

## Example Workflow

```bash
# 1. Check what needs to be done
python find_episodes.py

# 2. Check for conflicts
python find_episodes.py --check-collisions

# 3. If conflicts exist, create backup strategy
python find_episodes.py --backup-conflicted

# 4. Create backups
backup_conflicted.bat

# 5. Manually resolve conflicts (follow manual_resolution.txt)

# 6. Generate safe rename script
python find_episodes.py --generate-script

# 7. Run safe renames
rename_episodes.bat
```

## Key Benefits

✅ **No Data Loss**: Always backs up before overwriting
✅ **Selective Processing**: Only renames safe files
✅ **Clear Guidance**: Step-by-step resolution instructions
✅ **Error Recovery**: Multiple safety checks and rollback options

---
*WHYcast Transcribe v0.3.0 - Collision-Safe Renaming*
