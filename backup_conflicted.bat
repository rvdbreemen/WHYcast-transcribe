@echo off
REM Backup strategy for conflicted episode files
REM WHYcast Transcribe v0.3.0
echo Creating backups for conflicted files...
echo.

echo Backing up existing episode_12.mp3...
copy "podcasts\episode_12.mp3" "podcasts\episode_12.mp3.backup.12"
echo Created backup: episode_12.mp3.backup.12
echo.

echo Backing up existing episode_13.mp3...
copy "podcasts\episode_13.mp3" "podcasts\episode_13.mp3.backup.13"
echo Created backup: episode_13.mp3.backup.13
echo.

echo Backing up existing episode_14.mp3...
copy "podcasts\episode_14.mp3" "podcasts\episode_14.mp3.backup.14"
echo Created backup: episode_14.mp3.backup.14
echo.

echo Backing up existing episode_15.mp3...
copy "podcasts\episode_15.mp3" "podcasts\episode_15.mp3.backup.15"
echo Created backup: episode_15.mp3.backup.15
echo.

echo Backing up existing episode_16.mp3...
copy "podcasts\episode_16.mp3" "podcasts\episode_16.mp3.backup.16"
echo Created backup: episode_16.mp3.backup.16
echo.

echo Backing up existing episode_22.mp3...
copy "podcasts\episode_22.mp3" "podcasts\episode_22.mp3.backup.22"
echo Created backup: episode_22.mp3.backup.22
echo.

echo Backing up existing episode_26.mp3...
copy "podcasts\episode_26.mp3" "podcasts\episode_26.mp3.backup.26"
echo Created backup: episode_26.mp3.backup.26
echo.

echo Backing up existing episode_28.mp3...
copy "podcasts\episode_28.mp3" "podcasts\episode_28.mp3.backup.28"
echo Created backup: episode_28.mp3.backup.28
echo.

echo Backing up existing episode_30.mp3...
copy "podcasts\episode_30.mp3" "podcasts\episode_30.mp3.backup.30"
echo Created backup: episode_30.mp3.backup.30
echo.

echo Backing up existing episode_32.mp3...
copy "podcasts\episode_32.mp3" "podcasts\episode_32.mp3.backup.32"
echo Created backup: episode_32.mp3.backup.32
echo.

echo Backing up existing episode_33.mp3...
copy "podcasts\episode_33.mp3" "podcasts\episode_33.mp3.backup.33"
echo Created backup: episode_33.mp3.backup.33
echo.

echo Backing up existing episode_37.mp3...
copy "podcasts\episode_37.mp3" "podcasts\episode_37.mp3.backup.37"
echo Created backup: episode_37.mp3.backup.37
echo.

echo Backup complete!
echo Now manually resolve conflicts as described in manual_resolution.txt
pause