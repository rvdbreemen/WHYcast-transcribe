"""
RSS feed fetching and episode download for the WHYcast pipeline (ADR-008).

Functions extracted verbatim from transcribe.py (delete_episode_files,
podcast_fetching_workflow, download_all_episodes_from_rssfeed) as part of
the mechanical monolith split. No logic changes beyond print()->emit() and
the module-level import scaffolding.
"""

import glob
import logging
import os
from typing import List, Optional

import requests

from whycast._deps import feedparser
from whycast.events import emit
from whycast.io_utils import atomic_writer

logger = logging.getLogger(__name__)


def delete_episode_files(base_name: str, output_dir: str, exclude_files: Optional[List[str]] = None):
    """
    Delete all files related to a given episode base name in the output directory, except those in exclude_files and except .mp3 files.
    """
    patterns = [
        f"{base_name}*.*",  # matches all files for this episode
    ]
    for pattern in patterns:
        for file_path in glob.glob(os.path.join(output_dir, pattern)):
            abs_file_path = os.path.abspath(file_path)
            # Never delete the input file or any .mp3 file
            if exclude_files and abs_file_path in [os.path.abspath(f) for f in exclude_files]:
                continue
            if abs_file_path.lower().endswith('.mp3'):
                continue
            try:
                os.remove(abs_file_path)
                logging.info(f"Deleted file: {abs_file_path}")
            except Exception as e:
                logging.error(f"Failed to delete file {abs_file_path}: {e}")

def podcast_fetching_workflow(rssfeed, output_dir, return_base_name=False):
    """
    Fetch the latest episode from the RSS feed and download the audio file.
    Returns the path to the downloaded audio file, or None if not found.
    If return_base_name is True, also returns the base name for the episode file.
    """
    import glob
    feed = feedparser.parse(rssfeed)
    if not feed.entries:
        emit("feed", "No episodes found in RSS feed.")
        return (None, None) if return_base_name else None

    latest = feed.entries[0]
    # Try to find the audio link
    audio_url = None
    for link in latest.get('links', []):
        if isinstance(link, dict) and str(link.get('type', '')).startswith('audio'):
            audio_url = link.get('href')
            break
    if not audio_url or not isinstance(audio_url, str):
        emit("feed", "No audio file found for the latest episode.")
        return (None, None) if return_base_name else None

    os.makedirs(output_dir, exist_ok=True)
    safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in latest.get('title', 'episode'))
    audio_ext = os.path.splitext(audio_url)[-1].split('?')[0] if '.' in os.path.basename(audio_url) else '.mp3'
    audio_file = os.path.join(output_dir, f"{safe_title}{audio_ext}")
    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    # First check if the expected file exists
    if os.path.exists(audio_file):
        emit("feed", f"Audio file already exists: {audio_file}")
        return (audio_file, base_name) if return_base_name else audio_file

    # If not found, try to find similar files in the directory
    # Look for files that might match the episode (with different naming conventions)
    episode_title = latest.get('title', '').lower()
    emit("feed", f"Looking for existing files matching episode: {episode_title}")

    # Try various patterns to find existing files
    patterns_to_try = [
        f"{safe_title}.*",  # exact match
        f"*{safe_title.split('_')[-1]}*.*" if '_' in safe_title else f"*{safe_title}*.*",  # partial match
        "*episode*44*.*",  # fallback pattern for episode 44
        "*44*.*"  # very broad pattern
    ]

    for pattern in patterns_to_try:
        matching_files = glob.glob(os.path.join(output_dir, pattern))
        audio_files = [f for f in matching_files if f.lower().endswith(('.mp3', '.m4a', '.wav', '.flac'))]

        if audio_files:
            # Sort by modification time to get the most recent
            audio_files.sort(key=os.path.getmtime, reverse=True)
            found_file = audio_files[0]
            found_base_name = os.path.splitext(os.path.basename(found_file))[0]
            emit("feed", f"Found existing audio file: {found_file}")
            return (found_file, found_base_name) if return_base_name else found_file

    # If no existing file found, download the new one
    emit("feed", f"No existing file found. Downloading latest episode to {audio_file} ...")
    try:
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            # backup=False: the audio is the source recording, there is no
            # previous version worth a 20-100 MB .bak copy (ADR-009).
            with atomic_writer(audio_file, backup=False) as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        emit("feed", f"Downloaded: {audio_file}")
        return (audio_file, base_name) if return_base_name else audio_file
    except Exception as e:
        emit("feed", f"Failed to download episode: {e}")
        return (None, None) if return_base_name else None

def download_all_episodes_from_rssfeed(rssfeed: str, output_dir: str = './podcasts'):
    """
    Download all mp3 audio files from the RSS feed to the output directory.
    Skips files that already exist. Shows a progress bar.
    """
    feed = feedparser.parse(rssfeed)
    if not feed.entries:
        emit("feed", "No episodes found in RSS feed.")
        return

    os.makedirs(output_dir, exist_ok=True)
    emit("feed", f"Found {len(feed.entries)} episodes in RSS feed.")
    from tqdm import tqdm
    for entry in tqdm(feed.entries, desc="Downloading episodes", unit="episode"):
        audio_url = None
        for link in entry.get('links', []):
            if isinstance(link, dict) and str(link.get('type', '')).startswith('audio'):
                audio_url = link.get('href')
                break
        if not audio_url or not isinstance(audio_url, str):
            emit("feed", f"No audio file found for episode: {entry.get('title', 'unknown')}")
            continue
        safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in entry.get('title', 'episode'))
        audio_ext = os.path.splitext(audio_url)[-1].split('?')[0] if '.' in os.path.basename(audio_url) else '.mp3'
        audio_file = os.path.join(output_dir, f"{safe_title}{audio_ext}")
        if os.path.exists(audio_file):
            tqdm.write(f"Already exists: {audio_file}")
            continue
        tqdm.write(f"Downloading: {audio_url}\n  -> {audio_file}")
        try:
            with requests.get(audio_url, stream=True) as r:
                r.raise_for_status()
                # backup=False: see the note at the other download site.
                with atomic_writer(audio_file, backup=False) as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            tqdm.write(f"Downloaded: {audio_file}")
        except Exception as e:
            tqdm.write(f"Failed to download {audio_url}: {e}")
    emit("feed", "All episodes processed.")
