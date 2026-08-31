"""
RSS feed fetching and episode download for the WHYcast pipeline (ADR-008).

Functions extracted from transcribe.py (delete_episode_files,
podcast_fetching_workflow, download_all_episodes_from_rssfeed) as part of
the mechanical monolith split.

Two things have changed since the extraction, both of them about not putting
rubbish on disk (ADR-009):

* A download is only published if what arrived is as long as it was promised
  to be. See :func:`_verify_download`.
* ``delete_episode_files`` matches the episode base the way the scanner does,
  on a separator boundary, instead of on a bare filename prefix.
"""

import logging
import os
from typing import List, Optional

import requests

from whycast._deps import feedparser, feedparser_available
from whycast.episodes import belongs_to_base, is_input_filename
from whycast.errors import ConfigurationError, PipelineError
from whycast.events import emit
from whycast.io_utils import atomic_writer

logger = logging.getLogger(__name__)

#: ``(connect, read)`` timeouts for an episode download, in seconds.
#:
#: The read timeout is the gap allowed *between* chunks, not a budget for the
#: whole transfer, so a 100 MB episode over a slow line is in no danger; a
#: server that accepts the connection and then goes quiet is. Without it a
#: stalled download hangs the job forever - no ceiling, no error, no progress -
#: and never reaches the failure handling below.
DOWNLOAD_TIMEOUT = (10, 60)


def _require_feedparser() -> None:
    """Fail with a sentence a person can act on, not an AttributeError.

    ``feedparser`` is optional (ADR-008), and when it is absent the shim in
    whycast._deps is None. Calling ``.parse`` on that raises
    ``AttributeError: 'NoneType' object has no attribute 'parse'`` halfway
    through a workflow, which says nothing about what to install.
    """
    if not feedparser_available:
        raise ConfigurationError(
            "feedparser is not installed, so the RSS feed cannot be read. "
            "Install it with 'pip install feedparser', or pass an audio file "
            "directly instead of fetching from the feed."
        )

def delete_episode_files(base_name: str, output_dir: str, exclude_files: Optional[List[str]] = None):
    """
    Delete all files related to a given episode base name in the output directory, except those in exclude_files and except .mp3 files.

    "Related to" means what :func:`whycast.episodes.belongs_to_base` means, and
    that is not a detail. This used to glob ``f"{base_name}*.*"``, a bare prefix
    match, so a forced re-run of ``episode_1`` deleted every artifact of
    ``episode_10`` through ``episode_19`` as well - all of them present in this
    corpus, none of them recoverable, since ``podcasts/`` is gitignored and
    artifacts carry no ``.bak`` (ADR-009). Sharing the scanner's rule means a
    file the scanner attributes to ``episode_10`` can never be deleted by a
    re-run of ``episode_1``.

    Pipeline *inputs* are not deleted either, whichever episode they belong to
    (:func:`whycast.episodes.is_input_filename`). ADR-010's Must Not is
    explicit: *do not delete a mapping automatically, on re-transcription or
    otherwise. It is human work; discarding it is a decision only a person
    makes.* This function is what ``--force`` and the web UI's "force
    reprocess" call, and it used to take ``<base>_speakers.json`` **and** the
    ``.bak`` kept precisely to make that file recoverable - so a single click
    destroyed both copies of somebody's typing, silently, in a gitignored
    directory. The mapping may of course be *stale* after a re-transcription;
    that is the fingerprint's job to notice and a person's job to resolve
    (:func:`whycast.pipeline.speakers.mapping_is_stale`), and it is a decision
    made with the file still in front of them.

    What survives a force, then: the audio, anything in ``exclude_files``, and
    human input. Everything else the pipeline can make again.
    """
    if not os.path.isdir(output_dir):
        return
    excluded = {os.path.abspath(f) for f in (exclude_files or [])}
    for name in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, name)
        if not os.path.isfile(file_path):
            continue
        # Checked before the base match on purpose: human input is spared
        # whoever it belongs to, and the ``.bak`` does not decompose into a
        # base plus a known suffix anyway.
        if is_input_filename(name):
            logging.info("Kept human input, not an artifact: %s", file_path)
            continue
        stem = os.path.splitext(name)[0]
        if not belongs_to_base(stem, base_name):
            continue
        abs_file_path = os.path.abspath(file_path)
        # Never delete the input file or any .mp3 file
        if abs_file_path in excluded:
            continue
        if abs_file_path.lower().endswith('.mp3'):
            continue
        try:
            os.remove(abs_file_path)
            logging.info(f"Deleted file: {abs_file_path}")
        except Exception as e:
            logging.error(f"Failed to delete file {abs_file_path}: {e}")


def _enclosure_length(link: dict) -> Optional[int]:
    """Size the RSS enclosure claims, in bytes, or None when it does not say.

    Parsed here, away from the download, so a malformed ``length`` in a feed can
    never raise inside the download block and get reported as a failed download
    of a perfectly good file. Anything non-numeric, absent, or <= 0 means "no
    information" - never "expect zero bytes".
    """
    try:
        declared = int(str(link.get("length", "")).strip())
    except (TypeError, ValueError):
        return None
    return declared if declared > 0 else None


def _header_length(response) -> Optional[int]:
    """Size the response headers claim, in bytes, or None when unusable.

    ``Content-Length`` counts the bytes *on the wire*. When the server compresses
    the body, requests hands us the decoded bytes, so comparing the two would
    fail every such download; a present ``Content-Encoding`` therefore means "no
    information" here.
    """
    if response.headers.get("Content-Encoding"):
        return None
    try:
        value = int(str(response.headers.get("Content-Length", "")).strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _verify_download(
    audio_url: str,
    written: int,
    header_length: Optional[int],
    declared_length: Optional[int],
) -> None:
    """Raise unless the bytes received are as long as they were promised to be.

    Called *inside* the ``atomic_writer`` block on purpose: raising there
    discards the temp file and leaves the target path exactly as it was
    (absent, or holding the previous episode). Raising after the block would be
    too late - the rename has already published the file.

    Three ways a download can be worthless while still looking like an HTTP
    success, all three measured against a local server:

    * **Nothing arrived.** ``Content-Length: 0`` with an empty body is a
      self-consistent 200 that the header check below would wave through. Zero
      bytes is never an episode.
    * **Short of the header.** The server said how long the body would be and
      sent less. ADR-009: a body shorter than the advertised ``Content-Length``
      counts as failed, not complete.
    * **Short of the feed.** With close-delimited framing (HTTP/1.0, or
      ``Connection: close`` without a ``Content-Length``) a truncated body ends
      like a complete one and there is no header to compare against. The RSS
      enclosure's ``length`` is then the only witness, and it is free: the same
      links dict already read for ``type`` and ``href`` carries it. It is also
      what catches an error page served as ``200 text/html`` - 49 bytes where
      the feed says 409610.

    The feed check is deliberately one-sided: shorter than declared fails,
    longer only logs. A shorter file is truncated or substituted and has no
    value. A longer one is the normal result of a host that re-encoded or
    injected something after publishing the feed, and refusing it would break
    downloads that are perfectly fine.
    """
    if written == 0:
        raise PipelineError(f"empty response body (0 bytes) from {audio_url}")
    if header_length is not None and written != header_length:
        raise PipelineError(
            f"incomplete download from {audio_url}: got {written} bytes, "
            f"Content-Length promised {header_length}"
        )
    if declared_length is not None and written < declared_length:
        raise PipelineError(
            f"truncated download from {audio_url}: got {written} bytes, "
            f"the feed says the episode is {declared_length}"
        )
    if declared_length is not None and written > declared_length:
        logger.info(
            "Downloaded %s bytes from %s where the feed declared %s; keeping it.",
            written, audio_url, declared_length,
        )


def _download_audio(audio_url: str, audio_file: str, declared_length: Optional[int]) -> None:
    """Stream ``audio_url`` into ``audio_file``, or leave nothing behind.

    Publishes the file only when the whole body arrived; on any failure the
    temp file is discarded and ``audio_file`` is untouched. A partial recording
    has no value and, worse, the "skip files that already exist" rule would stop
    it from ever being re-fetched (ADR-009).
    """
    with requests.get(audio_url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        header_length = _header_length(r)
        written = 0
        # backup=False: the audio is the source recording, there is no
        # previous version worth a 20-100 MB .bak copy (ADR-009).
        with atomic_writer(audio_file, backup=False) as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                written += len(chunk)
            _verify_download(audio_url, written, header_length, declared_length)

def podcast_fetching_workflow(rssfeed, output_dir, return_base_name=False):
    """
    Fetch the latest episode from the RSS feed and download the audio file.
    Returns the path to the downloaded audio file, or None if not found.
    If return_base_name is True, also returns the base name for the episode file.
    """
    import glob
    _require_feedparser()
    feed = feedparser.parse(rssfeed)
    if not feed.entries:
        emit("feed", "No episodes found in RSS feed.")
        return (None, None) if return_base_name else None

    latest = feed.entries[0]
    # Try to find the audio link
    audio_url = None
    declared_length = None
    for link in latest.get('links', []):
        if isinstance(link, dict) and str(link.get('type', '')).startswith('audio'):
            audio_url = link.get('href')
            # The enclosure's own size claim: the only witness that the body we
            # get back is the whole episode when the server sends no usable
            # Content-Length. Read it here, where the link dict is already open.
            declared_length = _enclosure_length(link)
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
        _download_audio(audio_url, audio_file, declared_length)
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
    _require_feedparser()
    feed = feedparser.parse(rssfeed)
    if not feed.entries:
        emit("feed", "No episodes found in RSS feed.")
        return

    os.makedirs(output_dir, exist_ok=True)
    emit("feed", f"Found {len(feed.entries)} episodes in RSS feed.")
    from tqdm import tqdm
    for entry in tqdm(feed.entries, desc="Downloading episodes", unit="episode"):
        audio_url = None
        declared_length = None
        for link in entry.get('links', []):
            if isinstance(link, dict) and str(link.get('type', '')).startswith('audio'):
                audio_url = link.get('href')
                declared_length = _enclosure_length(link)
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
            _download_audio(audio_url, audio_file, declared_length)
            tqdm.write(f"Downloaded: {audio_file}")
        except Exception as e:
            tqdm.write(f"Failed to download {audio_url}: {e}")
    emit("feed", "All episodes processed.")
