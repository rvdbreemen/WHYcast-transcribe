"""
Before-snapshots of an episode's artifacts, so a re-run can be diffed.

Why this is not a ``.bak``
-------------------------
ADR-009 removed per-artifact backups: a full re-run of ~48 episodes doubled the
file count in ``podcasts/`` and buried the corpus under ``.bak`` rows nobody
read. That decision is unchanged here. What it also removed, though, was the
cheap way to answer "what did this re-run actually change?" - and that question
is the whole point of the diff view (TASK-005).

So the copy lives with the *job*, not with the artifact:

    logs/jobs/<job_id>/before/<filename>

which keeps three properties the ``.bak`` did not have. ``podcasts/`` stays
exactly as clean as ADR-009 made it. The copy is bounded by job-log retention
rather than living forever beside the artifact. And it is evidence about one
run, next to that run's event log, which is where somebody looking at a job
already is.

Snapshots are best-effort by design: failing to copy a file for later
inspection must never abort a job that would otherwise have run. A missing
snapshot degrades the diff view to "no previous version recorded", which is
honest and harmless.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from webui import jobs

logger = logging.getLogger(__name__)

__all__ = [
    "snapshot_dir",
    "manifest_path",
    "take_snapshot",
    "load_manifest",
    "snapshot_file",
]

#: Name of the directory inside a job's log directory holding the copies.
SNAPSHOT_DIRNAME = "before"

#: Name of the manifest describing what was copied.
MANIFEST_NAME = "manifest.json"

#: Artifacts larger than this are recorded in the manifest but not copied. A
#: diff of a 40 MB file helps nobody, and the job log directory is not the place
#: to grow one. Text artifacts are kilobytes; this only ever excludes accidents.
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024


def snapshot_dir(job_id: str) -> str:
    """Directory holding this job's before-copies. Not created by this call."""
    return os.path.join(jobs.jobs_dir(), job_id, SNAPSHOT_DIRNAME)


def manifest_path(job_id: str) -> str:
    """Path of the manifest describing this job's snapshot."""
    return os.path.join(snapshot_dir(job_id), MANIFEST_NAME)


def snapshot_file(job_id: str, filename: str) -> str:
    """Path of one snapshotted file inside this job's snapshot directory.

    ``filename`` is a bare basename taken from the manifest, never from a
    request; callers that accept user input must look the name up in the
    manifest first.
    """
    return os.path.join(snapshot_dir(job_id), os.path.basename(filename))


def take_snapshot(
    job_id: str,
    base_name: str,
    artifacts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Copy an episode's current artifacts aside before a job overwrites them.

    Args:
        job_id: the job about to run; the snapshot lives in its log directory.
        base_name: the episode, recorded in the manifest for the diff view.
        artifacts: artifact rows as the index reports them - dicts carrying at
            least ``kind``, ``fmt`` and ``path``.

    Returns:
        The manifest that was written, or None when nothing could be recorded.
        Never raises: a job must not fail because its snapshot did not.
    """
    entries: List[Dict[str, Any]] = []
    directory = snapshot_dir(job_id)
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        logger.warning("No snapshot for job %s: %s", job_id, exc)
        return None

    for artifact in artifacts:
        path = artifact.get("path")
        if not path or not os.path.isfile(path):
            continue
        entry = {
            "kind": artifact.get("kind"),
            "fmt": artifact.get("fmt"),
            "filename": os.path.basename(path),
            "source": path,
        }
        try:
            size = os.path.getsize(path)
            entry["size"] = size
            if size > MAX_SNAPSHOT_BYTES:
                entry["copied"] = False
                entry["reason"] = f"larger than {MAX_SNAPSHOT_BYTES} bytes"
            else:
                shutil.copy2(path, snapshot_file(job_id, entry["filename"]))
                entry["copied"] = True
        except OSError as exc:
            entry["copied"] = False
            entry["reason"] = str(exc)
            logger.warning("Could not snapshot %s for job %s: %s", path, job_id, exc)
        entries.append(entry)

    manifest = {"job_id": job_id, "base_name": base_name, "artifacts": entries}
    try:
        with open(manifest_path(job_id), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning("Could not write the snapshot manifest for %s: %s", job_id, exc)
        return None
    return manifest


def load_manifest(job_id: str) -> Optional[Dict[str, Any]]:
    """Read a job's snapshot manifest, or None when it has none."""
    try:
        with open(manifest_path(job_id), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    return manifest
