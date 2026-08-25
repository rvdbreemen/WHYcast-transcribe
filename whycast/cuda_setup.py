"""
CUDA library discovery for CTranslate2 / faster-whisper (ADR-001, ADR-008).

Why this exists
---------------
faster-whisper runs on CTranslate2, which is built against CUDA 12 and loads
cublas64_12.dll at runtime. PyTorch (used for diarization) ships its own CUDA
11 runtime (cublas64_11.dll) in torch/lib, so a torch+cu118 install does not
satisfy CTranslate2: transcription failed with

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

The CUDA 12 runtime is installed as pip wheels (nvidia-cublas-cu12 and
friends), which unpack into site-packages/nvidia/<component>/bin on Windows and
site-packages/nvidia/<component>/lib on Linux. On Linux the loader uses the
wheels' own RPATH, so this module is a no-op there.

On Windows both mechanisms below are required, and PATH is the load-bearing one:

* os.add_dll_directory() covers DLLs resolved through the Python extension
  loader (LOAD_LIBRARY_SEARCH_* flags).
* CTranslate2 loads cuBLAS *lazily*, on the first GPU compute call, through a
  delay-loaded import that uses the classic Win32 search order - which consults
  PATH and ignores add_dll_directory. Measured 2026-08-24: with only
  add_dll_directory, WhisperModel(device="cuda") constructs fine and then
  transcribe() dies with the cublas64_12 error; prepending the directories to
  PATH makes a real transcription succeed. Registering only the DLL directory
  therefore looks like it works right up until actual audio is transcribed.

The two CUDA runtimes coexist because their file names differ (cublas64_11 for
torch, cublas64_12 for CTranslate2); nothing is overridden.

Call ensure_cuda_libs() before importing faster_whisper / ctranslate2.
It is idempotent, cheap, and does not initialize the GPU or load any model,
so it is safe at import time (ADR-008: no side effects).
"""

import glob
import logging
import os
import sys

logger = logging.getLogger(__name__)

_registered = False


def nvidia_library_dirs() -> list:
    """Return the CUDA runtime directories shipped by the nvidia-* pip wheels."""
    base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
    if not os.path.isdir(base):
        # Non-Windows layout, or a venv whose site-packages sits elsewhere.
        try:
            import nvidia  # type: ignore[import-not-found]
        except ImportError:
            return []
        base = os.path.dirname(nvidia.__file__)
    dirs = []
    for sub in ("bin", "lib"):
        dirs.extend(sorted(glob.glob(os.path.join(base, "*", sub))))
    return [d for d in dirs if os.path.isdir(d)]


def ensure_cuda_libs() -> list:
    """
    Make the pip-installed CUDA 12 runtime discoverable for CTranslate2.

    Returns the directories that were registered (empty on non-Windows or when
    the nvidia wheels are not installed). Safe to call repeatedly.
    """
    global _registered
    if _registered or not hasattr(os, "add_dll_directory"):
        return []

    dirs = nvidia_library_dirs()
    for directory in dirs:
        try:
            os.add_dll_directory(directory)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not register CUDA library dir %s: %s", directory, exc)

    # PATH is what the lazy cuBLAS load actually consults; without this the
    # failure only surfaces once real audio is transcribed (see module docstring).
    if dirs:
        current = os.environ.get("PATH", "")
        missing = [d for d in dirs if d not in current.split(os.pathsep)]
        if missing:
            os.environ["PATH"] = os.pathsep.join(missing + ([current] if current else []))
    if dirs:
        logger.debug("Registered CUDA library dirs: %s", dirs)
    else:
        logger.debug(
            "No nvidia-* CUDA runtime wheels found; CTranslate2 will rely on a "
            "system CUDA installation (pip install nvidia-cublas-cu12 if it fails)."
        )
    _registered = True
    return dirs
