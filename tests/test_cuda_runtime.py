"""
CUDA runtime regression tests (ADR-001).

The failure these guard against: CTranslate2 is built against CUDA 12 and loads
cublas64_12.dll lazily, on the first GPU compute call. PyTorch ships CUDA 11
(cublas64_11.dll), so a torch+cu118 environment does not satisfy it, and
whycast.cuda_setup.ensure_cuda_libs() has to make the pip-installed CUDA 12
wheels findable.

The trap: constructing WhisperModel(device="cuda") succeeds even when cuBLAS 12
cannot be loaded - the error only appears once audio is actually transcribed.
A test that stops at model construction proves nothing, so the GPU test below
transcribes real audio.
"""

import os
import shutil
import subprocess
import sys
import wave
import math
import struct

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from whycast.cuda_setup import ensure_cuda_libs, nvidia_library_dirs  # noqa: E402


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def test_ensure_cuda_libs_is_idempotent():
    """Repeated calls must not keep appending to PATH."""
    ensure_cuda_libs()
    path_after_first = os.environ.get("PATH", "")
    assert ensure_cuda_libs() == []
    assert os.environ.get("PATH", "") == path_after_first


@pytest.mark.skipif(sys.platform != "win32", reason="DLL directory handling is Windows-only")
def test_cuda_libs_land_on_path_when_wheels_installed():
    dirs = nvidia_library_dirs()
    if not dirs:
        pytest.skip("nvidia-* CUDA wheels not installed in this environment")
    ensure_cuda_libs()
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for directory in dirs:
        assert directory in path_entries, (
            f"{directory} missing from PATH; CTranslate2's lazy cuBLAS load "
            "consults PATH, not add_dll_directory"
        )


@pytest.mark.skipif(sys.platform != "win32", reason="cublas64_12 naming is Windows-specific")
def test_cublas12_is_reachable():
    """The CUDA 12 cuBLAS that CTranslate2 needs must exist somewhere on PATH."""
    if not nvidia_library_dirs():
        pytest.skip("nvidia-* CUDA wheels not installed in this environment")
    ensure_cuda_libs()
    assert shutil.which("cublas64_12.dll") or any(
        os.path.exists(os.path.join(d, "cublas64_12.dll")) for d in nvidia_library_dirs()
    ), "cublas64_12.dll not found; run: pip install nvidia-cublas-cu12"


def _write_tone_wav(path: str, seconds: float = 1.0, rate: int = 16000) -> str:
    """A tiny silent-ish WAV; content does not matter, only that decoding works."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(1000 * math.sin(2 * math.pi * 220 * t / rate)))
            for t in range(int(rate * seconds))
        )
        w.writeframes(frames)
    return path


@pytest.mark.slow
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA GPU available")
def test_real_transcription_runs_on_gpu(tmp_path):
    """
    End-to-end GPU proof: transcribe actual audio.

    Runs in a subprocess so a hard CUDA/DLL failure cannot poison the pytest
    process, and so the import order (ensure_cuda_libs before faster_whisper)
    is exercised exactly as production does it.
    """
    audio = _write_tone_wav(str(tmp_path / "tone.wav"))
    script = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from whycast.pipeline.transcription import setup_model\n"
        "from faster_whisper import WhisperModel\n"
        "m = WhisperModel('tiny', device='cuda', compute_type='float16')\n"
        "segments, info = m.transcribe(r'%s', beam_size=1)\n"
        "list(segments)\n"  # forces the lazy cuBLAS load - the actual assertion
        "print('GPU_TRANSCRIBE_OK')\n" % (REPO_ROOT, audio)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600, cwd=REPO_ROOT,
    )
    assert "GPU_TRANSCRIBE_OK" in result.stdout, (
        "GPU transcription failed - CUDA runtime is not usable.\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    )
