import platform
import torch

def get_best_device():
    """
    Detects the best available device for ML acceleration.
    Returns: 'cuda', 'mps', or 'cpu'
    """
    system = platform.system()
    if system in ("Windows", "Linux"):
        if torch.cuda.is_available():
            return "cuda"
    elif system == "Darwin":
        # macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"
