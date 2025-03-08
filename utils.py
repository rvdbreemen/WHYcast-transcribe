# ...existing code...

def cleanup_resources(model=None):
    """
    Clean up resources to prevent memory leaks, particularly important for CUDA.
    Args:
        model: WhisperModel instance to clean up
    """
    # Force garbage collection
    gc.collect()
    
    # Special handling for CUDA devices
    if 'cuda' in DEVICE.lower():
        try:
            import torch
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("CUDA cache cleared")
        except ImportError:
            logging.warning("Torch is not installed. CUDA cleanup skipped.")
        except Exception as e:
            logging.warning(f"Error while cleaning up CUDA resources: {str(e)}")

# ...existing code...
