"""
GPU setup and utilization helpers for the WHYcast pipeline (ADR-008).

Extracted verbatim from transcribe.py (lines 1924-1989 and 2874-3057) during
the ADR-008 library extraction. enable_tf32() wraps the module-level TF32
block from transcribe.py lines 49-56 so it no longer runs at import time.
"""

import logging

from whycast.events import emit

logger = logging.getLogger(__name__)


def enable_tf32():
    """Enable TensorFloat-32 on Ampere-or-newer GPUs (moved from module level, ADR-008)."""
    import torch
    # Enable TensorFloat-32 for improved performance on NVIDIA Ampere GPUs
    if torch.cuda.is_available():
        # Check if we have an Ampere or newer GPU
        compute_capability = torch.cuda.get_device_capability(0)
        if compute_capability[0] >= 8:  # Ampere GPUs have compute capability 8.0+
            logging.info("Enabling TensorFloat-32 for improved performance on Ampere GPU")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

def is_cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Log CUDA device information for better diagnostics
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
            logging.info(f"CUDA is available: {device_count} device(s) - {device_name}")
        return cuda_available
    except ImportError:
        return False

def get_default_device() -> str:
    """Return 'cuda' if CUDA is available else 'cpu'."""
    return "cuda" if is_cuda_available() else "cpu"

def force_cuda_device() -> str:
    """
    Aggressively try to select CUDA device for maximum GPU utilization.
    
    This function checks for CUDA availability and selects the best CUDA device.
    If CUDA is not available, it falls back to CPU but logs appropriate warnings.
    
    Returns:
        str: The selected device string ('cuda', 'cuda:0', or 'cpu')
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            logging.warning("🚫 CUDA not available - falling back to CPU")
            return "cpu"
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            logging.warning("🚫 No CUDA devices found - falling back to CPU")
            return "cpu"
        
        # Select the best available GPU (typically GPU 0)
        selected_device = "cuda:0" if device_count > 0 else "cuda"
        
        # Log device selection details
        try:
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"🎯 Selected GPU device: {selected_device} ({device_name}, {device_memory:.1f}GB)")
        except Exception as e:
            logging.warning(f"Could not get GPU device details: {e}")
            logging.info(f"🎯 Selected GPU device: {selected_device}")
        
        return selected_device
        
    except ImportError:
        logging.error("❌ PyTorch not available - falling back to CPU")
        return "cpu"
    except Exception as e:
        logging.error(f"❌ Error in device selection: {e} - falling back to CPU")
        return "cpu"

def verify_gpu_setup():
    """
    Comprehensive verification of GPU setup for both Whisper and pyannote.
    This function checks CUDA availability and performs test operations.
    
    Returns:
        dict: GPU status information
    """
    try:
        import torch
        
        gpu_info = {
            'cuda_available': False,
            'device_count': 0,
            'device_name': None,
            'device_memory': 0,
            'pytorch_version': torch.__version__,
            'cuda_version': None,
            'cudnn_version': None,
            'test_passed': False,
            'recommendations': []
        }
        
        emit("gpu", "🔍 Verifying GPU setup...")
        logging.info("Starting GPU verification")
        
        # Basic CUDA availability check
        cuda_available = torch.cuda.is_available()
        gpu_info['cuda_available'] = cuda_available
        
        if not cuda_available:
            emit("gpu", "❌ CUDA not available")
            gpu_info['recommendations'].append("Install NVIDIA GPU drivers")
            gpu_info['recommendations'].append("Install PyTorch with CUDA support")
            gpu_info['recommendations'].append("Check CUDA_PATH environment variable")
            return gpu_info
        
        # Get device information
        device_count = torch.cuda.device_count()
        gpu_info['device_count'] = device_count
        
        if device_count > 0:
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info['device_name'] = device_name
            gpu_info['device_memory'] = device_memory
            
            emit("gpu", f"✅ CUDA available: {device_count} device(s)")
            emit("gpu", f"🎯 Primary GPU: {device_name}")
            emit("gpu", f"💾 GPU Memory: {device_memory:.1f} GB")
            
            # Get CUDA version information
            try:
                cuda_version = torch.version.cuda
                gpu_info['cuda_version'] = cuda_version
                emit("gpu", f"🔧 CUDA Version: {cuda_version}")
            except:
                emit("gpu", "⚠️  CUDA version unknown")
            
            try:
                cudnn_version = torch.backends.cudnn.version()
                gpu_info['cudnn_version'] = cudnn_version
                emit("gpu", f"🔧 cuDNN Version: {cudnn_version}")
            except:
                emit("gpu", "⚠️  cuDNN version unknown")
        
        # Perform test operations
        try:
            emit("gpu", "🧪 Testing GPU operations...")
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Test tensor creation and operations
            test_tensor = torch.ones(1000, 1000, device='cuda')
            result = test_tensor @ test_tensor  # Matrix multiplication
            
            # Check memory usage
            memory_used = torch.cuda.memory_allocated(0) / (1024**2)
            emit("gpu", f"✅ GPU test passed - Memory used: {memory_used:.2f} MB")
            
            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()
            
            gpu_info['test_passed'] = True
            
        except Exception as e:
            emit("gpu", f"❌ GPU test failed: {e}")
            gpu_info['recommendations'].append("Check GPU compatibility")
            gpu_info['recommendations'].append("Verify PyTorch CUDA installation")
        
        # Memory recommendations
        if gpu_info['device_memory'] < 4:
            gpu_info['recommendations'].append("GPU has limited memory (<4GB) - consider using smaller models")
        elif gpu_info['device_memory'] >= 8:
            emit("gpu", "🚀 GPU has sufficient memory for large models")
        
        # Final status
        if gpu_info['test_passed']:
            emit("gpu", "✅ GPU setup verification completed successfully")
            logging.info("GPU setup verification passed")
        else:
            emit("gpu", "⚠️  GPU setup has issues - check recommendations")
            logging.warning("GPU setup verification failed")
        
        return gpu_info
        
    except ImportError:
        emit("gpu", "❌ PyTorch not available")
        return {'cuda_available': False, 'recommendations': ['Install PyTorch']}
    except Exception as e:
        emit("gpu", f"❌ GPU verification error: {e}")
        logging.error(f"GPU verification error: {e}")
        return {'cuda_available': False, 'recommendations': ['Check PyTorch installation']}

def maximize_gpu_utilization():
    """
    Configure optimal GPU settings for Whisper inference.
    Focus on proper batching rather than excessive worker threads.
    """
    import os
    import torch
    
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            
            logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logging.info(f"GPU memory: {gpu_memory:.1f}GB")
            
            # Calculate optimal batch size based on GPU memory
            # RTX 3080 (10GB): batch_size=8-16
            # RTX 4090 (24GB): batch_size=16-32
            if gpu_memory >= 20:
                optimal_batch_size = 16
                num_workers = 8
            elif gpu_memory >= 8:
                optimal_batch_size = 8
                num_workers = 6
            else:
                optimal_batch_size = 4
                num_workers = 4
            
            logging.info(f"Optimal batch size: {optimal_batch_size}")
            logging.info(f"Optimal workers: {num_workers}")
            
            # Set reasonable environment variables for GPU optimization
            env_vars = {
                'OMP_NUM_THREADS': str(num_workers),
                'MKL_NUM_THREADS': str(num_workers),
                'NUMBA_NUM_THREADS': str(num_workers),
                'CUDA_LAUNCH_BLOCKING': '0',
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'CUDA_VISIBLE_DEVICES': '0',
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,expandable_segments:True',
                'TORCH_NUM_THREADS': str(num_workers),
            }
            
            for key, value in env_vars.items():
                os.environ[key] = value
                logging.info(f"Set {key}={value}")
            
            # PyTorch optimizations for GPU inference
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                
            torch.set_num_threads(num_workers)
            
            logging.info("✅ GPU optimization settings applied!")
            
            return optimal_batch_size, num_workers
        else:
            logging.warning("❌ No CUDA GPU detected")
            return 4, 4
            
    except Exception as e:
        logging.error(f"❌ Error setting GPU optimization: {e}")
        return 4, 4

