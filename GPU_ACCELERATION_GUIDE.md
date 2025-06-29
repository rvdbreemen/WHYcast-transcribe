# GPU Acceleration Setup and Verification Guide

## Overview

This document provides a comprehensive guide for setting up and verifying GPU acceleration for the WHYcast transcription pipeline. The codebase has been updated to aggressively use CUDA (GPU) acceleration for both Whisper transcription and pyannote speaker diarization when available.

## Recent Improvements

### 1. Enhanced Whisper Model Setup (`setup_model()`)

The `setup_model()` function has been completely refactored to:

- **Aggressively select GPU**: Uses the new `force_cuda_device()` function to maximize GPU utilization
- **MAXIMUM worker threads**: Dynamically calculates worker count based on GPU multiprocessors (up to 32 workers for high-end GPUs)
- **Automatic optimization**: Automatically selects optimal compute types (`float16` for GPU, `int8` for CPU)
- **Memory-aware configuration**: Uses 95% of GPU memory for maximum throughput
- **TensorFloat-32 acceleration**: Enables TF32 for improved performance on Ampere GPUs
- **Robust error handling**: Graceful fallback to CPU with detailed error reporting
- **Comprehensive logging**: Provides detailed diagnostics for troubleshooting

### 2. Maximum GPU Utilization (`maximize_gpu_utilization()`)

New function that applies aggressive GPU settings:

- **Environment variables**: Sets optimal CUDA environment variables for maximum performance
- **Memory allocation**: Uses 95% of GPU memory with expandable segments
- **Threading**: Uses all CPU cores for PyTorch operations
- **Caching**: Enables aggressive CUDA caching and optimization
- **TensorFloat-32**: Enables TF32 for both cuDNN and matrix operations

### 2. Robust Device Selection (`force_cuda_device()`)

New function that:

- Checks CUDA availability and device count
- Selects the best available GPU (typically `cuda:0`)
- Provides detailed device information logging
- Falls back gracefully to CPU when needed

### 3. Enhanced Diarization

The `diarize_audio()` function already implements:

- **Force CUDA usage**: Automatically moves pipeline and audio tensors to GPU
- **Memory management**: Clears GPU cache before and after processing
- **Detailed diagnostics**: Comprehensive GPU status reporting

### 4. Comprehensive GPU Verification (`verify_gpu_setup()`)

Enhanced verification function that:

- Tests CUDA availability and device properties
- Performs actual GPU operations to verify functionality
- Provides memory recommendations based on GPU capacity
- Offers specific troubleshooting suggestions

## Quick Verification Commands

### Test GPU Setup Functions

```powershell
cd "your-project-directory"
python -c "
from transcribe import force_cuda_device, verify_gpu_setup

# Test device selection
device = force_cuda_device()
print(f'Selected device: {device}')

# Full GPU verification
gpu_info = verify_gpu_setup()
"
```

### Test Whisper Model Setup

```powershell
python -c "
from transcribe import setup_model
import logging
logging.basicConfig(level=logging.INFO)

# Test with small model for quick verification
model = setup_model(model_size='tiny')
print('Model setup successful!')
"
```

## Expected Output for Working GPU Setup

When GPU acceleration is working correctly, you should see:

```text
🔍 Verifying GPU setup...
✅ CUDA available: 1 device(s)
🎯 Primary GPU: NVIDIA GeForce RTX 3080 Laptop GPU
💾 GPU Memory: 16.0 GB
🔧 CUDA Version: 11.8
🔧 cuDNN Version: 90100
🧪 Testing GPU operations...
✅ GPU test passed - Memory used: 15.75 MB
🚀 GPU has sufficient memory for large models
✅ GPU setup verification completed successfully

🚀 MAXIMUM GPU UTILIZATION: All performance settings enabled
🔧 Setting up Whisper model with GPU acceleration...
🚀 MAXIMUM GPU UTILIZATION: 24 workers on NVIDIA GeForce RTX 3080 Laptop GPU
✅ Whisper model loaded on GPU (cuda) with float16 precision

🚀 FORCING MAXIMUM CUDA UTILIZATION for diarization
🎯 MAXIMUM GPU UTILIZATION: NVIDIA GeForce RTX 3080 Laptop GPU (16.0 GB, 48 MPs)
✅ Diarization pipeline using: cuda:0
```

## Troubleshooting GPU Issues

### 1. CUDA Not Available

If you see:
```
❌ CUDA not available
```

**Solutions:**
1. Install NVIDIA GPU drivers from [NVIDIA website](https://www.nvidia.com/drivers/)
2. Install PyTorch with CUDA support:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify CUDA installation:
   ```powershell
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### 2. GPU Memory Issues

For GPUs with limited memory (<4GB):
- Use smaller Whisper models (`tiny`, `base`, `small`)
- Reduce the number of workers in the configuration
- Close other GPU-intensive applications

### 3. Performance Optimization

**Environment Variables** (optional but recommended):
```powershell
# Force GPU usage
$env:CUDA_VISIBLE_DEVICES="0"

# Optimize memory usage
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

## Configuration Files

### Update `config.py` (if needed)

```python
# Force GPU device selection
DEVICE = "cuda"  # or "auto" for automatic selection

# Optimize compute type for GPU
COMPUTE_TYPE = "float16"  # Best for GPU, use "int8" for CPU
```

### Update `.env` file (optional)

```env
# Force CUDA usage
CUDA_VISIBLE_DEVICES=0

# Hugging Face token for pyannote (required)
HUGGINGFACE_TOKEN=your_token_here
```

## Best Practices

### 1. Pre-flight Checks

Always run the verification before processing:
```python
from transcribe import verify_gpu_setup
gpu_info = verify_gpu_setup()
```

### 2. Memory Management

The code automatically:
- Clears GPU cache before diarization and transcription
- Frees memory after each major operation
- Monitors memory usage during processing

### 3. Model Size Selection

Choose Whisper model size based on GPU memory:
- **16GB+**: `large-v3`, `large-v2`
- **8-16GB**: `medium`, `small`
- **4-8GB**: `small`, `base`
- **<4GB**: `base`, `tiny`

## Integration with Existing Workflow

The enhanced GPU acceleration is automatically activated in the main workflow:

1. **Step 1**: Audio preparation
2. **Step 2**: GPU verification + Diarization with GPU acceleration
3. **Step 3**: Whisper transcription with GPU acceleration + Memory cleanup
4. **Step 4**: Transcript processing

No changes are needed to your existing workflow - GPU acceleration is enabled automatically when available.

## Performance Impact

Expected performance improvements with GPU acceleration:

- **Whisper Transcription**: 3-10x faster depending on model size
- **Speaker Diarization**: 2-5x faster depending on audio length
- **Overall Pipeline**: 2-6x faster end-to-end processing

## Verification Script

Create a simple test script to verify everything works:

```python
#!/usr/bin/env python3
"""GPU Acceleration Verification Script for WHYcast Transcription"""

import logging
from transcribe import verify_gpu_setup, force_cuda_device, setup_model

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("=== WHYcast GPU Acceleration Verification ===\n")
    
    # Step 1: Verify GPU setup
    print("1. Verifying GPU setup...")
    gpu_info = verify_gpu_setup()
    
    # Step 2: Test device selection
    print("\n2. Testing device selection...")
    device = force_cuda_device()
    print(f"Selected device: {device}")
    
    # Step 3: Test Whisper model setup
    print("\n3. Testing Whisper model setup...")
    try:
        model = setup_model(model_size='tiny')
        print("✅ Whisper model setup successful!")
        del model
    except Exception as e:
        print(f"❌ Whisper model setup failed: {e}")
    
    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    main()
```

Save as `test_gpu.py` and run with:
```powershell
python test_gpu.py
```

## Support and Documentation

For additional support:

1. **Whisper GPU Setup**: [OpenAI Whisper GitHub](https://github.com/openai/whisper)
2. **PyTorch CUDA**: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
3. **pyannote GPU**: [pyannote.audio Documentation](https://github.com/pyannote/pyannote-audio)

The codebase now provides comprehensive GPU acceleration with robust error handling and detailed diagnostics to ensure optimal performance for both Whisper transcription and pyannote speaker diarization.
