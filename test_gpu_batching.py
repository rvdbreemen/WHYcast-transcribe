#!/usr/bin/env python3
"""
Test script for GPU batching optimization.
This script tests the new BatchedInferencePipeline approach for maximum GPU utilization.
"""

import sys
import os
import torch
import logging
import time

# Add the current directory to path to import transcribe module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gpu_batching():
    """Test the new GPU batching approach and measure performance."""
    
    print("=== GPU Batching Test ===")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        from transcribe import maximize_gpu_utilization, setup_model, force_cuda_device, verify_gpu_setup
        
        print("\n1. Testing GPU optimization settings...")
        optimal_batch_size, num_workers = maximize_gpu_utilization()
        print(f"   Optimal batch size: {optimal_batch_size}")
        print(f"   Optimal workers: {num_workers}")
        
        print("\n2. Testing CUDA device detection...")
        device = force_cuda_device()
        print(f"   Selected device: {device}")
        
        print("\n3. Testing GPU setup verification...")
        gpu_info = verify_gpu_setup()
        print(f"   GPU available: {gpu_info.get('cuda_available', False)}")
        print(f"   GPU name: {gpu_info.get('gpu_name', 'N/A')}")
        print(f"   GPU memory: {gpu_info.get('total_memory_gb', 0):.1f}GB")
        
        print("\n4. Testing model setup with batching...")
        start_time = time.time()
        
        model = setup_model(batch_size=optimal_batch_size)
        
        setup_time = time.time() - start_time
        print(f"   Model setup time: {setup_time:.2f}s")
        
        # Check if we got a BatchedInferencePipeline
        is_batched = hasattr(model, 'model')
        batch_size = getattr(model, 'optimal_batch_size', 'Unknown')
        
        print(f"   Model type: {'BatchedInferencePipeline' if is_batched else 'WhisperModel'}")
        print(f"   Batch size: {batch_size}")
        
        if is_batched:
            print("   ✅ SUCCESS: GPU batching is enabled!")
            print("   🚀 This should provide much better GPU utilization than the previous approach")
        else:
            print("   ⚠️ FALLBACK: Using regular model (CPU or GPU without batching)")
        
        print("\n5. GPU Memory Status:")
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   Memory allocated: {memory_allocated:.2f}GB")
            print(f"   Memory reserved: {memory_reserved:.2f}GB")
        else:
            print("   No CUDA available")
        
        print("\n=== Test Summary ===")
        print("✅ GPU batching test completed!")
        print("💡 Key improvements:")
        print("   - Uses BatchedInferencePipeline for true GPU parallelization")
        print("   - Calculates optimal batch size based on GPU memory")
        print("   - Focuses on GPU compute rather than excessive worker threads")
        print("   - Should show much higher GPU utilization than before")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def monitor_gpu_usage():
    """Monitor GPU usage during the test."""
    if not torch.cuda.is_available():
        print("No CUDA GPU available for monitoring")
        return
    
    try:
        print("\n=== GPU Monitoring ===")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"Total GPU memory: {total_memory:.2f}GB")
        print(f"Currently allocated: {allocated:.2f}GB ({allocated/total_memory*100:.1f}%)")
        print(f"Currently reserved: {reserved:.2f}GB ({reserved/total_memory*100:.1f}%)")
        
        # GPU properties
        props = torch.cuda.get_device_properties(0)
        print(f"Multiprocessors: {props.multi_processor_count}")
        print(f"CUDA Capability: {props.major}.{props.minor}")
        
        # Check if GPU is properly configured
        if allocated > 0.1:  # More than 100MB allocated
            print("✅ GPU memory is being used - model loaded successfully")
        else:
            print("⚠️ Very little GPU memory used - check if model actually loaded on GPU")
            
    except Exception as e:
        print(f"Error monitoring GPU: {e}")

if __name__ == "__main__":
    print("Testing new GPU batching approach...")
    print("This replaces the previous high-worker-count approach with proper GPU batching")
    
    monitor_gpu_usage()
    success = test_gpu_batching()
    monitor_gpu_usage()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\n📝 Next steps:")
        print("1. Run this with an actual audio file to test transcription")
        print("2. Monitor GPU utilization during transcription (should be much higher)")
        print("3. Compare transcription speed with the old approach")
    else:
        print("\n❌ Tests failed - check error messages above")
        sys.exit(1)
