#!/usr/bin/env python3
"""
Test actual GPU utilization during transcription with a real audio file.
This script will monitor GPU memory and run transcription to verify GPU acceleration.
"""

import os
import sys
import time
import threading
import torch
from typing import Dict, Any

# Add the current directory to the path to import transcribe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def monitor_gpu_memory(monitor_seconds: int = 60, interval: float = 2.0) -> Dict[str, Any]:
    """Monitor GPU memory usage during transcription."""
    gpu_stats = {
        'max_memory_used': 0.0,
        'max_memory_cached': 0.0,
        'memory_samples': [],
        'total_samples': 0,
        'monitoring': True,
        'start_memory': 0.0
    }
    
    def monitor_loop():
        print("\n🔍 Starting GPU memory monitoring...")
        if torch.cuda.is_available():
            gpu_stats['start_memory'] = torch.cuda.memory_allocated(0) / 1024**3
        
        start_time = time.time()
        
        while gpu_stats['monitoring'] and (time.time() - start_time) < monitor_seconds:
            try:
                if torch.cuda.is_available():
                    # PyTorch GPU memory
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved(0) / 1024**3   # GB
                    
                    gpu_stats['max_memory_used'] = max(gpu_stats['max_memory_used'], memory_allocated)
                    gpu_stats['max_memory_cached'] = max(gpu_stats['max_memory_cached'], memory_cached)
                    gpu_stats['memory_samples'].append(memory_cached)
                    
                    print(f"  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
                    
                    gpu_stats['total_samples'] += 1
                    
            except Exception as e:
                print(f"  Warning: Could not read GPU stats: {e}")
            
            time.sleep(interval)
    
    # Start monitoring in background thread
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    
    return gpu_stats

def test_transcription_with_monitoring():
    """Test transcription with real GPU monitoring."""
    print("🚀 GPU Memory Test for WHYcast Transcribe")
    print("=" * 60)
    
    # Check if we have a CUDA GPU
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Cannot test GPU utilization.")
        return False
    
    print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
    print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU Memory: {total_memory:.1f}GB")
    except:
        print("💾 GPU Memory: Unable to determine")
    
    # Find a small audio file to test with
    podcasts_dir = "podcasts"
    test_files = []
    
    if os.path.exists(podcasts_dir):
        for file in os.listdir(podcasts_dir):
            if file.endswith(('.mp3', '.m4a', '.wav')):
                file_path = os.path.join(podcasts_dir, file)
                file_size = os.path.getsize(file_path) / 1024**2  # MB
                if file_size < 50:  # Use files smaller than 50MB for quick test
                    test_files.append((file, file_size))
    
    if not test_files:
        print("❌ No suitable test audio files found in podcasts directory")
        print("   Looking for audio files smaller than 50MB...")
        return False
    
    # Sort by size and pick the smallest
    test_files.sort(key=lambda x: x[1])
    test_file, file_size = test_files[0]
    test_path = os.path.join(podcasts_dir, test_file)
    
    print(f"🎵 Using test file: {test_file} ({file_size:.1f}MB)")
    
    # Start GPU monitoring
    monitor_time = max(60, int(file_size * 3))  # Estimate monitoring time
    gpu_stats = monitor_gpu_memory(monitor_seconds=monitor_time)
    
    try:
        print(f"\n🎯 Starting transcription...")
        start_time = time.time()
        
        # Import and run transcription
        from transcribe import full_workflow, maximize_gpu_utilization, verify_gpu_setup
        
        # Force GPU optimizations
        print("🔧 Maximizing GPU utilization...")
        maximize_gpu_utilization()
        verify_gpu_setup()
        
        # Run transcription with GPU acceleration
        print(f"   Transcribing: {test_path}")
        
        # Run the transcription directly
        success = full_workflow(
            audio_file=test_path,
            output_dir=os.path.dirname(test_path),
            rssfeed=None,
            force=False
        )
        
        transcription_time = time.time() - start_time
        
        # Stop monitoring
        gpu_stats['monitoring'] = False
        time.sleep(2)  # Wait for monitor thread to finish
        
        print(f"\n⏱️  Transcription completed in {transcription_time:.1f} seconds")
        
        # Report GPU memory usage
        print(f"\n📊 GPU Memory Usage Results:")
        print(f"  Starting Memory: {gpu_stats['start_memory']:.2f}GB")
        print(f"  Max Memory Allocated: {gpu_stats['max_memory_used']:.2f}GB")
        print(f"  Max Memory Cached: {gpu_stats['max_memory_cached']:.2f}GB")
        print(f"  Total Samples: {gpu_stats['total_samples']}")
        
        # Determine if GPU was properly utilized
        memory_increase = gpu_stats['max_memory_cached'] - gpu_stats['start_memory']
        
        if memory_increase > 0.5:  # At least 500MB increase
            print("✅ GPU memory was utilized properly")
            print(f"   Memory increase: +{memory_increase:.2f}GB")
        elif memory_increase > 0.1:  # At least 100MB increase
            print("⚠️  Moderate GPU memory usage")
            print(f"   Memory increase: +{memory_increase:.2f}GB")
        else:
            print("❌ Low GPU memory usage - may be using CPU fallback")
            print(f"   Memory increase: +{memory_increase:.2f}GB")
        
        return success
        
    except Exception as e:
        gpu_stats['monitoring'] = False
        print(f"❌ Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = test_transcription_with_monitoring()
    
    if success:
        print("\n🎉 GPU memory test completed successfully!")
        print("   Check the memory usage above to verify GPU acceleration")
    else:
        print("\n❌ GPU memory test failed")
    
    sys.exit(0 if success else 1)
