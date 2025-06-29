#!/usr/bin/env python3
"""GPU Acceleration Verification Script for WHYcast Transcription"""

import logging
import sys
import os

# Add the current directory to the path to import transcribe module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main verification function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== WHYcast GPU Acceleration Verification ===\n")
    
    try:
        from transcribe import verify_gpu_setup, force_cuda_device, setup_model
    except ImportError as e:
        print(f"❌ Failed to import transcribe module: {e}")
        print("Make sure you're running this script from the project directory.")
        return False
    
    success = True
    
    # Step 1: Verify GPU setup
    print("1. Verifying GPU setup...")
    try:
        gpu_info = verify_gpu_setup()
        print("   ✅ GPU verification completed")
    except Exception as e:
        print(f"   ❌ GPU verification failed: {e}")
        success = False
    
    print()
    
    # Step 2: Test device selection
    print("2. Testing device selection...")
    try:
        device = force_cuda_device()
        print(f"   ✅ Selected device: {device}")
    except Exception as e:
        print(f"   ❌ Device selection failed: {e}")
        success = False
    
    print()
    
    # Step 3: Test Whisper model setup (lightweight test)
    print("3. Testing Whisper model setup...")
    try:
        # Use tiny model for quick testing
        model = setup_model(model_size='tiny')
        print("   ✅ Whisper model setup successful!")
        
        # Clean up
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   ✅ GPU memory cleaned up")
        except:
            pass
            
    except Exception as e:
        print(f"   ❌ Whisper model setup failed: {e}")
        success = False
    
    print()
    
    # Summary
    if success:
        print("🎉 === Verification Complete - GPU Acceleration Ready! ===")
        print("Your system is properly configured for GPU-accelerated transcription.")
    else:
        print("⚠️  === Verification Issues Detected ===")
        print("Some components failed verification. Check the errors above.")
        print("Refer to GPU_ACCELERATION_GUIDE.md for troubleshooting steps.")
    
    return success

if __name__ == "__main__":
    main()
