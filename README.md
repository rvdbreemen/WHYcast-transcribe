# WHYcast Transcribe

WHYcast Transcribe is a tool for transcribing audio files and generating summaries, blogs, and history extractions from podcast episodes using OpenAI GPT models. It's designed specifically for the WHYcast podcast series focusing on hacker events, but can be used for any audio content.

## Features

- Transcribe audio files using the Whisper model
- Generate summaries and blog posts from transcripts using OpenAI GPT models
- Download the latest episode from a podcast RSS feed
- Apply custom vocabulary corrections to transcripts
- Batch processing of multiple audio files
- Regenerate summaries from existing transcripts
- Speaker diarization to identify different speakers
- Timestamped transcripts for easy reference
- Generate speaker assignment files (text, HTML, and Wiki)

## Project Structure

```
WHYcast-transcribe/
├── check_cuda.py              # Script to verify CUDA availability
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation configuration
├── README.md                  # This documentation
├── logs/                      # Log files directory
├── models/                    # Downloaded model files
│   ├── diarization/           # Speaker diarization models
│   └── whisper/               # Whisper models
├── podcasts/                  # Audio files and generated outputs
├── prompts/                   # LLM prompt templates
├── src/                       # Source code
│   └── whycast_transcribe/    # Main package
│       ├── __init__.py
│       ├── audio.py           # Audio processing utilities
│       ├── blog_generator.py  # Blog post generation
│       ├── cleanup.py         # Transcript cleanup
│       ├── cli.py             # Command line interface
│       ├── config.py          # Configuration handling
│       ├── content_generator.py  # Content generation utilities
│       ├── diarize.py         # Speaker diarization
│       ├── file.py            # File handling utilities
│       ├── formatters.py      # Output formatters
│       ├── history_generator.py  # History item extraction
│       ├── llm.py             # LLM interaction
│       ├── model_manager.py   # Model downloading and management
│       ├── speaker_assigner.py  # Speaker assignment
│       ├── summarizer.py      # Summary generation
│       ├── tokens.py          # Token counting utilities
│       ├── transcriber.py     # Audio transcription
│       └── vocabulary.py      # Custom vocabulary handling
└── tests/                     # Test suite
    ├── test_blog_generator.py
    ├── test_cleanup.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_formatters.py
    ├── test_history_generator.py
    ├── test_integration.py    # End-to-end integration tests
    ├── test_model_manager.py
    ├── test_speaker_assigner.py
    ├── test_summarizer.py
    ├── test_utils_tokens.py
    └── test_vocabulary.py
```

## Requirements
- CUDA enabled GPU (e.g. NVIDIA GeForce RTX 3060) recommended for faster processing
- Python 3.7+
- Required Python packages (listed in `requirements.txt`)
- OpenAI API key

## Installation

### Prerequisites

- Python 3.10 or later
- FFmpeg (for audio processing)
- CUDA-compatible GPU (optional, but recommended for faster processing)

## Installation (Platform-Specific)

### macOS/Linux
1. Open a terminal in the project directory.
2. Run:
   ```bash
   bash install.sh
   ```
   This will:
   - Create a Python virtual environment
   - Install all required dependencies from requirements.txt
   - Install the correct version of torch/torchaudio for your platform (MPS/CPU for Mac, CPU for Linux)
   - Check/install ffmpeg
3. Activate your environment:
   ```bash
   source venv/bin/activate
   ```

### Windows
1. Open Command Prompt in the project directory.
2. Run:
   ```bat
   install.bat
   ```
   This will:
   - Create a Python virtual environment
   - Install all required dependencies from requirements.txt
   - Install the correct version of torch/torchaudio for your platform (CUDA for Windows/NVIDIA)
   - Check for ffmpeg
3. Activate your environment:
   ```bat
   call venv\Scripts\activate
   ```

> **Note:**
> PyTorch (torch, torchaudio) is installed by the install script for your platform. This ensures you get the best performance and compatibility. If you install manually, see https://pytorch.org/get-started/locally/ for the correct command for your system.

## Usage

### Basic Transcription

```bash
whycast-transcribe transcribe podcasts/example.mp3
```

This will create a transcript file `podcasts/example.txt`.

### Transcription with Speaker Diarization

```bash
whycast-transcribe transcribe podcasts/example.mp3 --diarize
```

This will create a transcript with speaker labels.

### Generate Summary

```bash
whycast-transcribe summarize podcasts/example.mp3
```

This will generate a summary from the transcript.

### Generate Blog Post

```bash
whycast-transcribe blog podcasts/example.mp3
```

This creates HTML, text, and Wiki versions of a blog post.

### Extract History Items

```bash
whycast-transcribe history podcasts/example.mp3
```

This extracts history items mentioned in the podcast.

### Batch Processing

```bash
whycast-transcribe batch podcasts/ --type=mp3
```

Process all .mp3 files in the podcasts directory.

### Regenerate Outputs from Existing Transcript

```bash
whycast-transcribe summary --from-transcript podcasts/example.txt
whycast-transcribe blog --from-transcript podcasts/example.txt
```

### Full Processing Workflow

```bash
whycast-transcribe workflow podcasts/example.mp3
```

This will perform the complete workflow: transcription → cleaning → summary → blog → history.

## Command-Line Options

```
Usage: whycast-transcribe [OPTIONS] COMMAND [ARGS]...

Options:
  --debug / --no-debug  Enable debug mode with detailed logging
  --help                Show this message and exit

Commands:
  batch      Process multiple audio files in a directory
  blog       Generate blog post from audio or transcript
  clean      Clean a transcript
  history    Extract history items from audio or transcript
  summarize  Generate summary from audio or transcript
  transcribe Transcribe audio file to text
  workflow   Run the complete processing workflow
```

## Output Files

For each processed audio file, the following outputs are generated:
- `file.txt` - Raw transcript
- `file_ts.txt` - Transcript with timestamps
- `file_cleaned.txt` - Cleaned transcript
- `file_summary.txt` - Summary
- `file_blog.txt` - Blog post (text)
- `file_blog.html` - Blog post (HTML)
- `file_blog.wiki` - Blog post (Wiki markup)
- `file_history.txt` - History extraction (text)
- `file_history.html` - History extraction (HTML)
- `file_history.wiki` - History extraction (Wiki markup)
- `file_speaker_assignment.txt` - Speaker assignment (text)
- `file_speaker_assignment.html` - Speaker assignment (HTML)
- `file_speaker_assignment.wiki` - Speaker assignment (Wiki markup)

## Speaker Diarization

WHYcast Transcribe now supports speaker diarization which identifies and labels different speakers in the audio. This feature adds speaker labels like `[SPEAKER_00]` to the transcript, making it easier to follow conversations in podcasts or interviews.

### Speaker Diarization Options

- `--diarize`: Enable speaker diarization (override config setting)
- `--no-diarize`: Disable speaker diarization (override config setting)
- `--min-speakers`: Minimum number of speakers to identify (default: 1)
- `--max-speakers`: Maximum number of speakers to identify (default: 10)

### Requirements for Speaker Diarization

Speaker diarization requires:
1. PyAnnote Audio library (installed via requirements.txt)
2. A HuggingFace token in your `.env` file:
   ```
   HUGGINGFACE_TOKEN=your_huggingface_token
   ```
3. **Important**: You must also explicitly accept the user conditions for the diarization model by:
   - Log in to your HuggingFace account
   - Visit https://hf.co/pyannote/speaker-diarization-3.1
   - Click the "Access repository" button and accept the terms and conditions
   - This step is mandatory even if you already have a HuggingFace token
   - For better reliability, also visit https://hf.co/pyannote/segmentation-3.0 and accept those terms as well

### Environment Variables for Speaker Diarization

You can configure speaker diarization behavior through the following environment variables in your `.env` file:

```
# Enable or disable speaker diarization (True/False)
USE_SPEAKER_DIARIZATION=True

# Primary diarization model to use
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1

# Alternative model if the primary one is not available
DIARIZATION_ALTERNATIVE_MODEL=pyannote/segmentation-3.0

# Speaker count limits
DIARIZATION_MIN_SPEAKERS=1
DIARIZATION_MAX_SPEAKERS=10
```

### Example Usage

To transcribe an audio file with speaker diarization:
```sh
python transcribe.py path/to/audio/file.mp3 --diarize
```

To specify the expected number of speakers:
```sh
python transcribe.py path/to/audio/file.mp3 --diarize --min-speakers 2 --max-speakers 4
```

## Platform-specific ML acceleration (Windows, Mac, Linux)

### Device selection
This project now automatically detects and uses the best available device for machine learning acceleration:
- **Windows/Linux**: Uses CUDA (NVIDIA GPU) if available, otherwise CPU.
- **macOS (Apple Silicon)**: Uses MPS/Metal acceleration if available, otherwise CPU.
- **CPU**: Used as fallback on all platforms.

No manual configuration is needed. The device and compute type are selected at runtime for optimal performance.

#### Apple Silicon (M1/M2/M3) Setup
- Install PyTorch with MPS support:
  ```bash
  pip install torch torchvision torchaudio
  ```
- See: https://pytorch.org/get-started/locally/ for latest instructions.
- The code will automatically use MPS if available.

#### Logging
The application logs which device is used for model inference.

#### Troubleshooting
- If you encounter errors on Mac, ensure you have the latest PyTorch and that your Python environment is ARM64 (not Intel/x86).
- For best performance on Mac, use Python 3.9+ and the latest pip.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Support the Project ⭐

If you find this project useful, please give it a star! Your support is appreciated and helps keep the project growing. 🌟


# 🚀 NVIDIA CUDA Installation Guide

This guide walks you through installing NVIDIA CUDA Toolkit 11.8, cuDNN, and TensorRT on Windows, including setting up Python packages like Cupy and TensorRT. It ensures proper system configuration for CUDA development, with steps for setting environment variables and verifying installation via cmd.exe

### 1. **Download the NVIDIA CUDA Toolkit 11.8**

First, download the CUDA Toolkit 11.8 from the official NVIDIA website:

👉 [Nvidia CUDA Toolkit 11.8 - DOWNLOAD HERE](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### 2. **Install the CUDA Toolkit**

- After downloading, open the installer (`.exe`) and follow the instructions provided by the installer.
- Make sure to select the following components during installation:
  - CUDA Toolkit
  - CUDA Samples
  - CUDA Documentation (optional)

### 3. **Verify the Installation**

- After the installation completes, open the `cmd.exe` terminal and run the following command to ensure that CUDA has been installed correctly:
  ```
  nvcc --version
  ```
This will display the installed CUDA version.

### **4. Install Cupy**
Run the following command in your terminal to install Cupy:
  ```
  pip install cupy-cuda11x
  ```

## 5. CUDNN Installation 🧩
Download cuDNN (CUDA Deep Neural Network library) from the NVIDIA website:

👉 [Download CUDNN](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.6/local_installers/11.x/cudnn-windows-x86_64-8.9.6.50_cuda11-archive.zip/). (Requires an NVIDIA account – it's free).

## 6. Unzip and Relocate 📁➡️
Open the `.zip` cuDNN file and move all the folders/files to the location where the CUDA Toolkit is installed on your machine, typically:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```


## 7. Get TensorRT 8.6 GA 🔽
Download [TensorRT 8.6 GA](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/zip/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip).

## 8. Unzip and Relocate 📁➡️
Open the `.zip` TensorRT file and move all the folders/files to the CUDA Toolkit folder, typically located at:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
```


## 9. Python TensorRT Installation 🎡
Once all the files are copied, run the following command to install TensorRT for Python:

```
pip install "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\python\tensorrt-8.6.1-cp311-none-win_amd64.whl"
```

🚨 **Note:** If this step doesn't work, double-check that the `.whl` file matches your Python version (e.g., `cp311` is for Python 3.11). Just locate the correct `.whl` file in the `python` folder and replace the path accordingly.

## 10. Set Your Environment Variables 🌎
Add the following paths to your environment variables:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
```

# Setting Up CUDA 11.8 with cuDNN on Windows

Once you have CUDA 11.8 installed and cuDNN properly configured, you need to set up your environment via `cmd.exe` to ensure that the system uses the correct version of CUDA (especially if multiple CUDA versions are installed).

## Steps to Set Up CUDA 11.8 Using `cmd.exe`

### 1. Set the CUDA Path in `cmd.exe`

You need to add the CUDA 11.8 binaries to the environment variables in the current `cmd.exe` session.

Open `cmd.exe` and run the following commands:

```
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp;%PATH%
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64;%PATH%
```
These commands add the CUDA 11.8 binary, lib, and CUPTI paths to your system's current session. Adjust the paths as necessary depending on your installation directory.

2. Verify the CUDA Version
After setting the paths, you can verify that your system is using CUDA 11.8 by running:
```
nvcc --version
```
This should display the details of CUDA 11.8. If it shows a different version, check the paths and ensure the proper version is set.

3. **Set the Environment Variables for a Persistent Session**
If you want to ensure CUDA 11.8 is used every time you open `cmd.exe`, you can add these paths to your system environment variables permanently:

1. Open `Control Panel` -> `System` -> `Advanced System Settings`.
Click on `Environment Variables`.
Under `System variables`, select `Path` and click `Edit`.
Add the following entries at the top of the list:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
```
This ensures that CUDA 11.8 is prioritized when running CUDA applications, even on systems with multiple CUDA versions.

4. **Set CUDA Environment Variables for cuDNN**
If you're using cuDNN, ensure the `cudnn64_8.dll` is also in your system path:
```
set PATH=C:\tools\cuda\bin;%PATH%
```
This should properly set up CUDA 11.8 to be used for your projects via `cmd.exe`.

### Environmental Variable Setup

![pic](https://github.com/KernFerm/v7yw9N8TL/blob/main/Environtmental_Setup/pic.png)

```
import torch

print(torch.cuda.is_available())  # This will return True if CUDA is available
print(torch.version.cuda)  # This will print the CUDA version being used
print(torch.cuda.get_device_name(0))  # This will print the name of the GPU, e.g., 'NVIDIA GeForce RTX GPU Model'
```
run the `get_device.py` to see if you installed it correctly 

## Cuda Requirements
- run the `cuda-requirements.bat` after you get done with installion of nvidia.

```
@echo off
:: Batch script to install Python packages for CUDA 11.8 environment

echo MAKE SURE TO HAVE THE WHL DOWNLOADED BEFORE YOU CONTINUE!!! 
pause
echo Click the link to download the WHL: press ctrl then left click with mouse
echo https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl
pause

echo Installing CuPy from WHL...
pip install https://github.com/cupy/cupy/releases/download/v12.0.0b1/cupy_cuda11x-12.0.0b1-cp311-cp311-win_amd64.whl
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing ONNX Runtime with GPU support...
pip install onnxruntime-gpu==1.19.2
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing NVIDIA PyIndex...
pip install nvidia-pyindex
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing cuDNN for CUDA 11.8...
pip install nvidia-cudnn-cu11==8.6.0.163
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing TensorRT for CUDA 11.8...
pip install nvidia-tensorrt==8.6.1
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing NumPy...
pip install numpy
echo Press enter to continue with the rest of the dependency installs
pause

echo Installing cupy-cuda11x...
pip install cupy-cuda11x
echo Press enter to continue with the rest of the dependency installs
pause

echo All packages installed successfully!
pause
```
