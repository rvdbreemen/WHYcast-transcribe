# WHYcast Transcribe

WHYcast Transcribe is a tool for transcribing audio files and generating summaries using OpenAI GPT models. It supports downloading the latest episode from podcast feeds and provides various options for processing audio files.

## Features

- Transcribe audio files using the Whisper model
- Generate summaries and blog posts from transcripts using OpenAI GPT models
- Download the latest episode from a podcast RSS feed
- Apply custom vocabulary corrections to transcripts
- Batch processing of multiple audio files
- Regenerate summaries from existing transcripts

## Requirements
- CUDA enabled GPU (e.g. NVIDIA GeForce RTX 3060)
- Python 3.7+
- Required Python packages (listed in `requirements.txt`)


## NVIDIA CUDA Installation Guide

A good guide to enable CUDA and cuDNN for TensorFlow on Windows 11: 
https://medium.com/@gokulprasath100702/a-guide-to-enabling-cuda-and-cudnn-for-tensorflow-on-windows-11-a89ce11863f1


## Installation 

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/WHYcast-transcribe.git
    cd WHYcast-transcribe
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Create a `.env` file with your OpenAI API key:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

### Transcribe an Audio File

To transcribe an audio file and generate a summary:
```sh
python transcribe.py path/to/audio/file.mp3
```

### Download and Transcribe the Latest Podcast Episode

To download the latest episode from the default podcast feed and transcribe it:
```sh
python transcribe.py
```

### Batch Processing

To process multiple files matching a pattern:
```sh
python transcribe.py --batch "path/to/files/*.mp3"
```

### Process All MP3 Files in a Directory

To process all MP3 files in a directory:
```sh
python transcribe.py --all-mp3s path/to/directory
```

### Regenerate Summary from Existing Transcript

To regenerate the summary and blog post from an existing transcript file:
```sh
python transcribe.py --regenerate-summary path/to/transcript.txt
```

### Regenerate Summaries for All Transcripts in a Directory

To regenerate summaries for all transcript files in a directory:
```sh
python transcribe.py --regenerate-all-summaries path/to/directory
```

## Command-Line Options

- `--batch, -b`: Process multiple files matching pattern
- `--all-mp3s, -a`: Process all MP3 files in directory
- `--model, -m`: Model size (e.g., "large-v3", "medium", "small")
- `--output-dir, -o`: Directory to save output files
- `--skip-summary, -s`: Skip summary generation
- `--force, -f`: Force regeneration of transcriptions even if they exist
- `--verbose, -v`: Enable verbose logging
- `--version`: Show the version of WHYcast Transcribe
- `--regenerate-summary, -r`: Regenerate summary and blog from existing transcript
- `--regenerate-all-summaries, -R`: Regenerate summaries for all transcripts in directory
- `--skip-vocabulary`: Skip custom vocabulary corrections
- `--rss-feed, -rss`: URL of the RSS feed to process
- `--no-rss`: Skip checking the RSS feed
- `--limit, -l`: Limit the number of podcasts to process from RSS feed (default: RSS_DEFAULT_LIMIT)
- `--clear-cache`: Clear model and vocabulary cache before processing

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

🚨 **Note:** If this step doesn’t work, double-check that the `.whl` file matches your Python version (e.g., `cp311` is for Python 3.11). Just locate the correct `.whl` file in the `python` folder and replace the path accordingly.

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
