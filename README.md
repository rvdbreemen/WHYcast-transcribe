# Audio Transcription with AI Summary

This tool transcribes audio files using Faster Whisper and then generates a summary and blog post using OpenAI's API.

## Features

- Transcribes audio files with high accuracy using Faster Whisper (large-v3 model)
- Produces both plain and timestamped transcript files
- Generates an AI summary and blog post using OpenAI's API
- Customizable summary prompts

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (for optimal performance)

## Installation

1. Clone or download this repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install required packages:

```bash
pip install faster-whisper openai
```

## Configuration

### OpenAI API Key

Set your OpenAI API key as an environment variable:

- **Windows:**
  ```
  set OPENAI_API_KEY=your-api-key-here
  ```

- **Linux/macOS:**
  ```
  export OPENAI_API_KEY=your-api-key-here
  ```

### Summary Prompt

The file `summary_prompt_blog.txt` contains the instructions for OpenAI to generate the summary and blog post. You can modify this file to customize the output format and content.

## Usage

Run the script with your audio file as the argument:

```bash
python transcribe.py <path-to-audio-file>
```

Example:
```bash
python transcribe.py c:\recordings\podcast.mp3
```

## Output Files

For an input file named `podcast.mp3`, the script will generate:

- `podcast.txt`: Transcript without timestamps
- `podcast_ts.txt`: Transcript with timestamps
- `podcast_summary.txt`: AI-generated summary and blog post

## Troubleshooting

- **CUDA errors**: Make sure you have the correct CUDA toolkit installed for your GPU
- **API errors**: Verify your OpenAI API key is set correctly and has sufficient credits
- **Memory issues**: For long audio files, consider processing in smaller chunks
