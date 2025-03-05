# WHYcast Transcribe

A powerful tool for transcribing audio files and automatically generating summaries and blog posts using OpenAI's GPT models.

> **Note**: This code was generated using Claude 3.7 Sonnet Thinking with GitHub Co-Pilot assistance. Last updated: March, 6 of 2025.
>
> **Current version**: 0.0.1

## Features

- Transcribe audio files using the Faster Whisper implementation
- Generate timestamps for each segment of speech
- Automatically create summaries and blog posts from transcripts
- Handle files of any size through recursive summarization
- Batch processing for multiple files
- Configurable via environment variables or .env file

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/YourUsername/WHYcast-transcribe.git
   cd WHYcast-transcribe
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

## Usage

### Basic usage

```bash
python transcribe.py path/to/audiofile.mp3
```

This will:
1. Transcribe the audio file
2. Save plain text transcript as `audiofile.txt`
3. Save timestamped transcript as `audiofile_ts.txt`
4. Generate and save summary as `audiofile_summary.txt`

### Command-line options

```bash
python transcribe.py path/to/audiofile.mp3 [options]
```

Options:
- `--batch`, `-b`: Process multiple files matching a glob pattern
- `--model`, `-m`: Specify Whisper model size (e.g., "large-v3", "medium", "small")
- `--output-dir`, `-o`: Directory to save output files
- `--skip-summary`, `-s`: Skip summary generation
- `--verbose`, `-v`: Enable verbose logging
- `--version`: Show program's version number and exit

### Batch processing

```bash
python transcribe.py "path/to/*.mp3" --batch
```

## Configuration

The application can be configured using environment variables or a `.env` file.

### Environment variables

#### Whisper Configuration
- `WHISPER_MODEL_SIZE`: Model size to use (default: "large-v3")
- `WHISPER_DEVICE`: Device to run inference on (default: "cuda")
- `WHISPER_COMPUTE_TYPE`: Compute type for model (default: "int8")
- `WHISPER_BEAM_SIZE`: Beam size for transcription (default: 5)

#### OpenAI Configuration
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use for summaries (default: "gpt-4o")
- `OPENAI_LARGE_CONTEXT_MODEL`: Model for large transcripts (default: "gpt-4o-2024-05-13")
- `OPENAI_TEMPERATURE`: Temperature setting (default: 0.7)
- `OPENAI_MAX_TOKENS`: Maximum tokens in completion (default: 4000)
- `OPENAI_MAX_INPUT_TOKENS`: Maximum input tokens (default: 50000)

#### Advanced Configuration
- `USE_RECURSIVE_SUMMARIZATION`: Enable recursive summarization (default: True)
- `MAX_CHUNK_SIZE`: Maximum chunk size for recursive summarization (default: 40000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 1000)
- `MAX_FILE_SIZE_KB`: Maximum file size before warning (default: 250)

## File Structure

- `transcribe.py`: Main script for transcription and summarization
- `config.py`: Configuration settings
- `requirements.txt`: Required Python packages
- `summary_prompt_blog.txt`: Prompt template for generating summaries

## Example

1. Transcribe a single file:
   ```bash
   python transcribe.py podcast_episode.mp3
   ```

2. Batch process all mp3 files in a directory:
   ```bash
   python transcribe.py "podcasts/*.mp3" --batch --output-dir summaries
   ```

3. Use a smaller model for faster processing:
   ```bash
   python transcribe.py interview.mp3 --model medium
   ```

## Requirements

- Python 3.8+
- OpenAI API key
- For CUDA support (recommended): NVIDIA GPU with CUDA installed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This tool uses [Faster Whisper](https://github.com/guillaumekln/faster-whisper) for transcription
- Summaries are generated using OpenAI's GPT models
