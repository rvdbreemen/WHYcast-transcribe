# WHYcast-transcribe

A tool for transcribing audio files and generating summaries using OpenAI GPT models.

## Features

- **Audio Transcription**: Transcribes audio files using faster-whisper models
- **Summary Generation**: Creates summaries of transcripts using OpenAI GPT models
- **Blog Post Generation**: Automatically generates blog posts from transcriptions
- **Batch Processing**: Process multiple files at once
- **Flexible Output**: Generate plain text and timestamped transcripts
- **Recursive Summarization**: Handles very large transcripts by chunking
- **Summary Regeneration**: Regenerate summaries from existing transcripts

## Installation

1. Clone this repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

```bash
python transcribe.py path/to/audio_file.mp3
```

This will:
1. Transcribe the audio file
2. Generate a plain text transcript (.txt)
3. Generate a timestamped transcript (_ts.txt)
4. Generate a summary (_summary.txt)
5. Generate a blog post (_blog.txt)

### Command Line Options

```
usage: transcribe.py [-h] [--batch] [--all-mp3s] [--model MODEL] [--output-dir OUTPUT_DIR] [--skip-summary] [--force] [--verbose] [--version] [--regenerate-summary] [--regenerate-all-summaries] input

WHYcast Transcribe - Transcribe audio files and generate summaries

positional arguments:
  input                 Path to the input audio file, directory, or glob pattern

optional arguments:
  -h, --help            show this help message and exit
  --batch, -b           Process multiple files matching pattern
  --all-mp3s, -a        Process all MP3 files in directory
  --model MODEL, -m MODEL
                        Model size (e.g., "large-v3", "medium", "small")
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to save output files
  --skip-summary, -s    Skip summary generation
  --force, -f           Force regeneration of transcriptions even if they exist
  --verbose, -v         Enable verbose logging
  --version             show program's version number and exit
  --regenerate-summary, -r
                        Regenerate summary and blog from existing transcript
  --regenerate-all-summaries, -R
                        Regenerate summaries for all transcripts in directory
```

### Examples

**Process a single file:**
```bash
python transcribe.py podcast.mp3
```

**Process all MP3s in a directory:**
```bash
python transcribe.py directory_path --all-mp3s
```

**Process multiple files matching a pattern:**
```bash
python transcribe.py "*.mp3" --batch
```

**Use a specific model size:**
```bash
python transcribe.py podcast.mp3 --model large-v3
```

**Skip summary generation:**
```bash
python transcribe.py podcast.mp3 --skip-summary
```

**Regenerate summary from an existing transcript:**
```bash
python transcribe.py podcast.txt --regenerate-summary
```

**Regenerate all summaries in a directory:**
```bash
python transcribe.py directory_path --regenerate-all-summaries
```

### Batch Processing

When processing multiple files with `--batch` or `--all-mp3s`, the script processes one file at a time and continues to the next file even if errors occur with the current file.

**Tips for handling large batches:**

1. **Check the log file**: All processing events are logged to `transcribe.log` in the script directory
   ```bash
   # After a batch run, check for errors
   grep -i error transcribe.log
   ```

2. **Use the verbose flag**: Enable detailed logging to track progress
   ```bash
   python transcribe.py directory_path --all-mp3s --verbose
   ```

3. **Resume interrupted processing**: If a batch was interrupted, use `--force` only if needed
   ```bash
   # This will skip files that were already transcribed
   python transcribe.py directory_path --all-mp3s
   ```

4. **Process in smaller batches**: For very large collections, process directories separately
   ```bash
   python transcribe.py "podcasts/2023/*.mp3" --batch
   python transcribe.py "podcasts/2022/*.mp3" --batch
   ```

5. **Identify failed files**: After processing, any files without corresponding .txt output likely failed

6. **Run another pass for failed files**: To process only files that failed in previous runs
   ```bash
   # This will only process files that don't already have transcriptions
   python transcribe.py directory_path --all-mp3s
   ```

7. **Monitor with progress bar**: The script displays a progress bar showing completion status

If you need to retry only failed files, you can create a list of successfully processed files and use pattern matching to process the remaining ones.

## Configuration

Edit the `config.py` file to customize:

- Whisper model size and parameters
- OpenAI model selection
- Token limits and temperature settings
- Summarization parameters

## Requirements

- Python 3.7+
- faster-whisper
- openai
- tqdm
- python-dotenv

## License

MIT License - See LICENSE file for details

## Author

Robert van den Breemen
