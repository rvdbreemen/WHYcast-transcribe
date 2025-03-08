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

- Python 3.7+
- Required Python packages (listed in `requirements.txt`)

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
