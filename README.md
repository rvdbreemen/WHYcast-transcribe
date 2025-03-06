# WHYcast Transcribe

A tool for transcribing audio files and generating summaries using OpenAI GPT models.

## Features

- Transcribe audio files using Whisper models
- Generate summaries and blog posts from transcriptions
- Apply custom vocabulary corrections to improve transcription accuracy
- Handle very large transcripts with recursive summarization
- Process files in batch mode
- Regenerate summaries without re-transcribing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/WHYcast-transcribe.git
   cd WHYcast-transcribe
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Configuration

Create a `config.py` file with the following settings:

```python
# Model configuration
VERSION = "0.0.3"
MODEL_SIZE = "medium"  # Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "float32"
BEAM_SIZE = 5

# OpenAI API configuration
OPENAI_MODEL = "gpt-4"
OPENAI_LARGE_CONTEXT_MODEL = "gpt-4-32k"
TEMPERATURE = 0.3
MAX_TOKENS = 4000
MAX_INPUT_TOKENS = 16000
CHARS_PER_TOKEN = 4

# File configuration
PROMPT_FILE = "prompt.txt"
MAX_FILE_SIZE_KB = 1000

# Summarization options
USE_RECURSIVE_SUMMARIZATION = True
MAX_CHUNK_SIZE = 16000
CHUNK_OVERLAP = 1000

# Vocabulary correction
VOCABULARY_FILE = "vocabulary.json"
USE_CUSTOM_VOCABULARY = True
```

## Usage

### Basic Transcription

Transcribe a single audio file:

```bash
python transcribe.py path/to/audio.mp3
```

### Specifying a Model Size

Choose a different Whisper model size:

```bash
python transcribe.py path/to/audio.mp3 --model large-v3
```

Available model sizes: tiny, base, small, medium, large-v1, large-v2, large-v3

### Skip Summary Generation

Only generate transcription (no summary or blog post):

```bash
python transcribe.py path/to/audio.mp3 --skip-summary
```

### Batch Processing

Process all files matching a pattern:

```bash
python transcribe.py "path/to/*.mp3" --batch
```

Process all MP3 files in a directory:

```bash
python transcribe.py path/to/directory --all-mp3s
```

### Force Regeneration

Force regeneration of transcriptions even if they exist:

```bash
python transcribe.py path/to/audio.mp3 --force
```

### Summary Regeneration

Regenerate the summary for an existing transcript:

```bash
python transcribe.py path/to/transcript.txt --regenerate-summary
```

Regenerate summaries for all transcripts in a directory:

```bash
python transcribe.py path/to/directory --regenerate-all-summaries
```

### Custom Vocabulary

To use custom vocabulary corrections, create a `vocabulary.json` file with word mappings:

```json
{
  "incorrectTerm": "correctTerm",
  "misspelled": "correctly spelled",
  "acronym": "expanded form"
}
```

Set `USE_CUSTOM_VOCABULARY=True` in your `config.py` to enable this feature.

To skip vocabulary corrections for a specific run:

```bash
python transcribe.py path/to/audio.mp3 --skip-vocabulary
```

### Output Directory

Specify a custom output directory:

```bash
python transcribe.py path/to/audio.mp3 --output-dir path/to/output
```

### Verbose Logging

Enable detailed logging:

```bash
python transcribe.py path/to/audio.mp3 --verbose
```

## Prompt Template

Create a `prompt.txt` file with instructions for the summary generation:

```
You're an assistant that helps create concise summaries and blog posts from audio transcripts.

Please analyze the provided transcript and:

1. Create a concise summary (max 250 words) highlighting the key points
2. Create a well-structured blog post (500-800 words) based on the content
3. Include appropriate headings and subheadings in the blog post
4. Maintain the original meaning and key insights from the transcript

Format your response as:

## Summary
[Your summary here]

## Blog Post
[Your blog post with headings here]
```

## License

MIT License (see LICENSE file for details)

## Acknowledgements

- Uses [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) for transcription
- Uses [OpenAI API](https://openai.com/) for summary generation
