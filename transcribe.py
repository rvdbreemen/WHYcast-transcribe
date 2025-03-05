import os
import sys
import logging
from typing import Tuple, List, Optional

from faster_whisper import WhisperModel
from openai import OpenAI

# ==================== CONFIGURATION ====================
# Model configuration
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5

# OpenAI configuration
OPENAI_MODEL = "gpt-4"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

# File paths
PROMPT_FILE = "summary_prompt_blog.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==================== API FUNCTIONS ====================
def ensure_api_key() -> str:
    """
    Ensure that the OpenAI API key is available.
    
    Returns:
        str: The API key if available
        
    Raises:
        ValueError: If the API key is not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return api_key

def read_summary_prompt(prompt_file: str = PROMPT_FILE) -> Optional[str]:
    """
    Read the summary prompt from file.
    
    Args:
        prompt_file: Path to the prompt file
        
    Returns:
        The prompt text or None if there was an error
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading prompt file: {str(e)}")
        return None

def generate_summary_and_blog(transcript: str, prompt: str) -> Optional[str]:
    """
    Generate summary and blog post using OpenAI API.
    
    Args:
        transcript: The transcript text to summarize
        prompt: Instructions for the AI
        
    Returns:
        The generated summary and blog or None if there was an error
    """
    try:
        api_key = ensure_api_key()
        client = OpenAI(api_key=api_key)
        
        logging.info("Sending transcript to OpenAI for summary and blog generation...")
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes transcripts."},
                {"role": "user", "content": f"{prompt}\n\nHere's the transcript to summarize:\n\n{transcript}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating summary: {str(e)}")
        return None

# ==================== TRANSCRIPTION FUNCTIONS ====================
def setup_model() -> WhisperModel:
    """
    Initialize and return the Whisper model.
    
    Returns:
        The initialized WhisperModel
    """
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

def transcribe_audio(model: WhisperModel, audio_file: str) -> Tuple[List, object]:
    """
    Transcribe an audio file using the provided model.
    
    Args:
        model: The WhisperModel to use
        audio_file: Path to the audio file
        
    Returns:
        Tuple of (segments, info)
    """
    logging.info(f"Transcribing {audio_file}...")
    return model.transcribe(audio_file, beam_size=BEAM_SIZE)

def create_output_paths(input_file: str) -> Tuple[str, str, str]:
    """
    Create output file paths based on the input filename.
    
    Args:
        input_file: Path to the input audio file
        
    Returns:
        Tuple of (plain_text_path, timestamped_path, summary_path)
    """
    base = os.path.splitext(input_file)[0]
    return (
        f"{base}.txt",             # Without timestamps
        f"{base}_ts.txt",          # With timestamps
        f"{base}_summary.txt"      # For summary and blog
    )

def write_transcript_files(segments, output_file: str, output_file_timestamped: str) -> str:
    """
    Write transcript files and return the full transcript.
    
    Args:
        segments: Transcript segments from WhisperModel
        output_file: Path for plain text output
        output_file_timestamped: Path for timestamped output
        
    Returns:
        The full transcript as a string
    """
    full_transcript = ""
    
    with open(output_file, "w", encoding="utf-8") as f_plain, \
         open(output_file_timestamped, "w", encoding="utf-8") as f_timestamped:
        
        for segment in segments:
            # Print to console with timestamps
            logging.info("[%.2fs -> %.2fs] %s", segment.start, segment.end, segment.text)
            
            # Store for OpenAI processing
            full_transcript += segment.text + "\n"
            
            # Write to plain text file without timestamps
            f_plain.write(segment.text + "\n")
            
            # Write to timestamped file with timestamps
            f_timestamped.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
    
    logging.info(f"Transcription saved to: {output_file}")
    logging.info(f"Timestamped transcription saved to: {output_file_timestamped}")
    
    return full_transcript

def process_summary(full_transcript: str, output_summary_file: str) -> bool:
    """
    Process the transcript to generate and save a summary.
    
    Args:
        full_transcript: The complete transcript text
        output_summary_file: Path to save the summary
        
    Returns:
        Success status (True/False)
    """
    summary_prompt = read_summary_prompt()
    if not summary_prompt:
        logging.warning("Summary prompt file not found or empty, skipping summary generation")
        return False
        
    logging.info("Generating summary and blog post...")
    summary_and_blog = generate_summary_and_blog(full_transcript, summary_prompt)
    
    if not summary_and_blog:
        logging.error("Failed to generate summary and blog post")
        return False
        
    # Save summary and blog to file
    with open(output_summary_file, "w", encoding="utf-8") as f_summary:
        f_summary.write(summary_and_blog)
    logging.info(f"Summary and blog post saved to: {output_summary_file}")
    return True

# ==================== MAIN FUNCTION ====================
def main(input_file: str) -> None:
    """
    Main function to process an audio file.
    
    Args:
        input_file: Path to the input audio file
    """
    try:
        # Setup
        model = setup_model()
        output_file, output_file_timestamped, output_summary_file = create_output_paths(input_file)
        
        # Transcribe
        segments, info = transcribe_audio(model, input_file)
        logging.info(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        # Write transcript files
        full_transcript = write_transcript_files(segments, output_file, output_file_timestamped)
        
        # Generate and save summary
        process_summary(full_transcript, output_summary_file)
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("Usage: python transcribe.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        logging.error(f"Input file does not exist: {input_file}")
        sys.exit(1)

    main(input_file)
