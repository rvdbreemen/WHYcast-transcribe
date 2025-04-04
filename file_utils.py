"""
File utilities for handling file operations.
"""

import os
import logging
from typing import Tuple, Dict, Optional
import re

# Import from format_utils
from format_utils import (
    convert_markdown_to_html,
    convert_markdown_to_wiki,
    convert_existing_blogs
)

# Import configuration variables
try:
    from config import (
        MAX_FILE_SIZE_KB
    )
except ImportError as e:
    raise ImportError(f"Error importing file configuration: {e}")

def read_prompt_file(prompt_file: str) -> Optional[str]:
    """
    Read a prompt from file.
    
    Args:
        prompt_file: Path to the prompt file
        
    Returns:
        The prompt text or None if there was an error
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading prompt file {prompt_file}: {str(e)}")
        return None

def check_file_size(file_path: str) -> bool:
    """
    Check if the file size is within acceptable limits.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if file size is acceptable, False if it's too large
    """
    size_kb = os.path.getsize(file_path) / 1024
    if size_kb > MAX_FILE_SIZE_KB:
        logging.warning(f"File size ({size_kb:.1f} KB) exceeds recommended limit ({MAX_FILE_SIZE_KB} KB). " 
                      f"Processing might be slow or fail.")
        return False
    return True

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
        f"{base}_summary.txt",     # For summary
    )

def write_transcript_files(segments, output_file: str, output_file_timestamped: str, apply_vocabulary_corrections=None) -> str:
    """
    Write transcript files and return the full transcript.
    
    Args:
        segments: Transcript segments from WhisperModel
        output_file: Path for plain text output
        output_file_timestamped: Path for timestamped output
        apply_vocabulary_corrections: Function to apply vocabulary corrections (if None, no corrections)
        
    Returns:
        The full transcript as a string
    """
    full_transcript = ""
    
    with open(output_file, "w", encoding="utf-8") as f_plain, open(output_file_timestamped, "w", encoding="utf-8") as f_timestamped:
        
        for segment in segments:
            # Get the original segment text
            segment_text = segment.text
            
            # Apply vocabulary corrections to segment text if enabled
            if apply_vocabulary_corrections:
                segment_text = apply_vocabulary_corrections(segment_text)
            
            # Store for OpenAI processing
            full_transcript += segment_text + "\n"
            
            # Write to plain text file without timestamps
            f_plain.write(segment_text + "\n")
            
            # Write to timestamped file with timestamps
            f_timestamped.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment_text))
            
            # Print to console with timestamps - handle potential encoding errors
            try:
                logging.info("[%.2fs -> %.2fs] %s", segment.start, segment.end, segment_text)
            except UnicodeEncodeError:
                # Fall back to ASCII if Unicode fails
                safe_text = segment_text.encode('ascii', 'replace').decode('ascii')
                logging.info("[%.2fs -> %.2fs] %s", segment.start, segment.end, safe_text)
    
    logging.info(f"Transcription saved to: {output_file}")
    logging.info(f"Timestamped transcription saved to: {output_file_timestamped}")
    
    # Apply a final pass of vocabulary corrections to the full transcript if enabled
    # This catches any patterns that might span across segments
    if apply_vocabulary_corrections:
        original_transcript = full_transcript
        full_transcript = apply_vocabulary_corrections(full_transcript)
        
        # If the final pass made additional corrections, update the saved files
        if full_transcript != original_transcript:
            logging.info("Applying final vocabulary correction pass to catch cross-segment patterns")
            with open(output_file, "w", encoding="utf-8") as f_plain:
                f_plain.write(full_transcript)
            
            # Simplification: regenerate timestamped file from full transcript
            # This is imperfect since timestamps might not perfectly align with corrected text
            lines = full_transcript.split('\n')
            timestamped_lines = []
            with open(output_file_timestamped, "r", encoding="utf-8") as f_original:
                original_timestamped = f_original.readlines()
                
            # Try to maintain timestamp information while updating text
            for i, line in enumerate(lines):
                if i < len(original_timestamped):
                    ts_line = original_timestamped[i]
                    ts_match = re.match(r'^\[\s*(\d+\.\d+)s\s*->\s*(\d+\.\d+)s\s*\]', ts_line)
                    if ts_match and line.strip():
                        start, end = ts_match.groups()
                        timestamped_lines.append(f"[{start}s -> {end}s] {line}")
                    elif line.strip():
                        timestamped_lines.append(line)
            
            with open(output_file_timestamped, "w", encoding="utf-8") as f_updated:
                f_updated.write('\n'.join(timestamped_lines))
    
    return full_transcript

def write_workflow_outputs(results: Dict[str, str], output_base: str) -> None:
    """
    Write the outputs from the workflow to files.
    
    Args:
        results: Dictionary with cleaned_transcript, summary, and blog
        output_base: Base path for output files
    """
    # Write cleaned transcript
    if 'cleaned_transcript' in results and results['cleaned_transcript']:
        cleaned_path = f"{output_base}_cleaned.txt"
        with open(cleaned_path, "w", encoding="utf-8") as f:
            f.write(results['cleaned_transcript'])
        logging.info(f"Cleaned transcript saved to: {cleaned_path}")
    
    # Write summary
    if 'summary' in results and results['summary']:
        summary_path = f"{output_base}_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(results['summary'])
        logging.info(f"Summary saved to: {summary_path}")
    
    # Write blog post
    if 'blog' in results and results['blog']:
        # Get base filename without any special suffixes
        base_filename = os.path.basename(output_base)
        output_dir = os.path.dirname(output_base)
        blog_path = os.path.join(output_dir, f"{base_filename}_blog.txt")
        
        with open(blog_path, "w", encoding="utf-8") as f:
            f.write(results['blog'])
        logging.info(f"Blog post saved to: {blog_path}")
        
        # Generate and write HTML version
        html_content = convert_markdown_to_html(results['blog'])
        html_path = os.path.join(output_dir, f"{base_filename}_blog.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML blog post saved to: {html_path}")
        
        # Generate and write Wiki version
        wiki_content = convert_markdown_to_wiki(results['blog'])
        wiki_path = os.path.join(output_dir, f"{base_filename}_blog.wiki")
        with open(wiki_path, "w", encoding="utf-8") as f:
            f.write(wiki_content)
        logging.info(f"Wiki blog post saved to: {wiki_path}")
        
    # Write blog post alt1
    if 'blog_alt1' in results and results['blog_alt1']:
        # Get base filename without any special suffixes
        base_filename = os.path.basename(output_base)
        output_dir = os.path.dirname(output_base)
        blog_alt1_path = os.path.join(output_dir, f"{base_filename}_blog_alt1.txt")
        
        with open(blog_alt1_path, "w", encoding="utf-8") as f:
            f.write(results['blog_alt1'])
        logging.info(f"Blog alt1 post saved to: {blog_alt1_path}")
        
        # Generate and write HTML version
        html_content = convert_markdown_to_html(results['blog_alt1'])
        html_path = os.path.join(output_dir, f"{base_filename}_blog_alt1.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"HTML blog alt 1post saved to: {html_path}")
        
        # Generate and write Wiki version
        wiki_content = convert_markdown_to_wiki(results['blog_alt1'])
        wiki_path = os.path.join(output_dir, f"{base_filename}_blog_alt1.wiki")
        with open(wiki_path, "w", encoding="utf-8") as f:
            f.write(wiki_content)
        logging.info(f"Wiki blog alt1 post saved to: {wiki_path}")

def transcription_exists(input_file: str) -> bool:
    """
    Check if a transcription already exists for the given audio file.
    
    Args:
        input_file: Path to the input audio file    
        
    Returns:
        True if transcription exists, False otherwise
    """
    base = os.path.splitext(input_file)[0]
    transcript_file = f"{base}.txt"
    return os.path.exists(transcript_file)