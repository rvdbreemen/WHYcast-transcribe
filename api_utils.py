"""
API utilities for interacting with OpenAI's API.
"""

import os
import logging
from typing import Optional, Dict, List
from tqdm import tqdm
from openai import OpenAI, BadRequestError

# Import configuration variables
try:
    from config import (
        OPENAI_MODEL, OPENAI_LARGE_CONTEXT_MODEL, 
        TEMPERATURE, MAX_TOKENS, MAX_INPUT_TOKENS, CHARS_PER_TOKEN,
        MAX_CHUNK_SIZE, CHUNK_OVERLAP
    )
except ImportError as e:
    raise ImportError(f"Error importing API configuration: {e}")

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

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    
    Args:
        text: The text to estimate
        
    Returns:
        Estimated number of tokens
    """
    return len(text) // CHARS_PER_TOKEN

def truncate_transcript(transcript: str, max_tokens: int) -> str:
    """
    Truncate a transcript to fit within token limits.
    
    Args:
        transcript: The transcript text
        max_tokens: Maximum token count allowed
        
    Returns:
        Truncated transcript
    """
    estimated_tokens = estimate_token_count(transcript)
    if estimated_tokens <= max_tokens:
        return transcript
        
    # If transcript is too long, keep the first part and last part
    chars_to_keep = max_tokens * CHARS_PER_TOKEN
    first_part_size = chars_to_keep // 2
    last_part_size = chars_to_keep - first_part_size - 100  # Leave room for ellipsis message
    
    first_part = transcript[:first_part_size]
    last_part = transcript[-last_part_size:]
    
    return first_part + "\n\n[...transcript truncated due to length...]\n\n" + last_part

def choose_appropriate_model(transcript: str) -> str:
    """
    Choose the appropriate model based on transcript length.
    
    Args:
        transcript: The transcript text
        
    Returns:
        Model name to use
    """
    estimated_tokens = estimate_token_count(transcript)
    logging.info(f"Estimated transcript length: ~{estimated_tokens} tokens")
    
    # If transcript is long, use large context model
    if estimated_tokens > MAX_INPUT_TOKENS and OPENAI_LARGE_CONTEXT_MODEL:
        logging.info(f"Using large context model: {OPENAI_LARGE_CONTEXT_MODEL} due to transcript length")
        return OPENAI_LARGE_CONTEXT_MODEL
    
    return OPENAI_MODEL

def split_into_chunks(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks of specified maximum size with overlap.
    
    Args:
        text: The text to split
        max_chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
        
    chunks = []
    start = 0
    
    while start < len(text):
        # Find a good breaking point (end of sentence or paragraph)
        end = min(start + max_chunk_size, len(text))
        
        # Try to find paragraph break
        paragraph_break = text.rfind('\n\n', start, end)
        if paragraph_break != -1 and paragraph_break > start + max_chunk_size // 2:
            end = paragraph_break + 2
        else:
            # Try to find sentence break (period followed by space)
            sentence_break = text.rfind('. ', start, end)
            if sentence_break != -1 and sentence_break > start + max_chunk_size // 2:
                end = sentence_break + 2
        
        chunks.append(text[start:end])
        # Start the next chunk with some overlap for context
        start = max(start + 1, end - overlap)
        
    return chunks

def summarize_large_transcript(transcript: str, prompt: str) -> Optional[str]:
    """
    Handle very large transcripts by chunking and recursive summarization.
    
    Args:
        transcript: The transcript text
        prompt: The summarization prompt
        
    Returns:
        Generated summary or None if failed
    """
    estimated_tokens = estimate_token_count(transcript)
    logging.info(f"Starting recursive summarization for large transcript (~{estimated_tokens} tokens)")
    
    # Split into chunks
    chunks = split_into_chunks(transcript)
    logging.info(f"Split transcript into {len(chunks)} chunks")
    
    # Process each chunk using process_with_openai
    intermediate_summaries = []
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        try:
            chunk_prompt = "This is part of a longer transcript. Please summarize just this section, focusing on key points."
            summary = process_with_openai(chunk, chunk_prompt, OPENAI_MODEL, max_tokens=MAX_TOKENS // 2)
            
            if summary:
                intermediate_summaries.append(summary)
                logging.info(f"Completed summary for chunk {i+1}")
            else:
                logging.warning(f"Failed to summarize chunk {i+1}")
        except Exception as e:
            logging.error(f"Error summarizing chunk {i+1}: {str(e)}")
            # Continue with partial results if available
    
    # If we have intermediate summaries, combine them
    if not intermediate_summaries:
        return None
    
    # Combine intermediate summaries into a final summary
    combined_text = "\n\n".join(intermediate_summaries)
    
    # Use process_with_openai for final combination
    final_prompt = f"{prompt}\n\nHere are summaries from different parts of the transcript. Please combine them into a cohesive summary and blog post:"
    return process_with_openai(combined_text, final_prompt, OPENAI_LARGE_CONTEXT_MODEL, max_tokens=MAX_TOKENS)

def process_with_openai(text: str, prompt: str, model_name: str, max_tokens: int = MAX_TOKENS) -> Optional[str]:
    """
    Process text with OpenAI model using the specified prompt.
    
    Args:
        text: The text to process
        prompt: Instructions for the AI
        model_name: Name of the model to use
        max_tokens: Maximum tokens for output
        
    Returns:
        The generated text or None if there was an error
    """
    try:
        api_key = ensure_api_key()
        client = OpenAI(api_key=api_key)
        
        estimated_tokens = estimate_token_count(text)
        logging.info(f"Processing text with OpenAI (~{estimated_tokens} tokens)")
        
        # If text is too long, truncate it
        if estimated_tokens > MAX_INPUT_TOKENS:
            logging.warning(f"Text too long (~{estimated_tokens} tokens > {MAX_INPUT_TOKENS} limit), truncating...")
            text = truncate_transcript(text, MAX_INPUT_TOKENS)
        
        try:
            # For transcript cleanup, we need to ensure we get complete output by using a higher max_tokens limit
            if "clean" in prompt.lower() and "transcript" in prompt.lower():
                # For cleanup, give more tokens for output - use at least text length or double MAX_TOKENS
                cleanup_max_tokens = max(estimate_token_count(text), MAX_TOKENS * 2)
                logging.info(f"Using expanded token limit for cleanup: {cleanup_max_tokens}")
                
                # Check if text needs to be processed in chunks due to size
                if estimated_tokens > MAX_INPUT_TOKENS // 2:
                    return process_large_text_in_chunks(text, prompt, model_name, client)
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
                        {"role": "user", "content": f"{prompt}\n\nHere's the text to process:\n\n{text}"}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=cleanup_max_tokens
                )
            else:
                # Regular processing for non-cleanup tasks
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
                        {"role": "user", "content": f"{prompt}\n\nHere's the text to process:\n\n{text}"}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=max_tokens
                )
                
            result = response.choices[0].message.content
            
            # Check if result might be truncated (ends abruptly without proper punctuation)
            if len(result) > 100 and not result.rstrip().endswith(('.', '!', '?', '"', ':', ';', ')', ']', '}')):
                logging.warning("Generated text may be truncated (doesn't end with punctuation)")
                
            return result
                
        except BadRequestError as e:
            if "maximum context length" in str(e).lower():
                # Try with more aggressive truncation
                logging.warning(f"Context length exceeded. Retrying with further truncation...")
                text = truncate_transcript(text, MAX_INPUT_TOKENS // 2)
                logging.info(f"Retrying with reduced text (~{estimate_token_count(text)} tokens)...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
                        {"role": "user", "content": f"{prompt}\n\nHere's the text to process:\n\n{text}"}
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:
                raise
    except Exception as e:
        logging.error(f"Error processing with OpenAI: {str(e)}")
        return None

def process_large_text_in_chunks(text: str, prompt: str, model_name: str, client: OpenAI) -> str:
    """
    Process very large text by breaking it into chunks and reassembling the results.
    
    Args:
        text: The text to process
        prompt: The processing instructions
        model_name: Model to use
        client: OpenAI client
        
    Returns:
        Combined processed text
    """
    logging.info("Text is very large, processing in chunks")
    chunks = split_into_chunks(text, max_chunk_size=MAX_CHUNK_SIZE*2)
    logging.info(f"Split text into {len(chunks)} chunks")
    
    modified_prompt = f"{prompt}\n\nThis is a chunk of a longer transcript. Process this chunk following the instructions."
    processed_chunks = []
    
    # Add progress bar for chunk processing
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        logging.info(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that processes transcript chunks."},
                    {"role": "user", "content": f"{modified_prompt}\n\nChunk {i+1}/{len(chunks)}:\n\n{chunk}"}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS * 2  # Use larger output limit for chunks
            )
            processed_chunks.append(response.choices[0].message.content)
            logging.info(f"Successfully processed chunk {i+1}")
        except Exception as e:
            logging.error(f"Error processing chunk {i+1}: {str(e)}")
            # If processing fails, include original chunk to avoid data loss
            processed_chunks.append(chunk)
    
    # Combine processed chunks
    combined_text = "\n\n".join(processed_chunks)
    
    # Optionally, run a final pass to ensure consistency across chunk boundaries
    if len(processed_chunks) > 1:
        try:
            logging.info("Running final pass to ensure consistency across chunk boundaries")
            # Estimate token count of combined text
            combined_tokens = estimate_token_count(combined_text)
            
            # If combined text is still too large, just return it as is
            if combined_tokens > MAX_INPUT_TOKENS:
                logging.warning(f"Combined text is too large for final pass (~{combined_tokens} tokens)")
                return combined_text
                
            consistency_prompt = "This is a processed transcript that was handled in chunks. Please ensure consistency across chunk boundaries and fix any obvious issues."
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that ensures transcript consistency."},
                    {"role": "user", "content": f"{consistency_prompt}\n\n{combined_text}"}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in final consistency pass: {str(e)}")
            return combined_text
    
    return combined_text