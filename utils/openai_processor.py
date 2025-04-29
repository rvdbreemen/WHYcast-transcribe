#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI API utilities for WHYcast-transcribe.

This module provides functions for interacting with the OpenAI API, including:
- Model selection based on transcript length
- Robust text processing with chunking and error handling
- Utilities for handling large transcripts and token limits

Intended for use in all modules that require LLM-based summarization, cleanup, or content generation.
"""

# openai_processor.py: OpenAI API utilities for WHYcast Transcribe
import logging
from typing import Optional
from openai import OpenAI, BadRequestError
from config import MAX_TOKENS, MAX_INPUT_TOKENS, TEMPERATURE, MAX_CHUNK_SIZE, OPENAI_MODEL, OPENAI_LARGE_CONTEXT_MODEL
from utils.text_processing import estimate_token_count, split_into_chunks


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
        
        # Determine if we're using an o-series model (like o3-mini) that requires special parameter handling
        # NOTE: GPT-4o is NOT considered an o-series model in this context - it uses standard parameters
        is_o_series_model = model_name.startswith("o") and not model_name.startswith("gpt")
        
        # Set up parameters based on model type
        token_param = "max_completion_tokens" if is_o_series_model else "max_tokens"
        
        # Create base parameters dict
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
                {"role": "user", "content": f"{prompt}\n\nHere's the text to process:\n\n{text}"}
            ]
        }
        
        # Add temperature only for models that support it (non-"o" series)
        if not is_o_series_model:
            params["temperature"] = TEMPERATURE
        
        try:
            # For transcript cleanup, we need to ensure we get complete output by using a higher max_tokens limit
            if "clean" in prompt.lower() and "transcript" in prompt.lower():
                # For cleanup, give more tokens for output - use at least text length or double MAX_TOKENS
                cleanup_max_tokens = max(estimate_token_count(text), MAX_TOKENS * 2)
                logging.info(f"Using expanded token limit for cleanup: {cleanup_max_tokens}")
                
                # Check if text needs to be processed in chunks due to size
                if estimated_tokens > MAX_INPUT_TOKENS // 2:
                    return process_large_text_in_chunks(text, prompt, model_name, client)
                
                # Add token parameter
                params[token_param] = cleanup_max_tokens
                
                response = client.chat.completions.create(**params)
            else:
                # Regular processing for non-cleanup tasks
                # Add token parameter
                params[token_param] = max_tokens
                
                response = client.chat.completions.create(**params)
                
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
                
                # Update the message content with truncated text
                params["messages"][1]["content"] = f"{prompt}\n\nHere's the text to process:\n\n{text}"
                
                response = client.chat.completions.create(**params)
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
    
    # Determine if we're using an o-series model (like o3-mini) that requires special parameter handling
    # NOTE: GPT-4o is NOT considered an o-series model in this context - it uses standard parameters
    is_o_series_model = model_name.startswith("o") and not model_name.startswith("gpt")
    
    # Set up parameters based on model type
    token_param = "max_completion_tokens" if is_o_series_model else "max_tokens"
    token_limit = MAX_TOKENS * 2  # Use larger output limit for chunks
    
    modified_prompt = f"{prompt}\n\nThis is a chunk of a longer transcript. Process this chunk following the instructions."
    processed_chunks = []
    
    # Add progress bar for chunk processing
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        logging.info(f"Processing chunk {i+1}/{len(chunks)}")
        try:
            # Create parameters dictionary with proper token limit parameter
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that processes transcript chunks."},
                    {"role": "user", "content": f"{modified_prompt}\n\nChunk {i+1}/{len(chunks)}:\n\n{chunk}"}
                ]
            }
            
            # Add temperature only for models that support it (non-"o" series)
            if not is_o_series_model:
                params["temperature"] = TEMPERATURE
                
            # Add token parameter
            params[token_param] = token_limit
            
            response = client.chat.completions.create(**params)
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
            if combined_text > MAX_INPUT_TOKENS:
                logging.warning(f"Combined text is too large for final pass (~{combined_tokens} tokens)")
                return combined_text
            
            # Create parameters for the consistency pass
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that ensures transcript consistency."},
                    {"role": "user", "content": f"This is a processed transcript that was handled in chunks. Please ensure consistency across chunk boundaries and fix any obvious issues.\n\n{combined_text}"}
                ]
            }
            
            # Add temperature only for models that support it (non-"o" series)
            if not is_o_series_model:
                params["temperature"] = TEMPERATURE
                
            # Add token parameter
            params[token_param] = MAX_TOKENS
            
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in final consistency pass: {str(e)}")
            return combined_text
    
    return combined_text

