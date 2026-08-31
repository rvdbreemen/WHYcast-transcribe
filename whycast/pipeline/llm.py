"""
LLM (OpenAI) processing step for the WHYcast pipeline (ADR-008).

Functions extracted verbatim from transcribe.py (mechanical extraction,
lines 440-472, 575-648, 666-915): ensure_api_key, read_prompt_file,
estimate_token_count, truncate_transcript, choose_appropriate_model,
split_into_chunks, summarize_large_transcript, process_with_openai,
process_large_text_in_chunks. Only allowed transformation applied:
print() -> emit("llm", ...).
"""

import logging
import os
import uuid
from typing import List, Optional

from whycast._deps import OpenAI, BadRequestError, openai_available, tqdm
from whycast.config import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    MAX_INPUT_TOKENS,
    MAX_TOKENS,
    OPENAI_LARGE_CONTEXT_MODEL,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
)
from whycast.errors import ConfigurationError, SecurityError
from whycast.events import emit

logger = logging.getLogger(__name__)

# Chat models that still take the legacy ``max_tokens`` and accept a
# ``temperature``. Everything else - the o-series and every gpt-5.x model -
# takes ``max_completion_tokens``, rejects any temperature other than 1, and
# accepts ``reasoning_effort``.
#
# The list names the exceptions and the modern spelling is the default, which
# is the way round that survives: a model released after this line was written
# lands on the modern side without an edit. The previous spelling guessed from
# the model name's first letter - o-series unless it began with gpt - which put
# gpt-5.6-luna on the legacy side and made every call to it a 400. ADR-012
# forbids that shape, and the rule is a regex, so this comment describes the old
# heuristic rather than quoting it: quoting it would trip the check that exists
# to keep it out.
LEGACY_CHAT_MODEL_PREFIXES = ("gpt-4", "gpt-3.5")


def model_params(model_name: str, max_tokens: int, reasoning_effort: Optional[str] = None) -> dict:
    """The per-model half of a chat-completions payload.

    Args:
        model_name: The model the call is going to.
        max_tokens: Output budget, under whichever parameter name the model takes.
        reasoning_effort: none/low/medium/high/xhigh, or None to leave it to the
            API default. Dropped for legacy models, which do not accept it.

    Returns:
        The parameters that differ per model. Everything else in the payload is
        the same whatever the model is.
    """
    if model_name.startswith(LEGACY_CHAT_MODEL_PREFIXES):
        return {"max_tokens": max_tokens}

    params: dict = {"max_completion_tokens": max_tokens}
    if reasoning_effort:
        params["reasoning_effort"] = reasoning_effort
    return params


def _warn_if_truncated(call_id: str, response) -> None:
    """Log when the model stopped because it ran out of output budget.

    A reasoning model spends part of ``max_completion_tokens`` on thinking
    before it emits a word, so a budget that was ample for gpt-4.1 can leave a
    summary half-written. The API says so in ``finish_reason``; without this
    the only symptom is prose that stops mid-sentence.
    """
    try:
        finish_reason = response.choices[0].finish_reason
    except (AttributeError, IndexError):
        return
    if finish_reason == "length":
        logging.warning(
            f"[OpenAI Call {call_id}] Output hit the token limit (finish_reason=length); "
            "the result is truncated. Raise OPENAI_MAX_TOKENS or lower the reasoning effort."
        )

def ensure_api_key() -> str:
    """
    Ensure that the OpenAI API key is available and valid.
    
    Returns:
        str: The API key if available and valid
        
    Raises:
        ConfigurationError: If the API key is not set or invalid
        SecurityError: If the API key format is suspicious

    ConfigurationError rather than the bare ValueError this used to raise: a
    ValueError matched none of transcribe.py's except arms, so a missing key
    ended the CLI in a traceback while the web UI handled it cleanly. Both
    subclass WhycastError now, which is what ADR-008 asks the worker to
    translate. webui/runner.py:1158 catches both spellings.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set your OpenAI API key in the .env file or environment variables."
        )
    
    # Security: Basic validation of API key format
    api_key = api_key.strip()
    if len(api_key) < 20:
        raise ConfigurationError("OPENAI_API_KEY appears to be too short to be valid")
    
    # Security: Check for suspicious characters that could indicate injection
    if any(char in api_key for char in [';', '&', '|', '`', '$', '\n', '\r']):
        raise SecurityError("OPENAI_API_KEY contains suspicious characters")
    
    # Security: Basic format check for OpenAI API keys (should start with 'sk-')
    if not api_key.startswith('sk-'):
        logging.warning("OPENAI_API_KEY doesn't start with 'sk-', this may not be a valid OpenAI API key")
    
    return api_key


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
    """Pick the model for this transcript, by length (ADR-003).

    Args:
        transcript: The transcript text.

    Returns:
        ``OPENAI_LARGE_CONTEXT_MODEL`` for a transcript past
        ``MAX_INPUT_TOKENS``, otherwise ``OPENAI_MODEL``.

    The switch only does anything when the two env-vars actually name different
    models. They ship identical, because gpt-5.6-luna already handles the long
    inputs the large-context slot was introduced for, so by default this
    function returns the same model either way. It used to log "Using large
    context model: ..." regardless, which read like a decision had been taken
    when nothing had changed; that line now fires only on a real switch.
    """
    estimated_tokens = estimate_token_count(transcript)
    logging.info(f"Estimated transcript length: ~{estimated_tokens} tokens")

    if estimated_tokens > MAX_INPUT_TOKENS and OPENAI_LARGE_CONTEXT_MODEL:
        if OPENAI_LARGE_CONTEXT_MODEL != OPENAI_MODEL:
            logging.info(
                f"Transcript is long (~{estimated_tokens} tokens); switching from "
                f"{OPENAI_MODEL} to OPENAI_LARGE_CONTEXT_MODEL {OPENAI_LARGE_CONTEXT_MODEL}"
            )
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
        if (paragraph_break != -1 and paragraph_break > start + max_chunk_size // 2):
            end = paragraph_break + 2
        else:
            # Try to find sentence break (period followed by space)
            sentence_break = text.rfind('. ', start, end)
            if sentence_break != -1 and sentence_break > start + max_chunk_size // 2:
                end = sentence_break + 2
        
        chunks.append(text[start:end])

        # The text is covered once end reaches it. Without this the loop kept
        # going: start would move to len(text) - overlap, which is still short
        # of the end, and from there end never changed again, so start crept
        # forward one character per round for another `overlap` rounds. That
        # appended ~1000 shrinking scraps, each of which the callers turn into
        # a paid API call.
        if end >= len(text):
            break

        # Start the next chunk with some overlap for context. The max() keeps
        # start moving even when a break point lands within `overlap` of it.
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
        except ConfigurationError:
            # A missing package or key is not a bad chunk: retrying the other
            # chunks cannot help and would spend money doing it. ADR-008 wants
            # this at the worker, not logged away here.
            raise
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

def process_with_openai(text: str, prompt: str, model_name: str, max_tokens: int = MAX_TOKENS,
                        reasoning_effort: Optional[str] = OPENAI_REASONING_EFFORT) -> Optional[str]:
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
    # Both of these are deliberately OUTSIDE the try below. That block ends in
    # `except Exception: return None`, so anything raised inside it is swallowed
    # into "no result" - which is the opposite of what ADR-008 asks for: library
    # code raises and the worker translates. A missing package or a missing API
    # key is a configuration fault a person has to fix, not an empty answer.
    if not openai_available:
        # Without openai the shim is None, and OpenAI(api_key=...) would raise
        # "'NoneType' object is not callable", naming neither the package nor
        # the fix.
        raise ConfigurationError(
            "The openai package is not installed, so no language-model step "
            "can run. Install it with 'pip install openai'."
        )
    api_key = ensure_api_key()

    # Bound before the try because the handler formats it. It used to be
    # assigned after two statements that can throw, so an early failure died
    # with "UnboundLocalError: cannot access local variable 'call_id'" and the
    # real error was never logged.
    call_id = str(uuid.uuid4())

    try:
        client = OpenAI(api_key=api_key)
        prompt_first_line = prompt.strip().splitlines()[0] if prompt.strip().splitlines() else prompt.strip()
        logging.info(f"[OpenAI Call {call_id}] Model: {model_name}, Prompt first line: {prompt_first_line}")
        emit("llm", f"[OpenAI Call {call_id}] Prompt: {prompt_first_line}")
        
        estimated_tokens = estimate_token_count(text)
        logging.info(f"[OpenAI Call {call_id}] Processing text with OpenAI (~{estimated_tokens} tokens)")
        
        # If text is too long, truncate it
        if estimated_tokens > MAX_INPUT_TOKENS:
            logging.warning(f"[OpenAI Call {call_id}] Text too long (~{estimated_tokens} tokens > {MAX_INPUT_TOKENS} limit), truncating...")
            text = truncate_transcript(text, MAX_INPUT_TOKENS)
        
        # Create base parameters dict. Everything that varies per model is added
        # below by model_params().
        params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that processes transcripts."},
                {"role": "user", "content": f"{prompt}\n\nHere's the text to process:\n\n{text}"}
            ]
        }

        try:
            # For transcript cleanup, we need to ensure we get complete output by using a higher max_tokens limit
            if "clean" in prompt.lower() and "transcript" in prompt.lower():
                # For cleanup, give more tokens for output - use at least text length or double MAX_TOKENS
                cleanup_max_tokens = max(estimate_token_count(text), MAX_TOKENS * 2)
                logging.info(f"[OpenAI Call {call_id}] Using expanded token limit for cleanup: {cleanup_max_tokens}")
                
                # Check if text needs to be processed in chunks due to size
                if estimated_tokens > MAX_INPUT_TOKENS // 2:
                    return process_large_text_in_chunks(text, prompt, model_name, client, parent_call_id=call_id,
                                                                       reasoning_effort=reasoning_effort)
                
                params.update(model_params(model_name, cleanup_max_tokens, reasoning_effort))

                response = client.chat.completions.create(**params)
            else:
                # Regular processing for non-cleanup tasks
                params.update(model_params(model_name, max_tokens, reasoning_effort))

                response = client.chat.completions.create(**params)

            _warn_if_truncated(call_id, response)
            result = response.choices[0].message.content

            # Guess at truncation from the last character, but only when the API
            # has not already said the call finished cleanly. finish_reason is
            # authoritative and _warn_if_truncated above reports it; this
            # heuristic fires on any answer ending in a table row, a code fence
            # or a list item, which the speaker analysis does routinely. Left in
            # as a second signal for the case where finish_reason is missing.
            if (response.choices[0].finish_reason != "stop"
                    and len(result) > 100
                    and not result.rstrip().endswith(('.', '!', '?', '"', ':', ';', ')', ']', '}'))):
                logging.warning(f"[OpenAI Call {call_id}] Generated text may be truncated (doesn't end with punctuation)")
                
            return result
                
        except BadRequestError as e:
            if "maximum context length" in str(e).lower():
                # Try with more aggressive truncation
                logging.warning(f"[OpenAI Call {call_id}] Context length exceeded. Retrying with further truncation...")
                text = truncate_transcript(text, MAX_INPUT_TOKENS // 2)
                logging.info(f"[OpenAI Call {call_id}] Retrying with reduced text (~{estimate_token_count(text)} tokens)...")
                
                # Update the message content with truncated text
                params["messages"][1]["content"] = f"{prompt}\n\nHere's the text to process:\n\n{text}"
                
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content
            else:
                raise
    except Exception as e:
        logging.error(f"[OpenAI Call {call_id}] Error processing with OpenAI: {str(e)}")
        return None

def process_large_text_in_chunks(text: str, prompt: str, model_name: str, client: OpenAI, parent_call_id: str = None,
                                 reasoning_effort: Optional[str] = OPENAI_REASONING_EFFORT) -> str:
    """
    Process very large text by breaking it into chunks and reassembling the results.
    
    Args:
        text: The text to process
        prompt: The processing instructions
        model_name: Model to use
        client: OpenAI client
        parent_call_id: Optional parent call ID for traceability
    Returns:
        Combined processed text
    """
    call_id = str(uuid.uuid4())
    logging.info(f"[OpenAI Chunked Call {call_id}] Parent: {parent_call_id} | Model: {model_name} | Chunks incoming")
    emit("llm", f"[OpenAI Chunked Call {call_id}] Parent: {parent_call_id} | Prompt: {prompt.strip().splitlines()[0] if prompt.strip().splitlines() else prompt.strip()}")
    token_limit = MAX_TOKENS * 2
    modified_prompt = f"{prompt}\n\nThis is a chunk of a longer transcript. Process this chunk following the instructions."
    processed_chunks = []
    for i, chunk in enumerate(tqdm(split_into_chunks(text, max_chunk_size=MAX_CHUNK_SIZE*2), desc="Processing chunks")):
        chunk_call_id = str(uuid.uuid4())
        chunk_first_line = modified_prompt.strip().splitlines()[0] if modified_prompt.strip().splitlines() else modified_prompt.strip()
        logging.info(f"[OpenAI Chunk {chunk_call_id}] Parent: {call_id} | Chunk {i+1} | Prompt: {chunk_first_line}")
        emit("llm", f"[OpenAI Chunk {chunk_call_id}] Parent: {call_id} | Chunk {i+1} | Prompt: {chunk_first_line}")
        try:
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that processes transcript chunks."},
                    {"role": "user", "content": f"{modified_prompt}\n\nChunk {i+1} of text:\n\n{chunk}"}
                ]
            }
            params.update(model_params(model_name, token_limit, reasoning_effort))
            response = client.chat.completions.create(**params)
            _warn_if_truncated(chunk_call_id, response)
            processed_chunks.append(response.choices[0].message.content)
            logging.info(f"[OpenAI Chunk {chunk_call_id}] Successfully processed chunk {i+1}")
        except Exception as e:
            logging.error(f"[OpenAI Chunk {chunk_call_id}] Error processing chunk {i+1}: {str(e)}")
            processed_chunks.append(chunk)
    combined_text = "\n\n".join(processed_chunks)
    if len(processed_chunks) > 1:
        try:
            logging.info(f"[OpenAI Chunked Call {call_id}] Running final pass to ensure consistency across chunk boundaries")
            combined_tokens = estimate_token_count(combined_text)
            if combined_tokens > MAX_INPUT_TOKENS:
                logging.warning(f"[OpenAI Chunked Call {call_id}] Combined text is too large for final pass (~{combined_tokens} tokens)")
                return combined_text
            params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that ensures transcript consistency."},
                    {"role": "user", "content": f"This is a processed transcript that was handled in chunks. Please ensure consistency across chunk boundaries and fix any obvious issues.\n\n{combined_text}"}
                ]
            }
            params.update(model_params(model_name, MAX_TOKENS, reasoning_effort))
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"[OpenAI Chunked Call {call_id}] Error in final consistency pass: {str(e)}")
            return combined_text
    return combined_text

