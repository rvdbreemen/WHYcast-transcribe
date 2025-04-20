import logging
from datetime import datetime
from whycast_transcribe.config import (
    PROMPT_CLEANUP_FILE, PROMPT_SUMMARY_FILE, PROMPT_BLOG_FILE,
    PROMPT_BLOG_ALT1_FILE, PROMPT_HISTORY_EXTRACT_FILE, PROMPT_SPEAKER_ASSIGN_FILE,
    OPENAI_MODEL, OPENAI_LARGE_CONTEXT_MODEL, OPENAI_HISTORY_MODEL, OPENAI_SPEAKER_MODEL,
    TEMPERATURE, MAX_TOKENS, MAX_INPUT_TOKENS
)
from whycast_transcribe.postprocess.cleanup import cleanup_transcript
from whycast_transcribe.postprocess.summarizer import summarize_transcript
from whycast_transcribe.postprocess.blog_generator import generate_blog, generate_blog_alt
from whycast_transcribe.postprocess.history_generator import generate_history
from whycast_transcribe.postprocess.speaker_assigner import assign_speakers
from whycast_transcribe.postprocess.formatter import write_workflow_outputs

def run_postprocessing(segments, input_path):
    """
    Execute cleanup, summarization, blog, history extraction, and speaker assignment workflows.
    Write output files (text, HTML, Wiki) alongside original transcript.
    """
    base = input_path.rsplit('.', 1)[0]
    logging.info("Starting post-processing workflow")

    # Build transcript text from segments
    transcript_text = "\n".join(seg.text for seg in segments)

    # Execute individual post-processing steps
    results = {}
    cleaned = cleanup_transcript(transcript_text)
    results['cleaned_transcript'] = cleaned
    summary = summarize_transcript(cleaned)
    results['summary'] = summary
    blog = generate_blog(cleaned, summary)
    results['blog'] = blog
    blog_alt1 = generate_blog_alt(cleaned, summary)
    results['blog_alt1'] = blog_alt1
    history = generate_history(cleaned)
    results['history_extract'] = history
    speaker_assignment = assign_speakers(cleaned)
    results['speaker_assignment'] = speaker_assignment

    # Write outputs to files in various formats
    write_workflow_outputs(results, base)

    logging.info("Post-processing workflow completed")