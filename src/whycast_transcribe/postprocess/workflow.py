import logging
import os
from .cleanup import cleanup_transcript
from .summarizer import summarize_transcript
from .blog_generator import generate_blog, generate_blog_alt
from .history_generator import generate_history
from .speaker_assigner import assign_speakers
from .formatter import write_workflow_outputs

def run_postprocessing(segments, input_path):
    """
    Execute cleanup, summarization, blog, history extraction, and speaker assignment workflows.
    Write output files (text, HTML, Wiki) alongside original transcript.
    """
    base = input_path.rsplit('.', 1)[0]
    logging.info("Starting post-processing workflow")

    # Build transcript text
    transcript_text = "\n".join(seg.text for seg in segments)

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

    # Write outputs
    write_workflow_outputs(results, base)
    logging.info("Post-processing workflow completed")
