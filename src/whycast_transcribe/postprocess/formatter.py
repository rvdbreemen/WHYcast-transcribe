import logging
import os
import glob
from typing import Dict
from .cleanup import cleanup_transcript
from .summarizer import summarize_transcript
from .blog_generator import generate_blog, generate_blog_alt
from .history_generator import generate_history
from .speaker_assigner import assign_speakers
from whycast_transcribe.config import PROMPT_HISTORY_EXTRACT_FILE  # ensure at least one config import remains
from whycast_transcribe.utils.formatters import convert_markdown_to_html, convert_markdown_to_wiki


def write_workflow_outputs(
    base_filename: str,
    cleaned_transcript: str = None,
    summary: str = None,
    blog: str = None,
    blog_alt: str = None,
    history: str = None,
    speaker_assignment: str = None,
    force: bool = False
):
    """
    Write all generated content artifacts to files.
    Handles overwriting based on the force flag.
    Converts markdown outputs (blog, history, speaker assignment) to HTML and Wiki formats.
    """
    logging.info(f"Formatter: Starting to write workflow outputs for base: {base_filename}")
    outputs = {
        f"{base_filename}_cleaned.txt": cleaned_transcript,
        f"{base_filename}_summary.txt": summary,
        f"{base_filename}_blog.txt": blog,
        f"{base_filename}_blog_alt1.txt": blog_alt,
        f"{base_filename}_history.txt": history,
        f"{base_filename}_speaker_assignment.txt": speaker_assignment
    }

    for filepath, content in outputs.items():
        if content:
            logging.info(f"Formatter: Preparing to write {filepath}")
            if os.path.exists(filepath) and not force:
                logging.warning(f"Formatter: File exists, skipping write (use --force to overwrite): {filepath}")
                continue
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                logging.info(f"Formatter: Successfully wrote {filepath}")

                # Convert markdown files to HTML and Wiki
                if filepath.endswith(("_blog.txt", "_blog_alt1.txt", "_history.txt", "_speaker_assignment.txt")):
                    logging.info(f"Formatter: Converting {filepath} to HTML and Wiki formats")
                    html_path = filepath.replace(".txt", ".html")
                    wiki_path = filepath.replace(".txt", ".wiki")

                    html_content = convert_markdown_to_html(content)
                    wiki_content = convert_markdown_to_wiki(content)

                    if os.path.exists(html_path) and not force:
                        logging.warning(f"Formatter: HTML file exists, skipping write: {html_path}")
                    else:
                        with open(html_path, 'w', encoding='utf-8') as hf:
                            hf.write(html_content)
                        logging.info(f"Formatter: Successfully wrote {html_path}")

                    if os.path.exists(wiki_path) and not force:
                        logging.warning(f"Formatter: Wiki file exists, skipping write: {wiki_path}")
                    else:
                        with open(wiki_path, 'w', encoding='utf-8') as wf:
                            wf.write(wiki_content)
                        logging.info(f"Formatter: Successfully wrote {wiki_path}")
            except Exception as e:
                logging.error(f"Formatter: Failed to write or convert file {filepath}: {e}")
        else:
            logging.debug(f"Formatter: No content provided for {filepath}, skipping write.")
    
    logging.info(f"Formatter: Finished writing workflow outputs for base: {base_filename}")

def convert_existing_blogs(directory: str):
    """
    Convert existing blog .txt files in a directory to HTML and Wiki formats.
    """
    logging.info(f"Formatter: Starting conversion of existing blog files in directory: {directory}")
    blog_files = glob.glob(os.path.join(directory, "*_blog.txt"))
    blog_files += glob.glob(os.path.join(directory, "*_blog_alt1.txt"))
    logging.info(f"Formatter: Found {len(blog_files)} blog text files to convert")

    for txt_path in blog_files:
        logging.info(f"Formatter: Converting {txt_path}")
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            html_path = txt_path.replace(".txt", ".html")
            wiki_path = txt_path.replace(".txt", ".wiki")

            html_content = convert_markdown_to_html(content)
            wiki_content = convert_markdown_to_wiki(content)

            with open(html_path, 'w', encoding='utf-8') as hf:
                hf.write(html_content)
            logging.info(f"Formatter: Wrote {html_path}")
            with open(wiki_path, 'w', encoding='utf-8') as wf:
                wf.write(wiki_content)
            logging.info(f"Formatter: Wrote {wiki_path}")
        except Exception as e:
            logging.error(f"Formatter: Failed to convert {txt_path}: {e}")
    logging.info(f"Formatter: Finished converting existing blog files.")