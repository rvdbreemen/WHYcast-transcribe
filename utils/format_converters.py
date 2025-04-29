#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Format conversion utilities for WHYcast-transcribe.

This module provides functions for:
- Converting Markdown text to HTML with styling
- Converting Markdown text to MediaWiki markup
- Batch conversion of blog files to HTML and Wiki formats

Intended for use in all modules that require content format conversion for output or publishing.
"""

import os
import re
import glob
import logging
from tqdm import tqdm
import markdown

def convert_markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown text to HTML.
    
    Args:
        markdown_text: The markdown text to convert
        
    Returns:
        HTML formatted text
    """
    try:
        # Use the markdown library to convert text to HTML
        html = markdown.markdown(markdown_text, extensions=['extra', 'nl2br', 'sane_lists'])
        
        # Create a complete HTML document with basic styling
        html_document = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WHYcast Blog</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        a {{
            color: #0066cc;
        }}
        blockquote {{
            border-left: 4px solid #ccc;
            padding-left: 16px;
            margin-left: 0;
            color: #555;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""
        return html_document
    except Exception as e:
        logging.error(f"Error converting markdown to HTML: {str(e)}")
        # Return basic HTML with the original text if conversion fails
        return f"<!DOCTYPE html><html><body><pre>{markdown_text}</pre></body></html>"

def convert_markdown_to_wiki(markdown_text: str) -> str:
    """
    Convert markdown text to Wiki markup.
    
    Args:
        markdown_text: The markdown text to convert
        
    Returns:
        Wiki markup formatted text
    """
    try:
        # Basic conversion rules for common markdown to Wiki syntax
        wiki_text = markdown_text
        
        # Headers: Convert markdown headers to wiki headers
        # e.g., "# Heading 1" -> "= Heading 1 ="
        wiki_text = re.sub(r'^# (.+)$', r'= \1 =', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^## (.+)$', r'== \1 ==', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^### (.+)$', r'=== \1 ===', wiki_text, flags=re.MULTILINE)
        wiki_text = re.sub(r'^#### (.+)$', r'==== \1 ====', wiki_text, flags=re.MULTILINE)
        
        # Bold: Convert **text** or __text__ to '''text'''
        wiki_text = re.sub(r'\*\*(.+?)\*\*', r"'''\1'''", wiki_text)
        wiki_text = re.sub(r'__(.+?)__', r"'''\1'''", wiki_text)
        
        # Italic: Convert *text* or _text_ to ''text''
        wiki_text = re.sub(r'\*([^*]+?)\*', r"''\1''", wiki_text)
        wiki_text = re.sub(r'_([^_]+?)_', r"''\1''", wiki_text)
        
        # Lists: Convert markdown lists to wiki lists
        # Unordered lists: "- item" -> "* item"
        wiki_text = re.sub(r'^- (.+)$', r'* \1', wiki_text, flags=re.MULTILINE)
        
        # Ordered lists: "1. item" -> "# item"
        wiki_text = re.sub(r'^\d+\. (.+)$', r'# \1', wiki_text, flags=re.MULTILINE)
        
        # Links: Convert [text](url) to [url text]
        wiki_text = re.sub(r'\[(.+?)\]\((.+?)\)', r'[\2 \1]', wiki_text)
        
        # Code blocks: Convert ```code``` to <syntaxhighlight>code</syntaxhighlight>
        wiki_text = re.sub(r'```(.+?)```', r'<syntaxhighlight>\1</syntaxhighlight>', wiki_text, flags=re.DOTALL)
        
        # Inline code: Convert `code` to <code>code</code>
        wiki_text = re.sub(r'`(.+?)`', r'<code>\1</code>', wiki_text)
        
        # Blockquotes: Convert > quote to <blockquote>quote</blockquote>
        # First, group consecutive blockquote lines
        blockquote_blocks = re.findall(r'((?:^> .+\n?)+)', wiki_text, flags=re.MULTILINE)
        for block in blockquote_blocks:
            # Remove the > prefix from each line and wrap in blockquote tags
            cleaned_block = re.sub(r'^> (.+)$', r'\1', block, flags=re.MULTILINE).strip()
            wiki_text = wiki_text.replace(block, f'<blockquote>{cleaned_block}</blockquote>\n\n')
        
        return wiki_text
    except Exception as e:
        logging.error(f"Error converting markdown to Wiki markup: {str(e)}")
        return markdown_text  # Return original text if conversion fails

def convert_existing_blogs(directory: str) -> None:
    """
    Convert all existing blog.txt files in the given directory to HTML and Wiki formats.
    
    Args:
        directory: Directory containing blog text files
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        logging.error(f"Could not create or access directory {directory}: {str(e)}")
        return
    
    if not os.path.isdir(directory):
        logging.error(f"Invalid directory: {directory}")
        return
    
    # Look for blog files (txt files that have "_blog" in their name)
    blog_pattern = os.path.join(directory, "*_blog.txt")
    blog_files = glob.glob(blog_pattern)
    
    if not blog_files:
        logging.warning(f"No blog files found in directory: {directory}")
        return
    
    logging.info(f"Found {len(blog_files)} blog files to convert")
    converted_count = 0
    
    for blog_file in tqdm(blog_files, desc="Converting blogs"):
        base_filename = os.path.splitext(blog_file)[0]  # Remove .txt extension
        
        # Define output paths
        html_path = f"{base_filename}.html"
        wiki_path = f"{base_filename}.wiki"
        
        try:
            # Read the blog content
            with open(blog_file, 'r', encoding='utf-8') as f:
                blog_content = f.read()
            
            # Convert to HTML and save
            html_content = convert_markdown_to_html(blog_content)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            # Convert to Wiki and save
            wiki_content = convert_markdown_to_wiki(blog_content)
            with open(wiki_path, "w", encoding="utf-8") as f:
                f.write(wiki_content)
            
            logging.info(f"Converted {os.path.basename(blog_file)} to HTML and Wiki formats")
            converted_count += 1
            
        except Exception as e:
            logging.error(f"Error converting {blog_file}: {str(e)}")
    
    logging.info(f"Successfully converted {converted_count} out of {len(blog_files)} blog files")
