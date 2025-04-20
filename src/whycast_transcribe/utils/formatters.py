import markdown
import re
import logging


def convert_markdown_to_html(markdown_text: str) -> str:
    """
    Convert markdown text to HTML string.
    """
    try:
        html = markdown.markdown(markdown_text, extensions=['extra', 'nl2br', 'sane_lists'])
        return html
    except Exception as e:
        logging.error(f"Error converting markdown to HTML: {e}")
        return markdown_text


def convert_markdown_to_wiki(markdown_text: str) -> str:
    """
    Convert markdown text to basic Wiki markup.
    """
    try:
        wiki = markdown_text
        wiki = re.sub(r'^# (.+)$', r'= \1 =', wiki, flags=re.MULTILINE)
        wiki = re.sub(r'^## (.+)$', r'== \1 ==', wiki, flags=re.MULTILINE)
        wiki = re.sub(r'\*\*(.+?)\*\*', r"'''\1'''", wiki)
        wiki = re.sub(r'\*(.+?)\*', r"''\1''", wiki)
        wiki = re.sub(r'^- (.+)$', r'* \1', wiki, flags=re.MULTILINE)
        wiki = re.sub(r'\[(.+?)\]\((.+?)\)', r'[\2 \1]', wiki)
        # Convert inline list items ' - item' to separate lines '* item'
        wiki = re.sub(r' - (.+)', r'* \1', wiki)
        return wiki
    except Exception as e:
        logging.error(f"Error converting markdown to Wiki: {e}")
        return markdown_text