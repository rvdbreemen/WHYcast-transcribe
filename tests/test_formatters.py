from whycast_transcribe.utils.formatters import convert_markdown_to_html, convert_markdown_to_wiki

def test_convert_markdown_to_html():
    md = '# Title\n- item'
    html = convert_markdown_to_html(md)
    assert '<h1' in html and '<li>' in html

def test_convert_markdown_to_wiki():
    md = '# Title\n## Subtitle\n**bold** *italic* - item'
    wiki = convert_markdown_to_wiki(md)
    assert '= Title =' in wiki
    assert '== Subtitle ==' in wiki
    assert "'''bold'''" in wiki
    assert "''italic''" in wiki
    assert '* item' in wiki