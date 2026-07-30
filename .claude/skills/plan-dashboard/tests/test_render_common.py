"""
Tests for render_common.py: the Jinja2 environment factory and markdown-to-HTML
rendering (delegated to the `markdown` library, plus a heading-level shift on top).
"""

from render_common import (
    create_template_environment,
    render_markdown_to_html,
    sanitize_http_url,
)

# %% create_template_environment


def test_template_environment_finds_the_committed_templates():
    environment = create_template_environment()
    assert environment.get_template("dashboard.html")
    assert environment.get_template("index.html")


def test_template_environment_autoescapes_html():
    environment = create_template_environment()
    rendered = environment.from_string("{{ value }}").render(value="<script>")
    assert rendered == "&lt;script&gt;"


def test_template_environment_safe_filter_bypasses_autoescaping():
    environment = create_template_environment()
    rendered = environment.from_string("{{ value | safe }}").render(value="<b>bold</b>")
    assert rendered == "<b>bold</b>"


# %% render_markdown_to_html - heading level shift


def test_heading_level_is_shifted_and_capped():
    # h1 -> h4 (shifted by 3), h6 -> h6 (capped, not h9)
    assert render_markdown_to_html("# Title") == "<h4>Title</h4>"
    assert render_markdown_to_html("###### Deep") == "<h6>Deep</h6>"


def test_heading_closing_tag_is_shifted_too():
    rendered = render_markdown_to_html("## Section\ntext")
    assert "<h5>Section</h5>" in rendered
    assert "</h5>" in rendered


# %% render_markdown_to_html - delegates block/inline parsing to `markdown`


def test_paragraph():
    assert render_markdown_to_html("Just a paragraph.") == "<p>Just a paragraph.</p>"


def test_blank_line_separates_paragraphs():
    rendered = render_markdown_to_html("First.\n\nSecond.")
    assert rendered == "<p>First.</p>\n<p>Second.</p>"


def test_unordered_list():
    rendered = render_markdown_to_html("- first\n- second")
    assert "<li>first</li>" in rendered
    assert "<li>second</li>" in rendered


def test_ordered_list():
    rendered = render_markdown_to_html("1. one\n2. two")
    assert "<ol>" in rendered
    assert "<li>one</li>" in rendered
    assert "<li>two</li>" in rendered


def test_fenced_code_block_is_not_interpreted_as_markdown():
    rendered = render_markdown_to_html("```\n**not bold**\n```")
    assert "<pre><code>" in rendered
    assert "**not bold**" in rendered
    assert "<strong>" not in rendered


def test_github_flavored_markdown_table_renders_header_and_rows():
    markdown_text = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    rendered = render_markdown_to_html(markdown_text)
    assert "<th>A</th>" in rendered
    assert "<th>B</th>" in rendered
    assert "<td>1</td>" in rendered
    assert "<td>2</td>" in rendered


def test_inline_bold_italic_code_and_links():
    rendered = render_markdown_to_html(
        "**bold** *italic* `code` [text](https://example.com)"
    )
    assert "<strong>bold</strong>" in rendered
    assert "<em>italic</em>" in rendered
    assert "<code>code</code>" in rendered
    assert '<a href="https://example.com">text</a>' in rendered


def test_mixed_blocks_render_in_source_order():
    markdown_text = "# Title\n\nParagraph.\n\n- item"
    rendered = render_markdown_to_html(markdown_text)
    title_index = rendered.index("<h4>Title</h4>")
    paragraph_index = rendered.index("<p>Paragraph.</p>")
    list_index = rendered.index("<li>item</li>")
    assert title_index < paragraph_index < list_index


# %% render_markdown_to_html - sanitizes raw HTML in the source


def test_script_tag_is_stripped():
    rendered = render_markdown_to_html("<script>alert('xss')</script>\n\nsafe text")
    assert rendered == "\n\n<p>safe text</p>"


def test_event_handler_attribute_is_stripped():
    rendered = render_markdown_to_html('<img src="x" onerror="alert(1)">')
    assert rendered == '<p><img src="x"></p>'


def test_javascript_url_is_stripped():
    rendered = render_markdown_to_html("[click me](javascript:alert(1))")
    assert rendered == "<p><a>click me</a></p>"


# %% sanitize_http_url


def test_sanitize_http_url_keeps_an_http_url():
    assert sanitize_http_url("http://example.com") == "http://example.com"


def test_sanitize_http_url_keeps_an_https_url():
    assert sanitize_http_url("https://example.com/plan") == "https://example.com/plan"


def test_sanitize_http_url_rejects_a_javascript_url():
    assert sanitize_http_url("javascript:alert(1)") is None


def test_sanitize_http_url_rejects_a_data_url():
    assert sanitize_http_url("data:text/html,<script>alert(1)</script>") is None


def test_sanitize_http_url_passes_through_none():
    assert sanitize_http_url(None) is None
