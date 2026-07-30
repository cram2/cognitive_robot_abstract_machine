"""
Shared rendering helpers for the plan-dashboard scripts.

No plan-specific content belongs here - this module only knows how to (a)
build the Jinja2 environment every page template renders through and (b)
turn generic markdown into HTML. build_dashboard.py and build_index.py both
import it rather than duplicating the logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

import jinja2
import markdown as markdown_library
import nh3

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})

TEMPLATES_DIRECTORY = Path(__file__).parent / "templates"
"""
Where every page template (dashboard.html, index.html) lives.
"""

# Matches one opening or closing HTML heading tag, e.g. "<h1>"/"</h1>" through
# "<h6>"/"</h6>". _shift_heading_level() shifts each match down by
# _HEADING_LEVEL_SHIFT and caps it at _MAXIMUM_HEADING_LEVEL, so a roadmap's
# own "<h1>Title</h1>" becomes "<h4>Title</h4>" once embedded in a dashboard
# page, and a roadmap's "<h5>Deep</h5>" becomes "<h6>Deep</h6>" - shifted but
# capped, rather than overflowing to a nonexistent "<h8>".
_HEADING_TAG_PATTERN = re.compile(r"<(/?)h([1-6])>")

_HEADING_LEVEL_SHIFT = 3
"""
How many levels to push a roadmap's own headings down by, so they nest below the
embedding dashboard page's own h1-h3.
"""

_MAXIMUM_HEADING_LEVEL = 6
"""The deepest valid HTML heading level - shifting never produces anything
past this, even for an already-deep roadmap heading."""


def create_template_environment() -> jinja2.Environment:
    """
    Build the Jinja2 environment every page template renders through.

    Autoescaping is on for every value substituted into a template - the
    one deliberately unescaped value either script ever passes in
    (already-rendered roadmap markdown HTML) is marked with Jinja2's
    ``| safe`` filter at its point of use in the template itself, rather
    than disabling escaping globally and trusting every call site to
    remember to escape by hand.

    :return: A configured, ready-to-use Jinja2 environment.
    """
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIRECTORY),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_markdown_to_html(markdown_text: str) -> str:
    """
    Render GitHub-flavored markdown (headings, lists, code, tables) to HTML.

    Delegates to the ``markdown`` library (with its ``tables`` and
    ``fenced_code`` extensions) rather than a hand-rolled parser, then
    sanitizes the result with ``nh3`` - the ``markdown`` library passes raw
    HTML embedded in its source through unchanged, and this function's
    output is later marked ``| safe`` and embedded directly into a published
    page, so a ``<script>`` tag or an event-handler attribute in a
    contributor-authored ``roadmap.md`` must not survive to here.

    :param markdown_text: The raw markdown source (typically a plan's
        ``roadmap.md``).
    :return: The rendered, sanitized HTML. Callers embedding this into a
        Jinja2 template must mark it ``| safe`` - it is HTML, not text to
        escape.
    """
    html_text = markdown_library.markdown(
        markdown_text, extensions=["tables", "fenced_code"]
    )
    sanitized_html_text = nh3.clean(html_text, link_rel=None)
    return _HEADING_TAG_PATTERN.sub(_shift_heading_level, sanitized_html_text)


def sanitize_http_url(url: str | None) -> str | None:
    """
    Reject a manifest- or index-authored URL that isn't ``http``/``https``
    before it reaches an ``<a href>`` - a ``javascript:`` or ``data:`` value
    would otherwise render as a clickable, script-executing link.

    :param url: A raw URL from plan.yaml (``session``) or plans.json
        (``dashboard_url``), or ``None`` if unset.
    :return: *url* unchanged if its scheme is ``http`` or ``https``;
        ``None`` otherwise.
    """
    if url is None or urlsplit(url).scheme.lower() not in _ALLOWED_URL_SCHEMES:
        return None
    return url


def _shift_heading_level(heading_tag_match: re.Match[str]) -> str:
    """
    Shift one ``<h1>``-``<h6>`` tag match down by :data:`_HEADING_LEVEL_SHIFT`.

    Used as the substitution callback for :data:`_HEADING_TAG_PATTERN`, so a
    roadmap's own ``<h1>`` renders as the embedding dashboard page's
    ``<h4>``, staying below the dashboard's own h1-h3 - capped at h6 so a
    deeply-nested roadmap heading never overflows past the last valid level.

    :param heading_tag_match: A match of :data:`_HEADING_TAG_PATTERN` -
        group 1 is ``"/"`` for a closing tag or ``""`` for an opening one,
        group 2 is the original heading level.
    :return: The tag with its level shifted.
    """
    closing_slash, original_level = heading_tag_match.group(1), int(
        heading_tag_match.group(2)
    )
    shifted_level = min(original_level + _HEADING_LEVEL_SHIFT, _MAXIMUM_HEADING_LEVEL)
    return f"<{closing_slash}h{shifted_level}>"
