from __future__ import annotations

import logging
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing_extensions import TYPE_CHECKING, Optional

from jinja2 import Template

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.verbalization.context import MicroplanningServices
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.rendering.formatter import (
    ANSIFormatter,
    HTMLFormatter,
    PlainFormatter,
    detect_osc8_support,
)
from krrood.entity_query_language.verbalization.rendering.renderer import (
    FragmentRenderer,
    HierarchicalRenderer,
    ParagraphRenderer,
)
from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.query import Query

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.rendering.source_link_resolver import (
        SourceLinkResolver,
    )
    from krrood.entity_query_language.backends import QueryBackend
    from krrood.entity_query_language.verbalization.vocabulary.english import Directive

logger = logging.getLogger(__name__)


def directive_for_backend(backend: Optional[QueryBackend]) -> Optional[Directive]:
    """
    Resolve the opening directive implied by an evaluating backend.

    Each backend declares its own :attr:`~…backends.QueryBackend.opening_directive`, so this stays
    decoupled from the concrete backend types (no import of the backends module, no ``isinstance``).

    :param backend: The backend the expression would be evaluated with, or ``None``.
    :return: ``GENERATE`` for a generative backend, ``FIND`` for a selective one, or ``None`` when
        no backend is given (keep the query-type default).
    """
    return backend.opening_directive if backend is not None else None


_HTML_PAGE_TEMPLATE = Template("""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: monospace;
    font-size: 14px;
    padding: 1.5em;
    line-height: 1.8;
  }
  a { color: inherit; }
</style>
</head>
<body>{{ body }}</body>
</html>
""")
"""Standalone dark page for browser display; the rendered markup fills ``body``."""

_HTML_CELL_WRAPPER = Template(
    '<div style="background:#1e1e1e;color:#d4d4d4;font-family:monospace;'
    'font-size:14px;padding:0.75em;border-radius:0.4em;line-height:1.8;">'
    "{{ body }}"
    "</div>"
)
"""Inline dark wrapper for HTML cell output (Jupyter / built docs); mirrors the page colours so
the markup reads correctly in both environments."""


def _is_ipython() -> bool:
    """Return ``True`` when running inside an IPython / Jupyter session."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


@dataclass
class VerbalizationPipeline:
    """
    Pairs a fragment builder with a renderer to produce a finished string in one output mode:

    * plain — prose, no colour;
    * ANSI — true-colour terminal output;
    * HTML — coloured ``<span>`` markup for Jupyter.

    Each mode accepts an optional source-link resolver for hyperlinks, and an optional shared
    context so repeated mentions corefer across calls.
    """

    renderer: FragmentRenderer = field(default_factory=ParagraphRenderer)
    """Renderer that converts the fragment tree to a string."""

    _verbalizer: EQLVerbalizer = field(default_factory=EQLVerbalizer, init=False)
    """The verbalizer that builds fragment trees from EQL expressions."""

    def verbalize(
        self,
        expression: SymbolicExpression,
        services: Optional[MicroplanningServices] = None,
        backend: Optional[QueryBackend] = None,
    ) -> str:
        """
        Verbalize *expression* to a string using this pipeline's renderer.

        :param expression: Any EQL expression or query.
        :param services: Shared verbalization state; created automatically when omitted.  Pass the
            same services across calls so repeated mentions corefer (a Robot … the Robot).
        :param backend: The backend the expression would be evaluated with. When given it decides
            the opening verb (generative → *"Generate"*, selective → *"Find"*); when omitted the
            verb is derived from the query type as before.
        :return: Formatted natural-language string (plain, ANSI, or HTML, per the renderer).

        It runs the full path — build the fragment tree, then render it — whereas
        :meth:`verbalize_fragment` renders an already-built fragment.

        >>> VerbalizationPipeline.plain().verbalize(a(entity(variable(Robot, []))))
        'Find a Robot'
        """
        if isinstance(expression, Match):
            expression.expression.build()
        elif isinstance(expression, Query):
            expression.build()
        fragment = self._verbalizer.build(
            expression, services, performative=directive_for_backend(backend)
        )
        return self.verbalize_fragment(fragment)

    def _is_html_renderer(self) -> bool:
        """:return: ``True`` when this pipeline's renderer emits HTML."""
        return isinstance(self.renderer.formatter, HTMLFormatter)

    def verbalize_fragment(self, fragment: VerbalizationFragment) -> str:
        """
        Render a pre-built fragment using this pipeline's renderer.

        HTML output is wrapped in a dark container suitable for inline display.

        :param fragment: Root of the fragment tree to render.
        :return: Formatted string.

        Its contribution is the render half only: given a tree already built by
        :class:`EQLVerbalizer`, it applies this pipeline's renderer (and HTML wrapping) without
        re-building — the step :meth:`verbalize` calls after building.

        >>> tree = EQLVerbalizer().build(a(entity(variable(Robot, []))))
        >>> VerbalizationPipeline.plain().verbalize_fragment(tree)
        'Find a Robot'
        """
        result = self.renderer.render(fragment)
        if self._is_html_renderer():
            return _HTML_CELL_WRAPPER.render(body=result)
        return result

    def display(self, expression: SymbolicExpression) -> None:
        """
        Render *expression* and display it in the current environment — inline in
        Jupyter / IPython, or in the default browser elsewhere.

        :param expression: Any EQL expression or query.
        """
        self.display_fragment(self._verbalizer.build(expression))

    def display_fragment(self, fragment: VerbalizationFragment) -> None:
        """
        Display a pre-built fragment, with the same environment routing as ``display``.

        :param fragment: Root of the fragment tree to display.
        """
        raw_html = self.renderer.render(fragment)
        if _is_ipython():
            from IPython.display import display as _ipython_display, HTML

            wrapped = (
                _HTML_CELL_WRAPPER.render(body=raw_html)
                if self._is_html_renderer()
                else raw_html
            )
            _ipython_display(HTML(wrapped))
            return
        full_page = _HTML_PAGE_TEMPLATE.render(body=raw_html)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
            encoding="utf-8",
        ) as html_file:
            html_file.write(full_page)
            html_path = Path(html_file.name)
        webbrowser.open(html_path.as_uri())

    # %% Factories

    @classmethod
    def plain(cls) -> VerbalizationPipeline:
        """
        Create a plain-text, paragraph-prose pipeline with no colour markup.

        :return: A plain-text pipeline.
        """
        return cls(ParagraphRenderer(PlainFormatter()))

    @classmethod
    def ansi(
        cls,
        hierarchical: bool = False,
        link_resolver: Optional[SourceLinkResolver] = None,
    ) -> VerbalizationPipeline:
        """
        Create an ANSI true-colour (24-bit) pipeline for terminal display.

        When *link_resolver* is given and the terminal supports OSC 8 hyperlinks, class and
        attribute names become clickable; on unsupported terminals the resolver is disabled
        with a warning.

        :param hierarchical: When ``True``, render indented bullets instead of paragraph prose.
        :param link_resolver: Optional resolver mapping source references to URLs.
        :return: An ANSI-coloured pipeline.
        """
        formatter = ANSIFormatter()
        if link_resolver is not None and not detect_osc8_support():
            logger.warning(
                "The current terminal does not appear to support OSC 8 hyperlinks "
                "(VTE_VERSION / TERM_PROGRAM / TERM not recognised). "
                "link_resolver will be ignored for ANSI output."
            )
            link_resolver = None
        renderer: FragmentRenderer = (
            HierarchicalRenderer(formatter, link_resolver)
            if hierarchical
            else ParagraphRenderer(formatter, link_resolver)
        )
        return cls(renderer)

    @classmethod
    def html(
        cls,
        hierarchical: bool = False,
        link_resolver: Optional[SourceLinkResolver] = None,
    ) -> VerbalizationPipeline:
        """
        Create an HTML ``<span>`` colour pipeline for Jupyter / inline-HTML rendering.

        When *link_resolver* is given, class and attribute names are wrapped in anchors
        pointing to documentation pages.

        :param hierarchical: When ``True``, render indented bullets instead of paragraph prose.
        :param link_resolver: Optional resolver mapping source references to URLs.
        :return: An HTML-coloured pipeline.
        """
        formatter = HTMLFormatter()
        renderer = (
            HierarchicalRenderer(formatter, link_resolver)
            if hierarchical
            else ParagraphRenderer(formatter, link_resolver)
        )
        return cls(renderer)


def verbalize_expression(
    expression: SymbolicExpression, backend: Optional[QueryBackend] = None
) -> str:
    """
    Verbalize any EQL expression into a plain-text English phrase.

    :param expression: Any EQL expression or query.
    :param backend: The backend the expression would be evaluated with. When given it decides the
        opening verb (generative → *"Generate"*, selective → *"Find"*); when omitted the verb is
        derived from the query type as before.
    :return: Plain-text natural-language string.

    >>> verbalize_expression(a(entity(variable(Robot, []))))
    'Find a Robot'
    >>> robot = variable(Robot, [])
    >>> verbalize_expression(a(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'
    """
    return VerbalizationPipeline.plain().verbalize(expression, backend=backend)
