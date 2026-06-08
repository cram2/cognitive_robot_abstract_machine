"""
Fragment renderers — convert a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
tree into a string.

* :class:`ParagraphRenderer` — flattens everything into one prose line.
* :class:`HierarchicalRenderer` — indented bullet list, one bullet per block item.

Both accept a :class:`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter`
for colour / spacing control and an optional
:class:`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver`
for hyperlinks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    fold_fragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.rendering.formatter import (
    BulletStyle,
    Formatter,
    IndentSize,
    PlainFormatter,
    _first_docstring_line,
    _docstring_for_source_ref,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.rendering.source_link_resolver import (
        SourceLinkResolver,
    )


@dataclass
class FragmentRenderer(ABC):
    """
    Abstract base that converts a
    :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
    tree into a string.

    Subclasses differ in how they handle
    :class:`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment` nesting:

    * :class:`ParagraphRenderer` — flattens everything into one prose string.
    * :class:`HierarchicalRenderer` — renders blocks as indented bullet lists.
    """

    _formatter: Formatter = field(default_factory=PlainFormatter)
    """Format-specific markup logic (plain, ANSI, HTML)."""

    _link_resolver: Optional[SourceLinkResolver] = field(default=None)
    """Optional resolver that maps SourceRef instances to URL strings."""

    @abstractmethod
    def render(self, fragment: VerbFragment) -> str:
        """
        Render a :class:`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
        tree into a string.

        :param fragment: Root of the fragment tree to render.
        :type fragment: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        :return: Formatted string representation.
        :rtype: str
        """
        ...

    def _render_role(self, text: str, role, source_ref) -> str:
        """
        Colorise *text* for *role* and, when a resolver and source ref are present,
        wrap the result with a hyperlink.

        :param text: Plain display text.
        :type text: str
        :param role: Semantic role for colour lookup.
        :param source_ref: Source reference for link resolution; may be ``None``.
        :return: Coloured (and optionally linked) string.
        :rtype: str
        """
        colored = self._formatter.colorize(text, role)
        if source_ref is not None and self._link_resolver is not None:
            url = self._link_resolver.resolve(source_ref)
            if url is not None:
                tooltip = _docstring_for_source_ref(source_ref)
                return self._formatter.wrap_link(colored, url, tooltip=tooltip)
        return colored


@dataclass
class ParagraphRenderer(FragmentRenderer):
    """
    Flattens the entire fragment tree into a single prose string.

    :class:`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment`
    headers and items are joined with the formatter's space character; nesting
    adds no visual structure — only content is preserved.

    This is the default renderer used by
    :func:`~krrood.entity_query_language.verbalization.verbalizer.verbalize_expression`
    and :meth:`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline.plain`.
    """

    def render(self, fragment: VerbFragment) -> str:
        """
        Render *fragment* and all descendants into a flat prose string.

        :param fragment: Root of the fragment tree.
        :type fragment: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        :return: Plain or coloured prose string (no newlines or bullets).
        :rtype: str
        """

        def _block(block: BlockFragment) -> str:
            prose = ", ".join(self.render(i) for i in block.items)
            if block.header is None:
                return prose
            header_str = self.render(block.header)
            return (
                f"{header_str}{self._formatter.space}{prose}" if prose else header_str
            )

        return fold_fragment(
            fragment,
            word=lambda text: text,
            role=lambda text, role, ref: self._render_role(text, role, ref),
            phrase=lambda parts, separator: separator.join(parts),
            block=_block,
        )


@dataclass
class HierarchicalRenderer(FragmentRenderer):
    """
    Renders :class:`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment`
    trees as indented bullet lists.

    Each level of :class:`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment`
    nesting adds one :attr:`indent_size` step.  Non-block fragments are rendered
    inline using :meth:`_inline`.

    Example output (plain)::

        If:
          - there's a Handle
          - there's a PrismaticConnection, whose child is …
        Then:
          - there's a Drawer
            - whose container is …
    """

    indent_size: IndentSize = field(default=IndentSize.TWO_SPACES)
    """Indentation width per nesting level."""

    bullet: BulletStyle = field(default=BulletStyle.DASH)
    """Bullet character prepended to each list item."""

    def render(self, fragment: VerbFragment, depth: int = 0) -> str:
        """
        Render *fragment* with indented bullet structure.

        :param fragment: Root of the fragment tree.
        :type fragment: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment
        :param depth: Current indentation depth (incremented for each BlockFragment level).
        :type depth: int
        :return: Multi-line string with bullets and indentation.
        :rtype: str
        """
        match fragment:
            case BlockFragment(header=header, items=items):
                lines: list[str] = []
                if header is not None:
                    lines.append(self.formatted_indent * depth + self._inline(header))
                    depth = depth + 1
                for item in items:
                    lines.append(self._render_item(item, depth))
                return self._formatter.newline.join(lines)
            case _:
                return self.formatted_indent * depth + self._inline(fragment)

    @property
    def formatted_indent(self) -> str:
        """The indentation string, with spaces replaced by the formatter's space character."""
        return self.indent_size.value.replace(" ", self._formatter.space)

    def _render_item(self, fragment: VerbFragment, depth: int) -> str:
        """Render one item, prepending the bullet at its indentation level."""
        match fragment:
            case BlockFragment():
                return self.render(fragment, depth)
            case _:
                prefix = (
                    self.formatted_indent * depth
                    + self.bullet.value
                    + self._formatter.space
                )
                return prefix + self._inline(fragment)

    def _inline(self, fragment: VerbFragment) -> str:
        """Render a non-block fragment as a flat inline string.

        Expressed as a :func:`~krrood.entity_query_language.verbalization.fragments.base.fold_fragment`;
        a nested block falls back to the depth-aware :meth:`render` so its indented
        layout is preserved.
        """
        return fold_fragment(
            fragment,
            word=lambda text: text,
            role=lambda text, role, ref: self._render_role(text, role, ref),
            phrase=lambda parts, separator: separator.join(parts),
            block=lambda block: self.render(block, 0),
        )
