from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    fold_fragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.rendering.formatter import (
    BulletStyle,
    Formatter,
    IndentSize,
    PlainFormatter,
)
from krrood.entity_query_language.verbalization.rendering.source_documentation import (
    docstring_for_source_ref,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.rendering.source_link_resolver import (
        SourceLinkResolver,
    )


@dataclass
class FragmentRenderer(ABC):
    """
    Abstract base that converts a fragment tree into a string.

    Subclasses differ in how they handle block nesting: paragraph rendering flattens into one
    prose string; hierarchical rendering renders blocks as indented bullet lists.
    """

    formatter: Formatter = field(default_factory=PlainFormatter)
    """Format-specific markup logic (plain, ANSI, HTML)."""

    link_resolver: Optional[SourceLinkResolver] = field(default=None)
    """Optional resolver that maps source references to URL strings."""

    @abstractmethod
    def render(self, fragment: Fragment) -> str:
        """
        Render a fragment tree into a string.

        :param fragment: Root of the fragment tree to render.
        :return: Formatted string representation.
        """
        ...

    def _render_role(self, text: str, role, source_reference) -> str:
        """
        Colorise *text* for *role* and, when a resolver and source reference are present, wrap the
        result with a hyperlink.

        :param text: Plain display text.
        :param role: Semantic role for colour lookup.
        :param source_reference: Source reference for link resolution; may be ``None``.
        :return: Coloured (and optionally linked) string.

        >>> from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
        >>> ParagraphRenderer()._render_role("Robot", SemanticRole.VARIABLE, None)
        'Robot'
        """
        colored = self.formatter.colorize(text, role)
        if source_reference is not None and self.link_resolver is not None:
            url = self.link_resolver.resolve(source_reference)
            if url is not None:
                tooltip = docstring_for_source_ref(source_reference)
                return self.formatter.wrap_link(colored, url, tooltip=tooltip)
        return colored


@dataclass
class ParagraphRenderer(FragmentRenderer):
    """
    Flattens the entire fragment tree into a single prose string.

    Block headers and items are joined with the formatter's space character; nesting adds no
    visual structure — only content is preserved.
    """

    def render(self, fragment: Fragment) -> str:
        """
        Render *fragment* and all descendants into a flat prose string.

        :param fragment: Root of the fragment tree.
        :return: Plain or coloured prose string (no newlines or bullets).

        >>> from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
        >>> robot = variable(Robot, [])
        >>> tree = EQLVerbalizer().build(a(entity(robot).where(and_(robot.battery > 50, robot.battery < 90))))
        >>> ParagraphRenderer().render(tree)
        'Find a Robot whose battery is between 50 and 90'
        """

        def _block(block: BlockFragment) -> str:
            rendered = [self.render(item) for item in block.items]
            if block.conjunction is not None and len(rendered) > 1:
                # Oxford coordination: "a, b, and c" (the comma before the conjunction is the
                # ", "-join itself, matching the project's house style, incl. "a, and b").
                conjunction = self.render(block.conjunction)
                rendered[-1] = f"{conjunction}{self.formatter.space}{rendered[-1]}"
            prose = ", ".join(rendered)
            if block.header is None:
                return prose
            header_str = self.render(block.header)
            return f"{header_str}{self.formatter.space}{prose}" if prose else header_str

        return fold_fragment(
            fragment,
            word=lambda text: text,
            role=lambda text, role, reference: self._render_role(text, role, reference),
            phrase=lambda parts, separator: separator.join(parts),
            block=_block,
        )


@dataclass
class HierarchicalRenderer(FragmentRenderer):
    """
    Renders block trees as indented bullet lists.

    Each level of block nesting adds one indent step; non-block fragments are rendered inline.

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

    def render(self, fragment: Fragment, depth: int = 0) -> str:
        """
        Render *fragment* with indented bullet structure.

        :param fragment: Root of the fragment tree.
        :param depth: Current indentation depth (incremented for each block level).
        :return: Multi-line string with bullets and indentation.

        >>> from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
        >>> robot = variable(Robot, [])
        >>> tree = EQLVerbalizer().build(a(entity(robot).where(and_(robot.battery > 50, robot.battery < 90))))
        >>> print(HierarchicalRenderer().render(tree))
        Find a Robot
          whose
            - battery is between 50 and 90
        """
        match fragment:
            case BlockFragment(header=header, items=items):
                lines: list[str] = []
                if header is not None:
                    bullet = (
                        self.bullet.value + self.formatter.space
                        if fragment.bulleted_header
                        else ""
                    )
                    lines.append(
                        self.formatted_indent * depth + bullet + self._inline(header)
                    )
                    depth = depth + 1
                last = len(items) - 1
                for index, item in enumerate(items):
                    # The coordinating conjunction (when any) joins the last item — "… and c".
                    conjunction = (
                        fragment.conjunction
                        if fragment.conjunction is not None
                        and index == last
                        and last > 0
                        else None
                    )
                    lines.append(self._render_item(item, depth, conjunction))
                return self.formatter.newline.join(lines)
            case _:
                return self.formatted_indent * depth + self._inline(fragment)

    @property
    def formatted_indent(self) -> str:
        """:return: The indentation string, with spaces replaced by the formatter's space character.

        >>> HierarchicalRenderer().formatted_indent
        '  '
        """
        return self.indent_size.value.replace(" ", self.formatter.space)

    def _render_item(
        self, fragment: Fragment, depth: int, conjunction: Optional[Fragment] = None
    ) -> str:
        """Render one item, prepending the bullet at its indentation level (and a coordinating
        conjunction when this is the last item of a coordinated block).

        Its contribution is the per-item dispatch: a *block* item recurses into :meth:`render` (no
        bullet of its own), a *leaf* item gets the ``- `` prefix. The output matches :meth:`render`
        because the top item here is the whole block (recursed), while the bullet it adds is visible
        on the leaf line *"- battery is between 50 and 90"*:

        >>> from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
        >>> robot = variable(Robot, [])
        >>> tree = EQLVerbalizer().build(a(entity(robot).where(and_(robot.battery > 50, robot.battery < 90))))
        >>> print(HierarchicalRenderer()._render_item(tree, 0))
        Find a Robot
          whose
            - battery is between 50 and 90
        """
        match fragment:
            case BlockFragment():
                return self.render(fragment, depth)
            case _:
                prefix = (
                    self.formatted_indent * depth
                    + self.bullet.value
                    + self.formatter.space
                )
                content = self._inline(fragment)
                if conjunction is not None:
                    content = (
                        f"{self._inline(conjunction)}{self.formatter.space}{content}"
                    )
                return prefix + content

    def _inline(self, fragment: Fragment) -> str:
        """Render a fragment as a flat inline string. A nested block is flattened to prose (its
        items coordinated as in paragraph rendering) rather than expanded into bullets — bullets are
        for a block that sits at the item level, not one embedded in a phrase (e.g. a *"whose …"*
        modifier inside an inline noun phrase).

        >>> from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
        >>> robot = variable(Robot, [])
        >>> tree = EQLVerbalizer().build(a(entity(robot).where(and_(robot.battery > 50, robot.battery < 90))))
        >>> HierarchicalRenderer()._inline(tree)
        'Find a Robot whose battery is between 50 and 90'
        """

        def _flatten(block: BlockFragment) -> str:
            rendered = [self._inline(item) for item in block.items]
            if block.conjunction is not None and len(rendered) > 1:
                conjunction = self._inline(block.conjunction)
                rendered[-1] = f"{conjunction}{self.formatter.space}{rendered[-1]}"
            prose = ", ".join(rendered)
            if block.header is None:
                return prose
            header_str = self._inline(block.header)
            return f"{header_str}{self.formatter.space}{prose}" if prose else header_str

        return fold_fragment(
            fragment,
            word=lambda text: text,
            role=lambda text, role, reference: self._render_role(text, role, reference),
            phrase=lambda parts, separator: separator.join(parts),
            block=_flatten,
        )
