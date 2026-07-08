from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.verbalization.fragments.base import (
    map_structural_children,
    PhraseFragment,
    VerbalizationFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Spacing,
    Separator,
)
from krrood.entity_query_language.verbalization.rendering.passes import RealizationPass


class OrthographyProcessor(RealizationPass):
    """
    Remove the space adjacent to glued punctuation in every ``PhraseFragment`` (idempotent).

    Rules emit punctuation as ordinary, normally-separated tokens carrying a spacing feature (``,``
    / ``)`` hug the preceding token; ``(`` hugs the following one). This pass walks each phrase
    and regroups its parts so a glued token has no adjacent separator, yielding *"(x)"* from
    ``[OPEN_PAREN, x, CLOSE_PAREN]``.

    Reference: :cite:t:`reiter2000building` — linguistic realisation (orthography); :cite:t:`gatt2009simplenlg`,
    SimpleNLG — the realisation passes.
    """

    def process(self, fragment: VerbalizationFragment) -> VerbalizationFragment:
        """
        :param fragment: Root of the fragment tree.
        :return: A new tree with punctuation spacing fixed.

        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> parens = PhraseFragment(parts=[WordFragment(text="(", spacing=Spacing.RIGHT),
        ...                                WordFragment(text="x"),
        ...                                WordFragment(text=")", spacing=Spacing.LEFT)])
        >>> flatten_fragment_to_plain_text(parens)
        '( x )'
        >>> flatten_fragment_to_plain_text(OrthographyProcessor().process(parens))
        '(x)'
        """
        rebuilt = map_structural_children(fragment, self.process)
        node = rebuilt if rebuilt is not None else fragment
        if isinstance(node, PhraseFragment):
            return PhraseFragment(
                parts=self._apply_glue(node.parts), separator=node.separator
            )
        return node

    def _apply_glue(
        self, parts: List[VerbalizationFragment]
    ) -> List[VerbalizationFragment]:
        """Each merge is a zero-separator subgroup, so the surrounding separator is dropped.

        :param parts: The phrase's parts.
        :return: *parts* regrouped so a ``LEFT`` token hugs the previous part and a ``RIGHT``
            token the next.

        >>> glued = OrthographyProcessor()._apply_glue(
        ...     [WordFragment(text="x"), WordFragment(text=",", spacing=Spacing.LEFT), WordFragment(text="y")])
        >>> len(glued)
        2
        """
        out: List[VerbalizationFragment] = []
        # A RIGHT token (e.g. "(") held until its following part arrives, to attach to it.
        pending_right: List[VerbalizationFragment] = []
        for part in parts:
            spacing = part.spacing if isinstance(part, WordFragment) else Spacing.NONE
            if pending_right:  # attach the held "(" to this part
                part = self._merge(pending_right + [part])
                pending_right = []
            if spacing is Spacing.RIGHT:
                pending_right = [part]
                continue
            if spacing is Spacing.LEFT and out:  # hug the preceding part
                out[-1] = self._merge([out[-1], part])
            else:
                out.append(part)
        # A trailing RIGHT token with no following part (degenerate) stays as-is.
        out.extend(pending_right)
        return out

    @staticmethod
    def _merge(items: List[VerbalizationFragment]) -> VerbalizationFragment:
        """:return: A zero-separator group of *items* (the single item itself when there is only one).

        >>> only = WordFragment(text="x")
        >>> OrthographyProcessor._merge([only]) is only
        True
        >>> OrthographyProcessor._merge([only, WordFragment(text="y")]).separator
        <Separator.NONE: ''>
        """
        return (
            items[0]
            if len(items) == 1
            else PhraseFragment(parts=items, separator=Separator.NONE)
        )
