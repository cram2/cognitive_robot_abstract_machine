from __future__ import annotations

from typing_extensions import List

from krrood.entity_query_language.verbalization.fragments.base import (
    map_structural_children,
    PhraseFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Spacing,
    Separator,
)


class OrthographyProcessor:
    """
    Remove the space adjacent to glued punctuation in every ``PhraseFragment`` (idempotent).

    Rules emit punctuation as ordinary, normally-separated tokens carrying a spacing feature (``,``
    / ``)`` hug the preceding token; ``(`` hugs the following one). This pass walks each phrase
    and regroups its parts so a glued token has no adjacent separator, yielding *"(x)"* from
    ``[OPEN_PAREN, x, CLOSE_PAREN]``.

    Reference: Reiter & Dale (2000) — linguistic realisation (orthography); Gatt & Reiter (2009),
    SimpleNLG — the realisation passes.
    """

    def process(self, fragment: Fragment) -> Fragment:
        """
        :param fragment: Root of the fragment tree.
        :return: A new tree with punctuation spacing fixed.
        """
        rebuilt = map_structural_children(fragment, self.process)
        node = rebuilt if rebuilt is not None else fragment
        if isinstance(node, PhraseFragment):
            return PhraseFragment(
                parts=self._apply_glue(node.parts), separator=node.separator
            )
        return node

    def _apply_glue(self, parts: List[Fragment]) -> List[Fragment]:
        """Each merge is a zero-separator subgroup, so the surrounding separator is dropped.

        :param parts: The phrase's parts.
        :return: *parts* regrouped so a ``LEFT`` token hugs the previous part and a ``RIGHT``
            token the next.
        """
        out: List[Fragment] = []
        # A RIGHT token (e.g. "(") held until its following part arrives, to attach to it.
        pending_right: List[Fragment] = []
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
    def _merge(items: List[Fragment]) -> Fragment:
        """:return: A zero-separator group of *items* (the single item itself when there is only one)."""
        return (
            items[0]
            if len(items) == 1
            else PhraseFragment(parts=items, separator=Separator.NONE)
        )
