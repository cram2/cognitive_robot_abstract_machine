"""
The realisation-pass framework — one shared contract for the ordered lowering passes that
:func:`~…rendering.realization.realize_tree` runs over the fragment tree.

A :class:`RealizationPass` is *fragment in, fragment out*; the pipeline runs a list of them in
order. A :class:`RewritePass` captures the common case — a stateless pass that is fully described by
a single *leaf* rewrite — so determiner lowering and morphology need not each re-walk the tree by
hand: :func:`~…fragments.base.map_fragment` rebuilds it bottom-up around the rewritten leaves.

Reference: :cite:t:`gatt2009simplenlg` — the ordered realisation stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
    map_fragment,
)


class RealizationPass(ABC):
    """One ordered lowering pass over the realised fragment tree."""

    @abstractmethod
    def process(self, fragment: VerbalizationFragment) -> VerbalizationFragment:
        """
        :param fragment: Root of the (partly realised) fragment tree.
        :return: The tree after this pass (a new tree; passes never mutate in place).
        """


class RewritePass(RealizationPass, ABC):
    """
    A stateless pass expressed as a single *leaf* rewrite: :func:`~…fragments.base.map_fragment`
    rebuilds the tree bottom-up, replacing each leaf with :meth:`rewrite` and reconstructing the
    structural containers automatically — so a leaf-local pass (determiner lowering, morphology)
    declares only what to do to a leaf, never how to traverse.
    """

    @abstractmethod
    def rewrite(self, leaf: VerbalizationFragment) -> VerbalizationFragment:
        """:param leaf: A leaf fragment. :return: Its replacement (or *leaf* unchanged)."""

    def process(self, fragment: VerbalizationFragment) -> VerbalizationFragment:
        """:return: *fragment* rebuilt bottom-up with :meth:`rewrite` applied to every leaf.

        >>> from krrood.entity_query_language.verbalization.fragments.base import (
        ...     RoleFragment, PhraseFragment, flatten_fragment_to_plain_text)
        >>> class Shout(RewritePass):
        ...     def rewrite(self, leaf):
        ...         return RoleFragment.for_operator(leaf.text.upper())
        >>> phrase = PhraseFragment(parts=[RoleFragment.for_operator("is"), RoleFragment.for_literal(42)])
        >>> flatten_fragment_to_plain_text(Shout().process(phrase))
        'IS 42'
        """
        return map_fragment(fragment, self.rewrite)
