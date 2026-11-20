"""
The realisation pass that turns the grammatical features of leaves into surface words.
"""

from __future__ import annotations

from dataclasses import replace

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    CardinalNumber,
    IndefiniteArticle,
    IndexOrdinal,
    map_fragment,
    RoleFragment,
    VerbalizationFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.rendering.leaf_realization import (
    LeafRealization,
)
from krrood.entity_query_language.verbalization.rendering.passes import RewritePass

# %% the morphology pass


class MorphologyProcessor(RewritePass):
    """
    Realise the features a clause was built with as surface words: number, polarity and
    verb agreement, the singular of a collection-element noun, count and position words,
    and the *"a"* or *"an"* form of an indefinite article.

    Reference: :cite:t:`gatt2009simplenlg` — the MorphologyProcessor realisation stage.
    """

    def process(self, *, fragment: VerbalizationFragment) -> VerbalizationFragment:
        """
        :return: *fragment* with every leaf realised by :meth:`rewrite`, then every indefinite
            article agreeing with the realised word that follows it.

        >>> from krrood.entity_query_language.verbalization.fragments.base import PhraseFragment, flatten_fragment_to_plain_text
        >>> phrase = PhraseFragment(parts=[IndefiniteArticle(text="a"), IndexOrdinal(index=7), WordFragment(text="Robot")])
        >>> flatten_fragment_to_plain_text(MorphologyProcessor().process(fragment=phrase))
        'an eighth Robot'
        """
        realised = super().process(fragment=fragment)
        leaves: list[VerbalizationFragment] = []
        map_fragment(realised, lambda leaf: leaves.append(leaf) or leaf)
        agreed = iter(self._agree_indefinite_articles(leaves=leaves))
        return map_fragment(realised, lambda leaf: next(agreed))

    @staticmethod
    def _agree_indefinite_articles(
        *, leaves: list[VerbalizationFragment]
    ) -> list[VerbalizationFragment]:
        """
        :param leaves: The realised leaves, in reading order.
        :return: *leaves* with each indefinite article realised as *"a"* or *"an"* for the next
            leaf that spells a word; an article with no such leaf after it is left as it is.
        """
        agreed = list(leaves)
        worded = [
            position
            for position, leaf in enumerate(leaves)
            if isinstance(leaf, (WordFragment, RoleFragment))
            and any(character.isalnum() for character in leaf.text)
        ]
        for position, following in zip(worded, worded[1:]):
            article = leaves[position]
            if isinstance(article, IndefiniteArticle):
                agreed[position] = replace(
                    article,
                    text=morphology.indefinite_article(leaves[following].text),
                )
        return agreed

    def rewrite(self, *, leaf: VerbalizationFragment) -> VerbalizationFragment:
        """:return: *leaf* realized by the most specific :class:`LeafRealization` that applies to
        it; a leaf none applies to is returned unchanged.

        >>> MorphologyProcessor().rewrite(leaf=WordFragment(text="Robot", number=GrammaticalNumber.PLURAL)).text
        'Robots'
        >>> MorphologyProcessor().rewrite(leaf=WordFragment(text="Robot")).text
        'Robot'
        >>> MorphologyProcessor().rewrite(leaf=CardinalNumber(count=7)).text
        'seven'
        """
        realization = LeafRealization.most_applicable(leaf=leaf)
        if realization is None:
            return leaf
        return realization.realize(leaf=leaf)
