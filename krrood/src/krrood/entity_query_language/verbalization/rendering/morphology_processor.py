"""
The realisation pass that turns the grammatical features of leaves into surface words.
"""

from __future__ import annotations

from dataclasses import replace

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    CardinalNumber,
    CollectionElementNoun,
    IndefiniteArticle,
    IndexOrdinal,
    map_fragment,
    RoleFragment,
    VerbalizationFragment,
    VerbFormFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.rendering.passes import RewritePass
from krrood.entity_query_language.verbalization.vocabulary.english import Copulas

#: Plural suppletion for the copula (the only ``OPERATOR`` leaves ever tagged plural).
_COPULA_PLURAL = {
    Copulas.IS.text: Copulas.ARE.text,
    Copulas.IS_NOT.text: Copulas.ARE_NOT.text,
}

#: Negative suppletion for the copula (*"is"* → *"is not"*), realised from the ``negated`` feature.
_COPULA_NEGATIVE = {
    Copulas.IS.text: Copulas.IS_NOT.text,
    Copulas.ARE.text: Copulas.ARE_NOT.text,
}


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
        """:return: *leaf* with number and polarity realised, then reset; a leaf with no realisable
        feature is returned unchanged.

        >>> MorphologyProcessor().rewrite(leaf=WordFragment(text="Robot", number=GrammaticalNumber.PLURAL)).text
        'Robots'
        >>> MorphologyProcessor().rewrite(leaf=WordFragment(text="Robot")).text
        'Robot'
        >>> MorphologyProcessor().rewrite(leaf=CardinalNumber(count=7)).text
        'seven'
        """
        if isinstance(leaf, CardinalNumber):
            return WordFragment(text=morphology.cardinal(leaf.count))
        if isinstance(leaf, IndexOrdinal):
            return WordFragment(text=morphology.index_ordinal(leaf.index))
        if not isinstance(leaf, (WordFragment, RoleFragment)):
            return leaf
        if isinstance(leaf, CollectionElementNoun):
            leaf = replace(leaf, text=morphology.singular(leaf.text))
        if isinstance(leaf, RoleFragment) and leaf.role is SemanticRole.VERB:
            return self._realize_verb(leaf=leaf)
        if isinstance(leaf, RoleFragment) and leaf.role is SemanticRole.OPERATOR:
            return self._realize_copula(leaf=leaf)
        if leaf.number is not GrammaticalNumber.PLURAL:
            return leaf
        return replace(
            leaf,
            text=morphology.ensure_plural(leaf.text),
            number=GrammaticalNumber.SINGULAR,
        )

    @staticmethod
    def _realize_verb(*, leaf: RoleFragment) -> RoleFragment:
        """:return: the ``VERB`` leaf's lemma realised — present 3rd-person singular (*"works"*),
        bare plural (*"work"*), or do-support negation (*"does not work"* / *"do not work"*).

        >>> MorphologyProcessor()._realize_verb(leaf=RoleFragment(text="work", role=SemanticRole.VERB)).text
        'works'
        >>> MorphologyProcessor()._realize_verb(leaf=VerbFormFragment(text="connects", negated=True)).text
        'does not connect'
        """
        plural = leaf.number is GrammaticalNumber.PLURAL
        lemma = (
            morphology.verb_lemma(leaf.text)
            if isinstance(leaf, VerbFormFragment)
            else leaf.text
        )
        if leaf.negated:
            auxiliary = "do not" if plural else "does not"
            text = f"{auxiliary} {lemma}"
        else:
            text = lemma if plural else morphology.third_person_singular(lemma)
        return replace(
            leaf, text=text, number=GrammaticalNumber.SINGULAR, negated=False
        )

    @staticmethod
    def _realize_copula(*, leaf: RoleFragment) -> RoleFragment:
        """:return: an ``OPERATOR`` leaf with copula negation then plural suppletion applied; a
        non-copula operator (*"greater than"*) is returned unchanged but for its reset features.

        >>> MorphologyProcessor()._realize_copula(
        ...     leaf=RoleFragment(text="is", role=SemanticRole.OPERATOR, negated=True)
        ... ).text
        'is not'
        """
        text = leaf.text
        if leaf.negated:
            text = _COPULA_NEGATIVE.get(text, text)
        if leaf.number is GrammaticalNumber.PLURAL:
            text = _COPULA_PLURAL.get(text, text)
        return replace(
            leaf, text=text, number=GrammaticalNumber.SINGULAR, negated=False
        )
