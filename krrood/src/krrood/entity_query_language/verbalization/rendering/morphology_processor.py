from __future__ import annotations

from dataclasses import replace

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    RoleFragment,
    VerbalizationFragment,
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
    Realise grammatical agreement and polarity on every tagged leaf — the single
    realisation pass that turns feature decisions into surface words.

    Assemblers and lexicon frames tag a leaf with grammatical ``number`` and ``negated`` features
    (decisions); this pass walks the finished fragment tree and realises them:

    * a noun (any role-tagged or role-less content leaf) tagged plural → pluralised text;
    * a copula (an ``OPERATOR`` leaf, *"is"*) → its negative (*"is not"*) and/or plural (*"are"*)
      suppletion;
    * a verb (a ``VERB`` leaf carrying a lemma) → present-tense agreement (*"works"* / *"work"*) or
      do-support negation (*"does not work"* / *"do not work"*).

    Reference: :cite:t:`gatt2009simplenlg` — the MorphologyProcessor realisation stage.
    """

    def rewrite(self, leaf: VerbalizationFragment) -> VerbalizationFragment:
        """:return: *leaf* with number and polarity realised, then reset; a leaf with no realisable
        feature is returned unchanged.

        >>> MorphologyProcessor().rewrite(WordFragment(text="Robot", number=GrammaticalNumber.PLURAL)).text
        'Robots'
        >>> MorphologyProcessor().rewrite(WordFragment(text="Robot")).text
        'Robot'
        """
        if not isinstance(leaf, (WordFragment, RoleFragment)):
            return leaf
        if isinstance(leaf, RoleFragment) and leaf.role is SemanticRole.VERB:
            return self._realize_verb(leaf)
        if isinstance(leaf, RoleFragment) and leaf.role is SemanticRole.OPERATOR:
            return self._realize_copula(leaf)
        if leaf.number is not GrammaticalNumber.PLURAL:
            return leaf
        return replace(
            leaf,
            text=morphology.ensure_plural(leaf.text),
            number=GrammaticalNumber.SINGULAR,
        )

    @staticmethod
    def _realize_verb(leaf: RoleFragment) -> RoleFragment:
        """:return: the ``VERB`` leaf's lemma realised — present 3rd-person singular (*"works"*),
        bare plural (*"work"*), or do-support negation (*"does not work"* / *"do not work"*).

        >>> MorphologyProcessor()._realize_verb(RoleFragment(text="work", role=SemanticRole.VERB)).text
        'works'
        """
        plural = leaf.number is GrammaticalNumber.PLURAL
        if leaf.negated:
            auxiliary = "do not" if plural else "does not"
            text = f"{auxiliary} {leaf.text}"
        else:
            text = leaf.text if plural else morphology.third_person_singular(leaf.text)
        return replace(
            leaf, text=text, number=GrammaticalNumber.SINGULAR, negated=False
        )

    @staticmethod
    def _realize_copula(leaf: RoleFragment) -> RoleFragment:
        """:return: an ``OPERATOR`` leaf with copula negation then plural suppletion applied; a
        non-copula operator (*"greater than"*) is returned unchanged but for its reset features.

        >>> MorphologyProcessor()._realize_copula(
        ...     RoleFragment(text="is", role=SemanticRole.OPERATOR, negated=True)
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
