from __future__ import annotations

from dataclasses import replace

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.fragments.base import (
    map_fragment,
    RoleFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Number
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.vocabulary.english import Copulas

#: Plural suppletion for the copula (the only ``OPERATOR`` leaves ever tagged plural).
_COPULA_PLURAL = {
    Copulas.IS.text: Copulas.ARE.text,
    Copulas.IS_NOT.text: Copulas.ARE_NOT.text,
}


class MorphologyProcessor:
    """
    Realise grammatical number on every leaf tagged plural — the single realisation pass that
    applies grammatical agreement.

    Assemblers and lexicon frames tag a leaf with a grammatical number (a decision); this pass
    walks the finished fragment tree and realises every leaf tagged plural:

    * a noun (any role-tagged or role-less content leaf) → pluralised text;
    * a copula (an ``OPERATOR`` leaf, i.e. *"is"* / *"is not"*) → its plural suppletion (*"are"*
      / *"are not"*).

    Reference: Gatt & Reiter (2009), SimpleNLG — the MorphologyProcessor realisation stage.
    """

    def process(self, fragment: Fragment) -> Fragment:
        """
        :param fragment: Root of the fragment tree.
        :return: A new tree with all plural-tagged leaves agreed/inflected (idempotent).
        """
        return map_fragment(fragment, self._inflect)

    @staticmethod
    def _inflect(leaf: Fragment) -> Fragment:
        if not isinstance(leaf, (WordFragment, RoleFragment)):
            return leaf
        if leaf.number is not Number.PLURAL:
            return leaf
        if isinstance(leaf, RoleFragment) and leaf.role is SemanticRole.OPERATOR:
            return replace(
                leaf,
                text=_COPULA_PLURAL.get(leaf.text, leaf.text),
                number=Number.SINGULAR,
            )
        return replace(
            leaf, text=morphology.ensure_plural(leaf.text), number=Number.SINGULAR
        )
