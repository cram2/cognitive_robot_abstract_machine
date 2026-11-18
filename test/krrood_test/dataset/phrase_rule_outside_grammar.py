"""
A grammar rule defined at module level outside the verbalization grammar package, as a
krrood part defines the rules of its own constructs.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    RuleContext,
)

# %% construct and its rule


@dataclass(kw_only=True)
class ConstructOutsideGrammar:
    """
    A construct that no rule of the grammar package covers.
    """


class ConstructOutsideGrammarRule(PhraseRule):
    """
    The rule for :class:`ConstructOutsideGrammar`.
    """

    construct = ConstructOutsideGrammar

    def build(
        self, *, node: ConstructOutsideGrammar, context: RuleContext
    ) -> VerbalizationFragment:
        raise NotImplementedError
