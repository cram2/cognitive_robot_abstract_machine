"""
The grammar is first-class data: the loaded rules are a plain list of
:class:`PhraseRule` values, so it can be introspected with EQL itself — the "queryable
grammar" property of the redesign.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.grammar.framework.registry import (
    PhraseRuleRegistry,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
)


def test_registry_is_a_list_of_phrase_rules():
    """
    The loaded rules are plain :class:`PhraseRule` instances.
    """
    rules = PhraseRuleRegistry.loaded_rules()
    assert rules
    assert all(isinstance(rule, PhraseRule) for rule in rules)


def test_grammar_is_queryable_with_eql_by_construct():
    """
    An entity query over the loaded rules finds the one rule of a construct.
    """
    rule = variable(PhraseRule, domain=PhraseRuleRegistry.loaded_rules())
    matches = list(an(entity(rule).where(rule.construct == Comparator)).evaluate())
    assert len(matches) == 1
    assert matches[0].construct is Comparator
