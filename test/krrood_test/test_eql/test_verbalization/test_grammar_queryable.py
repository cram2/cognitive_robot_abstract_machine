"""
The grammar is first-class data: ``RULES`` is a plain list of
:class:`PhraseRule` values, so it can be introspected with EQL itself — the
"queryable grammar" property of the redesign.
"""

from __future__ import annotations

from krrood.entity_query_language.factories import an, entity, variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.grammar.english import RULES
from krrood.entity_query_language.verbalization.grammar.phrase_rule import PhraseRule


def test_registry_is_a_list_of_phrase_rules():
    assert RULES
    assert all(isinstance(rule, PhraseRule) for rule in RULES)


def test_grammar_is_queryable_with_eql_by_construct():
    rule = variable(PhraseRule, domain=RULES)
    matches = list(an(entity(rule).where(rule.construct == Comparator)).evaluate())
    assert [r.name for r in matches] == ["comparator"]


def test_grammar_is_queryable_with_eql_by_name():
    rule = variable(PhraseRule, domain=RULES)
    matches = list(an(entity(rule).where(rule.name == "comparator")).evaluate())
    assert len(matches) == 1
    assert matches[0].construct is Comparator
