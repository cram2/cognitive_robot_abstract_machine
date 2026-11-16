"""
The registry discovers every concrete grammar rule of the grammar package's ``rules``
modules, and the rules a module outside it defines once that module is loaded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from ...dataset.phrase_rule_outside_grammar import ConstructOutsideGrammarRule

from krrood.entity_query_language.verbalization import grammar as grammar_package
from krrood.patterns.exceptions import AmbiguousRuleError
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    select,
)
from krrood.entity_query_language.verbalization.grammar.framework.registry import (
    PhraseRuleRegistry,
)
from krrood.patterns.specificity_ranking import (
    concrete_subclasses,
    maxima,
    sole_maximum,
)


def _discovered_grammar_rule_types():
    """
    Concrete ``PhraseRule`` subclasses defined within the grammar package's own rule
    modules, excluding the throwaway subclasses other tests define.
    """
    return {
        subclass
        for subclass in concrete_subclasses(PhraseRule)
        # grammar_package.__name__ is the package's full dotted import path (e.g.
        # "krrood.entity_query_language.verbalization.grammar"), matching subclass.__module__.
        if subclass.__module__.startswith(f"{grammar_package.__name__}.")
    }


def test_concrete_subclasses_excludes_abstract_bases():
    class Base(ABC):
        @abstractmethod
        def do(self): ...

    class Middle(Base, ABC):  # still abstract — does not implement ``do``
        pass

    class Leaf(Base):
        def do(self):
            return 1

    found = concrete_subclasses(Base)
    assert Leaf in found
    assert Middle not in found and Base not in found


def test_rules_registry_is_exactly_one_instance_per_concrete_rule():
    """
    The registry holds exactly one instance of every concrete ``PhraseRule`` subclass of
    the grammar package — proving auto-discovery is complete and free of duplicates.
    """
    rules = PhraseRuleRegistry.loaded_rules()
    discovered = {type(rule) for rule in rules}
    grammar_rules = [
        rule for rule in rules if type(rule) in _discovered_grammar_rule_types()
    ]
    assert {type(rule) for rule in grammar_rules} == _discovered_grammar_rule_types()
    assert len(rules) == len(discovered)  # no duplicate instances


def test_registry_includes_a_module_level_rule_defined_outside_the_grammar_package():
    """
    A rule a loaded module outside the grammar package defines joins the grammar.
    """
    assert ConstructOutsideGrammarRule in {
        type(rule) for rule in PhraseRuleRegistry.loaded_rules()
    }


def test_registry_excludes_a_rule_that_opts_out_of_the_grammar():
    """
    A ``PhraseRule`` subclass that sets ``joins_grammar`` to ``False``, as tests do for
    the rules they define to exercise the dispatch contract, is neither part of the
    grammar package's rules nor loaded into the grammar, however long it stays alive.
    """

    class TransientNonGrammarRule(PhraseRule):
        construct = object
        joins_grammar = False

        def build(self, node, context):
            raise NotImplementedError

    assert TransientNonGrammarRule in concrete_subclasses(PhraseRule)
    assert TransientNonGrammarRule not in _discovered_grammar_rule_types()
    assert TransientNonGrammarRule not in {
        type(rule) for rule in PhraseRuleRegistry.loaded_rules()
    }


def test_registry_excludes_the_subclasses_of_a_rule_that_opts_out():
    """
    A rule inherits the opt-out of its base, so a test base rule covers the rules built
    on it.
    """

    class OptedOutBase(PhraseRule, ABC):
        joins_grammar = False

    class InheritingRule(OptedOutBase):
        construct = object

        def build(self, node, context):
            raise NotImplementedError

    assert InheritingRule not in {
        type(rule) for rule in PhraseRuleRegistry.loaded_rules()
    }


def test_registry_includes_a_rule_that_does_not_opt_out():
    """
    A concrete ``PhraseRule`` subclass joins the grammar unless it opts out.
    """

    class UnrelatedConstruct:
        pass

    class DefaultRule(PhraseRule):
        construct = UnrelatedConstruct

        def build(self, node, context):
            raise NotImplementedError

    assert DefaultRule in {type(rule) for rule in PhraseRuleRegistry.loaded_rules()}


def test_maxima_returns_all_co_maximal_candidates():
    """
    A single winner yields a one-element list; equal-key candidates all come back (a
    tie).
    """
    assert maxima(["a", "abc", "ab"], key=len) == ["abc"]
    assert maxima(["ab", "cd", "a"], key=len) == ["ab", "cd"]
    assert maxima([], key=len) == []


def test_sole_maximum_raises_supplied_error_on_tie():
    """
    A tie raises the injected collision error; a unique maximum is returned as-is.
    """
    assert sole_maximum(["a", "abc"], key=len, collision_error=AssertionError) == "abc"
    with pytest.raises(ValueError):
        sole_maximum(["ab", "cd"], key=len, collision_error=ValueError)


def test_select_raises_on_equally_specific_rule_collision():
    """
    Two unguarded rules on the same construct are equally specific — selecting between
    them is a collision, raised rather than resolved silently by registration order.
    """

    class Node:
        pass

    class FirstRule(PhraseRule):
        construct = Node
        joins_grammar = False

        def build(self, node, context):
            raise NotImplementedError

    class SecondRule(PhraseRule):
        construct = Node
        joins_grammar = False

        def build(self, node, context):
            raise NotImplementedError

    with pytest.raises(AmbiguousRuleError) as collision:
        select(Node(), [FirstRule(), SecondRule()], context=None)
    assert "FirstRule" in str(collision.value) and "SecondRule" in str(collision.value)


def test_every_construct_folder_contributes_rules():
    """
    A representative rule from each construct package is auto-discovered (the walk
    reaches every ``rules`` module, not a hand-listed subset).
    """
    modules = {
        type(rule).__module__.rsplit(".", 2)[-2]
        for rule in PhraseRuleRegistry.loaded_rules()
    }
    for construct_folder in (
        "terms",
        "chain",
        "conditions",
        "query",
        "inference",
        "aggregation",
        "instantiated",
    ):
        assert construct_folder in modules, (
            f"no rules discovered for grammar/{construct_folder}"
        )
