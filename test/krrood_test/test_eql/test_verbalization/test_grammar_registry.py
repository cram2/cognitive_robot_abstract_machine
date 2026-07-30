"""
The ``RULES`` registry auto-discovers every concrete grammar rule by walking the grammar
package, rather than hand-maintaining a list of imports — so adding a construct's
``rules`` module needs no edit to the registry.

``concrete_subclasses`` is the single subclass-discovery primitive shared by the
registry and the :class:`SpecificityRule` families.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pytest

from krrood.entity_query_language.verbalization import grammar as grammar_package
from krrood.entity_query_language.verbalization.exceptions import AmbiguousRuleError
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    PhraseRule,
    select,
)
from krrood.entity_query_language.verbalization.grammar.framework.registry import RULES
from krrood.patterns.specificity_ranking import (
    concrete_subclasses,
    maxima,
    sole_maximum,
)


def _discovered_grammar_rule_types():
    """
    Concrete ``PhraseRule`` subclasses defined within the grammar package's own rule
    modules.

    Excludes any subclass defined elsewhere (e.g. by unrelated tests exercising
    ``PhraseRule``'s dispatch contract with throwaway subclasses) so completeness checks
    against ``RULES`` stay independent of those subclasses' unrelated object lifetimes.
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
    The registry holds exactly one instance of every concrete ``PhraseRule`` subclass —
    proving auto-discovery is complete and free of duplicates.
    """
    discovered = {type(rule) for rule in RULES}
    assert discovered == _discovered_grammar_rule_types()
    assert len(RULES) == len(discovered)  # no duplicate instances


def test_registry_completeness_check_ignores_phrase_rule_subclasses_defined_outside_grammar():
    """
    A ``PhraseRule`` subclass defined outside the grammar package must not affect the
    registry's completeness check.

    Regression: ``concrete_subclasses(PhraseRule)`` reflects on every live subclass in the whole
    process via ``__subclasses__()``, including throwaway subclasses other tests define to exercise
    ``PhraseRule``'s dispatch contract (e.g. ``test_grammar_dispatch.py``'s ``ScopedRule``). Comparing
    ``RULES`` against the raw, unscoped set made the completeness check fail depending on whether
    such a subclass happened to still be alive when it ran — a flake tied to unrelated
    garbage-collection timing rather than any real registry defect.
    """

    class TransientNonGrammarRule(PhraseRule):
        construct = object

        def build(self, node, context):
            raise NotImplementedError

    assert TransientNonGrammarRule in concrete_subclasses(PhraseRule)
    assert TransientNonGrammarRule not in _discovered_grammar_rule_types()


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

        def build(self, node, context):
            raise NotImplementedError

    class SecondRule(PhraseRule):
        construct = Node

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
    modules = {type(rule).__module__.rsplit(".", 2)[-2] for rule in RULES}
    for construct_folder in (
        "terms",
        "chain",
        "conditions",
        "query",
        "inference",
        "aggregation",
        "instantiated",
    ):
        assert (
            construct_folder in modules
        ), f"no rules discovered for grammar/{construct_folder}"
