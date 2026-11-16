"""
A method call in a navigation chain reads as the method, not as an empty call.

A call adds nothing a reader can name — ``body.has_collision()`` says what
``body.has_collision`` says — so the call contributes no hop of its own, and a method
that returns a ``bool`` reaches the predicative form a ``bool`` field reaches.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.core.expression_structure import (
    boolean_terminal_attribute,
    walk_chain,
)
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.verbalization.attribute_predicates import (
    default_boolean_predicate,
)
from krrood.entity_query_language.verbalization.boolean_predicate import (
    AdjectivalPredicate,
    PossessivePredicate,
)
from krrood.entity_query_language.verbalization.navigation_path import build_path_parts
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression


@dataclass(kw_only=True)
class Container:
    """
    A thing whose geometry and capacity are computed by methods rather than stored in
    fields, so a query over it navigates through calls.
    """

    name: str
    """
    The container's name.
    """

    def has_collision(self) -> bool:
        """
        :return: Whether the container has collision geometry.
        """
        return True

    def is_sealed(self) -> bool:
        """
        :return: Whether the container is sealed.
        """
        return False

    def capacity(self) -> int:
        """
        :return: How much the container holds.
        """
        return 1


# %% the hops a call contributes


def test_a_call_names_no_hop_of_its_own():
    """
    The attribute the call invokes already names it, so the path is the method name
    alone.
    """
    chain, _ = walk_chain(variable(Container, []).capacity())

    assert [step.name for step in build_path_parts(chain)] == ["capacity"]


def test_a_value_returning_method_reads_as_the_value_it_names():
    """
    With no hop for the call, a value method reads as the possessive path over its own
    name.
    """
    assert (
        verbalize_expression(variable(Container, []).capacity())
        == "the capacity of a Container"
    )


# %% a method that returns a boolean


def test_a_boolean_method_is_the_chains_boolean_terminal():
    """
    The call resolves to ``bool`` from the method's own return annotation, so the chain
    ends in the attribute naming the method.
    """
    chain, _ = walk_chain(variable(Container, []).has_collision())

    terminal = boolean_terminal_attribute(chain)

    assert terminal is not None
    assert terminal._attribute_name_ == "has_collision"


def test_a_boolean_method_reads_predicatively():
    """
    A ``bool``-returning method reaches the predicative form a ``bool`` field reaches.
    """
    assert (
        verbalize_expression(variable(Container, []).has_collision())
        == "a Container has collision"
    )


def test_a_value_returning_method_is_no_boolean_terminal():
    """
    Only the ``bool`` return puts a call on the predicative path.
    """
    chain, _ = walk_chain(variable(Container, []).capacity())

    assert boolean_terminal_attribute(chain) is None


# %% a name that already spells its predicate's head


def test_a_possessive_name_does_not_repeat_its_verb():
    """
    ``has_collision`` says *have* itself, so the predicate takes the rest of the name as
    its object rather than saying *have* twice.
    """
    assert default_boolean_predicate("has_collision") == PossessivePredicate(
        noun="collision"
    )


def test_a_copular_name_does_not_repeat_its_copula():
    """
    ``is_sealed`` says *is* itself, so the predicate takes the rest of the name as its
    adjective.
    """
    assert default_boolean_predicate("is_sealed") == AdjectivalPredicate(
        adjective="sealed"
    )


def test_a_name_spelling_no_head_is_read_by_its_shape():
    """
    A name that spells no head of its own is still classified by the shape of its last
    word.
    """
    assert default_boolean_predicate("milk") == PossessivePredicate()
    assert default_boolean_predicate("completed") == AdjectivalPredicate()
