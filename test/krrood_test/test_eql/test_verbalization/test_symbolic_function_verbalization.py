"""
Symbolic-operation class hierarchy: a value :class:`SymbolicFunction` reads as a noun.

:class:`Predicate` (a boolean operation) and :class:`SymbolicFunction` (a value
operation) share the :class:`SymbolicCallable` base, so the symbolic-construction
machinery lives in one place. A ``SymbolicFunction`` subclass names the value it
computes through its own :meth:`Verbalizable._verbalization_fragment_` (a noun phrase) —
the way a ``Predicate`` names its clause — and evaluates to that value through
:meth:`__call__`.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.factories import a, an, entity, set_of, variable
from krrood.entity_query_language.core.bound_value import HasBoundValue
from krrood.entity_query_language.predicate import (
    Length,
    Predicate,
    SymbolicCallable,
    SymbolicFunction,
    length,
)
from krrood.entity_query_language.verbalization.fragments.base import WordFragment
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import Noun


@dataclass(eq=False)
class RemainingLoad(SymbolicFunction):
    """
    A value ``SymbolicFunction`` with a custom noun surface, exercising the class form.
    """

    capacity: int
    """
    The capacity it is computed from.
    """

    load: int
    """
    The load it is computed from.
    """

    def __call__(self) -> int:
        return self.capacity - self.load

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return Noun(WordFragment(text="the remaining load")).as_fragment()


def test_symbolic_function_and_predicate_share_a_base():
    """
    Both are self-verbalizing symbolic callables, so construction lives in one shared
    base.
    """
    assert issubclass(Predicate, SymbolicCallable)
    assert issubclass(SymbolicFunction, SymbolicCallable)


def test_symbolic_function_subclass_uses_its_custom_fragment():
    """
    A ``SymbolicFunction`` subclass names its value through its own noun-phrase
    fragment.
    """
    numbers = variable(int, [])
    assert (
        verbalize_expression(a(set_of(RemainingLoad(numbers, numbers))))
        == "Find the remaining load"
    )


def test_symbolic_function_subclass_evaluates_via_call():
    """
    Its :meth:`__call__` computes the value directly when constructed with concrete
    arguments.
    """
    assert RemainingLoad._construct_normally_(capacity=10, load=3)() == 7


@dataclass(eq=False)
class Doubled(SymbolicFunction):
    """
    A single-argument value ``SymbolicFunction``, exercising query evaluation.
    """

    number: int
    """
    The number it doubles.
    """

    def __call__(self) -> int:
        return self.number * 2

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return Noun(WordFragment(text="the doubled number")).as_fragment()


def test_symbolic_function_binds_its_computed_value_in_a_query():
    """
    In a query a value ``SymbolicFunction`` binds what it COMPUTES (constructed AND
    called), exactly like a ``@symbolic_function`` — not the constructed instance.
    """
    numbers = variable(int, domain=[1, 2, 3])
    values = sorted(an(entity(Doubled(numbers))).tolist())
    assert values == [2, 4, 6]


def test_migrated_length_reads_as_a_noun_phrase_and_keeps_its_value():
    """
    ``length`` is now ``class Length(SymbolicFunction)`` behind a
    ``symbolic_callable_to_function`` wrapper: it reads as the name-based noun phrase
    (*"the length of ..."*) and still computes the length.
    """
    assert issubclass(Length, SymbolicFunction)
    assert length([1, 2, 3]) == 3
    assert (
        verbalize_expression(a(set_of(length(variable(list, [])))))
        == "Find the length of a list"
    )


def test_symbolic_callables_implement_the_bound_value_contract():
    """The evaluator dispatches on the :class:`HasBoundValue` contract rather than probing for the
    method by name: every symbolic callable implements it, while a plain type does not (so it is
    called directly). A value function's bound value is what it COMPUTES, not the constructed
    instance."""
    assert issubclass(SymbolicCallable, HasBoundValue)
    assert issubclass(Predicate, HasBoundValue)
    assert issubclass(SymbolicFunction, HasBoundValue)
    assert not issubclass(int, HasBoundValue)
    assert Length._bound_value_(iterable=[1, 2, 3]) == 3
