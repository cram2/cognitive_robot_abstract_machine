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

from krrood.entity_query_language.factories import a, set_of, variable
from krrood.entity_query_language.predicate import (
    Predicate,
    SymbolicCallable,
    SymbolicFunction,
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
