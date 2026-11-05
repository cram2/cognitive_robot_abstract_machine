"""States that assign values to ground terms without changing live objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import Any

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.ground_term import (
    FunctionKey,
    GroundTerm,
    TermReader,
    TermValue,
)
from krrood.planning.fact import Fact

# %% states


@dataclass(frozen=True)
class State(TermReader):
    """
    A valuation of ground terms that differs from the live objects only where it assigns a
    value.

    A ground term (:class:`~krrood.entity_query_language.ground_term.GroundTerm`) is an
    attribute of an instance, an indexed element, or the result of a call, a symbolic
    function or a predicate for concrete arguments. A query evaluated with this state
    reads the assigned value of every assigned ground term, and the live value of every
    other. The live objects are never changed.

    >>> from dataclasses import dataclass
    >>> from krrood.entity_query_language.factories import a, literal
    >>> @dataclass
    ... class Box:
    ...     color: str
    >>> box = Box("red")
    >>> state = State().with_facts(literal(box).color == "blue")
    >>> a(Box).from_([box]).color.tolist(state=state)
    ['blue']
    >>> box.color
    'red'

    .. note:: An unassigned derived ground term follows the state only where its value comes
        from EQL: a body that is itself an EQL query, or inputs passed to it as EQL
        arguments, as in ``is_table(block.support)``. A Python body that reads an
        attribute directly, such as ``self.block.support``, reads the live value.

    .. warning:: An assigned ground term keeps its value even where it is derived from other
        ground terms, so assigning a predicate overrides what it would compute from the
        ground terms it reads, also after those change.
    """

    assignments: dict[GroundTerm, Any] = field(default_factory=dict)
    """The value of every ground term this state changes.

    Never mutated; a changed state is a new :class:`State`.
    """

    def __hash__(self) -> int:
        return hash(frozenset(self.assignments))

    @cached_property
    def assigned_functions(self) -> frozenset[FunctionKey]:
        """
        :return: The :attr:`~krrood.entity_query_language.ground_term.GroundTerm.function_key`
            of every ground term this state assigns.
        """
        return frozenset(term.function_key for term in self.assignments)

    def with_facts(self, *facts: SymbolicExpression) -> State:
        """
        :param facts: Conditions that each fix the value of ground terms, as
            :meth:`~krrood.planning.fact.Fact.from_condition` reads them. A fact whose
            expression has variables fixes the ground term for every binding of them.
        :return: A new state that assigns the values *facts* state on top of this
            state's own assignments, a later value winning where two assign a ground term.
        """
        assignments = dict(self.assignments)
        for fact in facts:
            assignments |= Fact.from_condition(fact).assignments()
        return State(assignments=assignments)

    def value_of(self, term: GroundTerm) -> TermValue | None:
        if term.function_key in self.assigned_functions and term in self.assignments:
            return TermValue(self.assignments[term])
        return term.live_term_value()
