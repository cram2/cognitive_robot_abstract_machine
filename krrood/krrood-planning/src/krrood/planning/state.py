"""
States that assign values to ground terms, and add and remove objects, without changing
live objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import Any, Iterable, Iterator

from krrood.class_diagrams.collection_kind import CollectionKind
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.ground_term import (
    FunctionKey,
    GroundTerm,
    TermReader,
    TermValue,
)
from krrood.entity_query_language.utils import T
from krrood.planning.effect import Assignment, StateChange

# %% states


@dataclass(kw_only=True, frozen=True)
class State(TermReader):
    """
    A valuation of *fluents*: ground terms whose value a state can change, each with its
    :class:`~krrood.entity_query_language.ground_term.FunctionKey` as fluent symbol. An
    unassigned derived fluent follows the state only where its value comes from EQL, and
    an assigned one keeps its value even after the fluents it derives from change.

    >>> from dataclasses import dataclass
    >>> from krrood.entity_query_language.factories import a, literal
    >>> @dataclass
    ... class Box:
    ...     color: str
    >>> box = Box("red")
    >>> state = State().add_facts(literal(box).color == "blue")
    >>> a(Box).from_(domain=[box]).color.tolist(state=state)
    ['blue']
    >>> box.color
    'red'
    """

    assignments: dict[GroundTerm, Any] = field(default_factory=dict)
    """
    The value of every ground term this state changes.

    Never mutated; a changed state is a new :class:`State`.
    """

    added_objects: dict[int, Any] = field(default_factory=dict)
    """
    The objects this state adds to the live symbol graph's instances, keyed by their
    :func:`id`. A variable without a domain ranges over them too.

    Never mutated; a changed state is a new :class:`State`.
    """

    removed_objects: dict[int, Any] = field(default_factory=dict)
    """
    The objects this state removes from the live symbol graph's instances, keyed by
    their :func:`id`. Every variable domain, and every ground term whose value is a
    collection of elements, reads without them; a literal still names its object.

    Never mutated; a changed state is a new :class:`State`.
    """

    def __hash__(self) -> int:
        return hash(
            (
                frozenset(self.assignments),
                frozenset(self.added_objects),
                frozenset(self.removed_objects),
            )
        )

    @cached_property
    def assigned_functions(self) -> frozenset[FunctionKey]:
        """
        :return: The :attr:`~krrood.entity_query_language.ground_term.GroundTerm.function_key`
            of every ground term this state assigns.
        """
        return frozenset(term.function_key for term in self.assignments)

    def add_facts(self, *facts: SymbolicExpression) -> State:
        """
        :param facts: Conditions that each fix the value of ground terms, as
            :meth:`~krrood.planning.fact.Fact.from_condition` reads them. A fact whose
            expression has variables fixes the ground term for every binding of them.
        :return: A new state that assigns the values *facts* state on top of this
            state's own assignments, a later value winning where two assign a ground term.
            Each fact reads its bindings and values in the state the facts before it
            build.
        """
        state = self
        for fact in facts:
            state = state.apply_change(change=Assignment(fact=fact).change(state=state))
        return state

    def apply_change(self, *, change: StateChange) -> State:
        """
        :param change: What an effect changes.
        :return: A new state that makes *change* on top of this state.
        """
        return (
            State(
                assignments=self.assignments | change.assignments,
                added_objects=self.added_objects,
                removed_objects=self.removed_objects,
            )
            .with_objects_added(*change.added_objects)
            .with_objects_removed(*change.removed_objects)
        )

    def with_objects_added(self, *objects: Any) -> State:
        """
        :param objects: The objects to add.
        :return: A new state in which *objects* exist, whether this state removed them
            or never had them.
        """
        added_ids = {id(added) for added in objects}
        return State(
            assignments=self.assignments,
            added_objects=self.added_objects | {id(added): added for added in objects},
            removed_objects={
                identifier: removed
                for identifier, removed in self.removed_objects.items()
                if identifier not in added_ids
            },
        )

    def with_objects_removed(self, *objects: Any) -> State:
        """
        :param objects: The objects to remove.
        :return: A new state in which *objects* do not exist, whether they are live or
            this state added them.
        """
        removed_ids = {id(removed) for removed in objects}
        return State(
            assignments=self.assignments,
            added_objects={
                identifier: added
                for identifier, added in self.added_objects.items()
                if identifier not in removed_ids
            },
            removed_objects=self.removed_objects
            | {id(removed): removed for removed in objects},
        )

    def value_of_term(self, *, term: GroundTerm) -> TermValue | None:
        if term.function_key in self.assigned_functions and term in self.assignments:
            return TermValue(
                value=self.without_removed_objects(value=self.assignments[term])
            )
        live_term_value = term.live_term_value()
        if live_term_value is None:
            return None
        return TermValue(
            value=self.without_removed_objects(value=live_term_value.value)
        )

    def without_removed_objects(self, *, value: Any) -> Any:
        """
        :param value: The value of a ground term.
        :return: *value* without the objects this state removes when it is a collection
            of elements that holds any of them, as a new collection of the same type;
            otherwise *value* itself.
        """
        if not isinstance(value, CollectionKind.ELEMENT_SEQUENCE.value):
            return value
        if not any(id(element) in self.removed_objects for element in value):
            return value
        return type(value)(
            element for element in value if id(element) not in self.removed_objects
        )

    def existing_values(self, *, values: Iterable[T]) -> Iterator[T]:
        return (value for value in values if id(value) not in self.removed_objects)

    def added_instances_of(self, *, type_: type[T]) -> Iterator[T]:
        return (
            added for added in self.added_objects.values() if isinstance(added, type_)
        )
