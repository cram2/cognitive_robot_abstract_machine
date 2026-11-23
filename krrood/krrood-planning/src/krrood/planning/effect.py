"""
Effects - what applying an action changes in a state - and the changes they compute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Any, Mapping

from krrood.class_diagrams.collection_kind import CollectionKind
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import ConditionType, set_of
from krrood.entity_query_language.ground_term import GroundTerm, GroundTermExpression
from krrood.planning.exceptions import ValueIsNotACollection
from krrood.planning.fact import Fact

if TYPE_CHECKING:
    from krrood.planning.state import State

# %% changes


@dataclass(kw_only=True, frozen=True)
class StateChange:
    """
    What one effect changes in the state it is computed in.
    """

    assignments: dict[GroundTerm, Any] = field(default_factory=dict)
    """
    The new value of every ground term the effect assigns.
    """

    added_objects: tuple[Any, ...] = ()
    """
    The objects the effect adds.
    """

    removed_objects: tuple[Any, ...] = ()
    """
    The objects the effect removes.
    """


# %% effects


@dataclass(kw_only=True)
class Effect(ABC):
    """
    A change an action makes to the state it is applied in, for every binding of the
    variables the effect names.
    """

    condition: ConditionType | None = None
    """
    The condition a binding must satisfy in the state for the effect to change anything
    for it, or ``None`` for an effect that changes every binding.
    """

    @abstractmethod
    def change(self, *, state: State) -> StateChange:
        """
        :param state: The state the action is applied in, which the effect reads.
        :return: What the effect changes in *state*.
        """

    def bindings(
        self, *, selected: tuple[SymbolicExpression, ...], state: State
    ) -> list[Mapping[SymbolicExpression, Any]]:
        """
        :param selected: The expressions whose values the effect needs.
        :param state: The state the bindings are read in.
        :return: The value of each of *selected* for every binding that satisfies
            :attr:`condition` in *state*.
        """
        query = set_of(*selected)
        if self.condition is not None:
            query = query.where(self.condition)
        return query.tolist(state=state)


@dataclass(kw_only=True)
class Assignment(Effect):
    """
    Gives ground terms the value a fact states, read in the state the action is applied
    in.
    """

    fact: SymbolicExpression
    """
    The condition that states the fact, as
    :meth:`~krrood.planning.fact.Fact.from_condition` reads it.
    """

    def change(self, *, state: State) -> StateChange:
        fact = Fact.from_condition(condition=self.fact)
        return StateChange(
            assignments=fact.assignments(
                bindings=self.bindings(selected=fact.selected_expressions, state=state)
            )
        )


@dataclass(kw_only=True)
class ElementInsertion(Effect):
    """
    Inserts an element into a collection, after the elements it already holds where it is
    ordered.

    Computing its change raises :class:`~krrood.planning.exceptions.ValueIsNotACollection`
    where a value of :attr:`collection` is not a collection of elements.
    """

    collection: GroundTermExpression
    """
    The expression whose ground terms hold the collections.
    """

    element: SymbolicExpression
    """
    The expression whose value is inserted.
    """

    def change(self, *, state: State) -> StateChange:
        assignments = {}
        selected = (
            *self.collection._ground_term_operands_,
            self.collection,
            self.element,
        )
        for binding in self.bindings(selected=selected, state=state):
            elements = binding[self.collection]
            if not isinstance(elements, CollectionKind.ELEMENT_SEQUENCE.value):
                raise ValueIsNotACollection(collection=self.collection, value=elements)
            inserted = type(elements)((*elements, binding[self.element]))
            assignments |= Fact(expression=self.collection, value=inserted).assignments(
                bindings=(binding,)
            )
        return StateChange(assignments=assignments)


@dataclass(kw_only=True)
class ObjectsEffect(Effect, ABC):
    """
    An effect on the objects that exist.
    """

    objects: SymbolicExpression
    """
    The expression whose values are the objects.
    """

    def objects_in(self, *, state: State) -> tuple[Any, ...]:
        """
        :param state: The state the objects are read in.
        :return: The value of :attr:`objects` for every binding of the effect.
        """
        return tuple(
            binding[self.objects]
            for binding in self.bindings(selected=(self.objects,), state=state)
        )


@dataclass(kw_only=True)
class ObjectCreation(ObjectsEffect):
    """
    Adds the objects an expression builds, such as one made with
    :func:`~krrood.entity_query_language.factories.inference`.

    .. warning:: Building an object of a class whose instances the live symbol graph
        caches adds it to the live symbol graph too; build only objects of classes that
        do not cache their instances.
    """

    def change(self, *, state: State) -> StateChange:
        return StateChange(added_objects=self.objects_in(state=state))


@dataclass(kw_only=True)
class ObjectRemoval(ObjectsEffect):
    """
    Removes objects, which then leave every domain and collection the state reads.
    """

    def change(self, *, state: State) -> StateChange:
        return StateChange(removed_objects=self.objects_in(state=state))
