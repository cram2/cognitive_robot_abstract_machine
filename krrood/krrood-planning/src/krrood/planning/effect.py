"""
Effects - what applying an action changes in a state - and the changes they compute.
"""

from __future__ import annotations

import collections.abc
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Collection, Mapping

from krrood.class_diagrams.collection_kind import CollectionKind
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import ConditionType, set_of
from krrood.entity_query_language.ground_term import GroundTerm, GroundTermExpression
from krrood.entity_query_language.query.query import Query
from krrood.planning.exceptions import (
    InsertedElementHasWrongType,
    ValueIsNotACollection,
)
from krrood.planning.fact import Fact

# %% changes


@dataclass(kw_only=True, frozen=True)
class CollectionInsertion:
    """
    Elements inserted into a collection, after the elements it holds where it is ordered.
    """

    elements: tuple[Any, ...]
    """
    The inserted elements, in the order they were inserted.
    """

    def insert_into(self, *, collection: Any) -> Any:
        """
        :param collection: A collection of elements, which is left unchanged.
        :return: A copy of *collection*, of the same type, that also holds
            :attr:`elements`.
        """
        inserted = copy.copy(collection)
        if isinstance(inserted, collections.abc.MutableSequence):
            inserted.extend(self.elements)
            return inserted
        if isinstance(inserted, collections.abc.MutableSet):
            inserted.update(self.elements)
            return inserted
        return type(collection)((*collection, *self.elements))

    def extend(self, *, later_insertion: CollectionInsertion) -> CollectionInsertion:
        """
        :param later_insertion: Elements inserted after these.
        :return: The insertion of these elements and then those of *later_insertion*;
            this insertion is left unchanged.
        """
        return CollectionInsertion(elements=(*self.elements, *later_insertion.elements))


@dataclass(kw_only=True, frozen=True)
class StateChange:
    """
    What one effect changes in the state it is computed in.
    """

    assignments: dict[GroundTerm, Any] = field(default_factory=dict)
    """
    The new value of every ground term the effect assigns.
    """

    insertions: dict[GroundTerm, CollectionInsertion] = field(default_factory=dict)
    """
    The elements the effect inserts into the collection each ground term holds.
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

    @property
    @abstractmethod
    def selected_expressions(self) -> tuple[SymbolicExpression, ...]:
        """
        :return: The expressions whose values a binding gives :meth:`change_of_binding`.
        """

    @abstractmethod
    def change_of_binding(
        self, *, binding: Mapping[SymbolicExpression, Any]
    ) -> StateChange:
        """
        :param binding: The values of :attr:`selected_expressions` for one result of
            :meth:`get_change_to_state_query`.
        :return: What the effect changes for *binding*.
        """

    def get_change_to_state_query(self) -> Query:
        """
        :return: The query whose results, each evaluated in the state the action is
            applied in, are the bindings :meth:`change_of_binding` turns into changes:
            the distinct values of :attr:`selected_expressions` where :attr:`condition`
            holds.
        """
        query = set_of(*self.selected_expressions).distinct()
        if self.condition is None:
            return query
        return query.where(self.condition)


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

    @property
    def selected_expressions(self) -> tuple[SymbolicExpression, ...]:
        return Fact.from_condition(condition=self.fact).selected_expressions

    def change_of_binding(
        self, *, binding: Mapping[SymbolicExpression, Any]
    ) -> StateChange:
        return StateChange(
            assignments=Fact.from_condition(condition=self.fact).assignments(
                bindings=(binding,)
            )
        )


@dataclass(kw_only=True)
class ElementInsertion(Effect):
    """
    Inserts an element into a collection, last where it is ordered. Turning a binding
    into a change raises :class:`~krrood.planning.exceptions.ValueIsNotACollection`,
    :class:`~krrood.planning.exceptions.InsertedElementHasWrongType` and what
    :meth:`~krrood.planning.fact.Fact.check_term_can_change` raises.
    """

    collection: GroundTermExpression[GroundTerm, Collection[Any]]
    """
    The expression whose ground terms hold the collections.
    """

    element: SymbolicExpression
    """
    The expression whose value is inserted.
    """

    @property
    def selected_expressions(self) -> tuple[SymbolicExpression, ...]:
        return (*self.collection._ground_term_operands_, self.collection, self.element)

    def change_of_binding(
        self, *, binding: Mapping[SymbolicExpression, Any]
    ) -> StateChange:
        elements = binding[self.collection]
        if not isinstance(elements, CollectionKind.ELEMENT_SEQUENCE.value):
            raise ValueIsNotACollection(collection=self.collection, value=elements)
        term = self.collection._ground_term_(
            operand_values=tuple(
                binding[operand] for operand in self.collection._ground_term_operands_
            )
        )
        Fact.check_term_can_change(term=term)
        element = binding[self.element]
        if not term.accepts_element(element=element):
            raise InsertedElementHasWrongType(term=term, element=element)
        return StateChange(insertions={term: CollectionInsertion(elements=(element,))})


@dataclass(kw_only=True)
class ObjectsEffect(Effect, ABC):
    """
    An effect on the objects that exist.
    """

    objects: SymbolicExpression
    """
    The expression whose values are the objects.
    """

    @property
    def selected_expressions(self) -> tuple[SymbolicExpression, ...]:
        return (self.objects,)


@dataclass(kw_only=True)
class ObjectCreation(ObjectsEffect):
    """
    Adds the objects an expression builds, such as one made with
    :func:`~krrood.entity_query_language.factories.inference`.

    .. note:: Build planned objects with ``_persist_instance_=False``, so a class whose
        instances the live symbol graph caches leaves them out of it.
    """

    def change_of_binding(
        self, *, binding: Mapping[SymbolicExpression, Any]
    ) -> StateChange:
        return StateChange(added_objects=(binding[self.objects],))


@dataclass(kw_only=True)
class ObjectRemoval(ObjectsEffect):
    """
    Removes objects, which then leave every domain and collection the state reads.
    """

    def change_of_binding(
        self, *, binding: Mapping[SymbolicExpression, Any]
    ) -> StateChange:
        return StateChange(removed_objects=(binding[self.objects],))
