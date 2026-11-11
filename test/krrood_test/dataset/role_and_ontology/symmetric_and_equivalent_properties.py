"""
A symmetric property, ``friendOf``, two equivalent properties, ``knows`` and ``isAcquaintedWith``, and a transitive
property, ``isPartOf``, equivalent to ``liesWithin``, held by areas and by the buildings standing in them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.mixins import (
    HasEquivalentProperties,
    SymmetricProperty,
    TransitiveProperty,
)
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)


@dataclass(kw_only=True, eq=False)
class PersonWithFriends(Symbol):
    """
    A person related to its friends and to the people it knows.
    """

    name: str = ""
    """
    The name of the person.
    """
    friend_of: Set[PersonWithFriends] = field(default_factory=set)
    """
    The friends of the person.
    """
    knows: Set[PersonWithFriends] = field(default_factory=set)
    """
    The people the person knows.
    """
    is_acquainted_with: Set[PersonWithFriends] = field(default_factory=set)
    """
    The people the person is acquainted with.
    """


@dataclass(kw_only=True)
class FriendOf(PropertyDescriptor, SymmetricProperty):
    """
    Relates a person to its friends, who are friends of the person in turn.
    """


@dataclass(kw_only=True)
class Knows(PropertyDescriptor, HasEquivalentProperties):
    """
    Relates a person to the people it knows, the people it is acquainted with.
    """

    @classmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        return [IsAcquaintedWith]


@dataclass(kw_only=True)
class IsAcquaintedWith(PropertyDescriptor, HasEquivalentProperties):
    """
    Relates a person to the people it is acquainted with, the people it knows.
    """

    @classmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        return [Knows]


PersonWithFriends.friend_of = FriendOf.from_eql_attribute(
    attribute=a(PersonWithFriends).friend_of
)
PersonWithFriends.knows = Knows.from_eql_attribute(attribute=a(PersonWithFriends).knows)
PersonWithFriends.is_acquainted_with = IsAcquaintedWith.from_eql_attribute(
    attribute=a(PersonWithFriends).is_acquainted_with
)


@dataclass(kw_only=True, eq=False)
class Area(Symbol):
    """
    An area that is part of larger areas.
    """

    name: str = ""
    """
    The name of the area.
    """
    is_part_of: Set[Area] = field(default_factory=set)
    """
    The areas the area is part of.
    """
    lies_within: Set[Area] = field(default_factory=set)
    """
    The areas the area lies within.
    """


@dataclass(kw_only=True)
class IsPartOf(PropertyDescriptor, TransitiveProperty, HasEquivalentProperties):
    """
    Relates an area to the areas it is part of, and to the areas those are part of.
    """

    @classmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        return [LiesWithin]


@dataclass(kw_only=True)
class LiesWithin(PropertyDescriptor, HasEquivalentProperties):
    """
    Relates an area to the areas it lies within, the areas it is part of.
    """

    @classmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        return [IsPartOf]


Area.is_part_of = IsPartOf.from_eql_attribute(attribute=a(Area).is_part_of)
Area.lies_within = LiesWithin.from_eql_attribute(attribute=a(Area).lies_within)


@dataclass(kw_only=True, eq=False)
class Building(Symbol):
    """
    A building standing in an area.
    """

    name: str = ""
    """
    The name of the building.
    """
    stands_in: Set[Area] = field(default_factory=set)
    """
    The areas the building is part of.
    """


Building.stands_in = IsPartOf.from_eql_attribute(attribute=a(Building).stands_in)
