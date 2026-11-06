"""
A symmetric property, ``friendOf``, two equivalent properties, ``knows`` and ``isAcquaintedWith``, and a transitive
property, ``isPartOf``, equivalent to ``liesWithin``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set

from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.mixins import (
    HasEquivalentProperties,
    SymmetricProperty,
    TransitiveProperty,
)
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)


@dataclass(eq=False)
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


@dataclass
class FriendOf(PropertyDescriptor, SymmetricProperty):
    """
    Relates a person to its friends, who are friends of the person in turn.
    """


@dataclass
class Knows(PropertyDescriptor, HasEquivalentProperties):
    """
    Relates a person to the people it knows, the people it is acquainted with.
    """

    @classmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        return [IsAcquaintedWith]


@dataclass
class IsAcquaintedWith(PropertyDescriptor, HasEquivalentProperties):
    """
    Relates a person to the people it is acquainted with, the people it knows.
    """

    @classmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        return [Knows]


PersonWithFriends.friend_of = FriendOf(PersonWithFriends, "friend_of")
PersonWithFriends.knows = Knows(PersonWithFriends, "knows")
PersonWithFriends.is_acquainted_with = IsAcquaintedWith(
    PersonWithFriends, "is_acquainted_with"
)


@dataclass(eq=False)
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


@dataclass
class IsPartOf(PropertyDescriptor, TransitiveProperty, HasEquivalentProperties):
    """
    Relates an area to the areas it is part of, and to the areas those are part of.
    """

    @classmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        return [LiesWithin]


@dataclass
class LiesWithin(PropertyDescriptor, HasEquivalentProperties):
    """
    Relates an area to the areas it lies within, the areas it is part of.
    """

    @classmethod
    def get_equivalent_properties(cls) -> list[type[PropertyDescriptor]]:
        return [IsPartOf]


Area.is_part_of = IsPartOf(Area, "is_part_of")
Area.lies_within = LiesWithin(Area, "lies_within")
