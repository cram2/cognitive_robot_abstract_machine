"""
A symmetric property, ``friendOf``, and two equivalent properties, ``knows`` and ``isAcquaintedWith``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set

from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.mixins import (
    HasEquivalentProperties,
    SymmetricProperty,
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
