"""
A hierarchy of three properties, ``loves`` ⊑ ``likes`` ⊑ ``knows``: whoever loves someone
likes them, and whoever likes someone knows them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)


@dataclass(eq=False)
class PersonWithFeelings(Symbol):
    """
    A person related to the people it knows, likes and loves.
    """

    name: str = ""
    """
    The name of the person.
    """
    knows: Set[PersonWithFeelings] = field(default_factory=set)
    """
    The people the person knows.
    """
    likes: Set[PersonWithFeelings] = field(default_factory=set)
    """
    The people the person likes.
    """
    loves: Set[PersonWithFeelings] = field(default_factory=set)
    """
    The people the person loves.
    """


@dataclass
class Knows(PropertyDescriptor):
    """
    Relates a person to the people it knows.
    """


@dataclass
class Likes(Knows):
    """
    Relates a person to the people it likes, whom it also knows.
    """


@dataclass
class Loves(Likes):
    """
    Relates a person to the people it loves, whom it also likes.
    """


PersonWithFeelings.knows = Knows.from_eql_attribute(a(PersonWithFeelings).knows)
PersonWithFeelings.likes = Likes.from_eql_attribute(a(PersonWithFeelings).likes)
PersonWithFeelings.loves = Loves.from_eql_attribute(a(PersonWithFeelings).loves)
