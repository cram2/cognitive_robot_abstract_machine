"""
Dataclass records that nest a dataclass and an enum, for tests that turn instances into
constructor source and back.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# %% records


class Species(Enum):
    """
    An enum field value, emitted as ``Species.<member>``.
    """

    CAT = "cat"
    DOG = "dog"


@dataclass(unsafe_hash=True)
class Owner:
    """
    Inner dataclass of a nested dataclass, used to exercise recursive (de)serialization.
    """

    name: str
    """
    The owner's name.
    """


@dataclass
class Animal:
    """
    A dataclass with scalar, enum and nested-dataclass fields.
    """

    name: str
    """
    The animal's name.
    """

    species: Species
    """
    The animal's species.
    """

    age: int
    """
    The animal's age in years.
    """

    owner: Owner
    """
    The animal's owner.
    """


@dataclass
class UnserializableRecord:
    """
    A dataclass whose field holds a value no constructor-source serializer supports.
    """

    tags: list
    """
    A list, which has no constructor-source form.
    """


# %% instances


def rex() -> Animal:
    """
    :return: A fresh :class:`Animal` with every kind of field filled in.
    """
    return Animal("Rex", Species.DOG, 3, Owner("Alice"))
