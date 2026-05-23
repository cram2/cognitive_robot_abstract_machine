"""
Plain ``Animal`` dataclass for the EQL-based RDR (zoo dataset).

Deliberately ordinary: no EQL base classes, no ORM, no special treatment. The RDR
declares a shared ``variable(Animal, domain=...)`` over instances of this class, and
``species`` is the *underspecified* attribute the RDR predicts.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from typing_extensions import Optional


class Species(enum.Enum):
    """The seven mutually-exclusive zoo species classes (UCI dataset target)."""

    mammal = 1
    bird = 2
    reptile = 3
    fish = 4
    amphibian = 5
    insect = 6
    molusc = 7

    def __repr__(self) -> str:
        return f"Species.{self.name}"


@dataclass
class Animal:
    """A zoo animal described by its boolean/numeric traits.

    ``species`` is ``None`` for an unclassified (underspecified) animal and is the
    attribute the RDR fills in.
    """

    name: str
    hair: bool
    feathers: bool
    eggs: bool
    milk: bool
    airborne: bool
    aquatic: bool
    predator: bool
    toothed: bool
    backbone: bool
    breathes: bool
    venomous: bool
    fins: bool
    legs: int
    tail: bool
    domestic: bool
    catsize: bool
    species: Optional[Species] = None
