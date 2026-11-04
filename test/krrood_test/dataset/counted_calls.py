"""
Mimic predicates and symbolic functions that count how often they are called, for testing
that one evaluation calls each of them once per distinct tuple of arguments.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Iterator

from krrood.entity_query_language.predicate import (
    Predicate,
    RenderedFields,
    SymbolicFunction,
    symbolic_function,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    FunctionVerbalizationTemplates,
)

# %% places and visits


@dataclass(eq=False)
class Place:
    """
    A place the counted procedures are asked about.

    Compared by identity, so two places at the same distance are different arguments.
    """

    distance: float
    """
    The distance to the place.
    """

    times_asked: int = 0
    """
    How often a counted procedure was called with this place.
    """


@dataclass(eq=False)
class Visit:
    """
    A visit to a place; several visits can share one place.
    """

    place: Place
    """
    The place visited.
    """


# %% counted procedures


@dataclass(eq=False)
class IsWithinReach(Predicate):
    """
    Whether a place is at most a given distance away.
    """

    place: Place
    """
    The place checked.
    """

    reach: float
    """
    The greatest distance that is within reach.
    """

    def __call__(self) -> bool:
        self.place.times_asked += 1
        return self.place.distance <= self.reach

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        return FunctionVerbalizationTemplates.possessive(cls, *fields.values())


@dataclass(eq=False)
class TravelTime(SymbolicFunction):
    """
    The time it takes to travel to a place at unit speed.
    """

    place: Place
    """
    The place travelled to.
    """

    def __call__(self) -> float:
        self.place.times_asked += 1
        return self.place.distance

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        return FunctionVerbalizationTemplates.possessive(cls, *fields.values())


@symbolic_function
def distance_to(place: Place) -> float:
    """
    :param place: The place measured.
    :return: The distance to *place*.
    """
    place.times_asked += 1
    return place.distance


@symbolic_function
def distances_on_the_way_to(place: Place) -> Iterator[float]:
    """
    :param place: The place travelled to.
    :return: The distance to *place* and each whole distance before it, which a
        reader consumes.
    """
    place.times_asked += 1
    yield from range(int(place.distance) + 1)
