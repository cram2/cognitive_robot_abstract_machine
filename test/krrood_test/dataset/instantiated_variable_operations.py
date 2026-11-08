from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Self

from krrood.entity_query_language.predicate import (
    Predicate,
    SymbolicFunction,
    symbolic_function,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Noun,
    Verb,
    clause,
)


# %% built types


@dataclass(frozen=True)
class Interval:
    """
    A closed range of integers that a query builds from its bounds.
    """

    lower: int
    """
    The smallest integer in the range.
    """

    upper: int
    """
    The largest integer in the range.
    """

    @classmethod
    def around(cls, center: int, radius: int) -> Interval:
        """
        :param center: The integer in the middle of the range.
        :param radius: How far the range reaches on each side of *center*.
        :return: The range from ``center - radius`` to ``center + radius``.
        """
        return cls(center - radius, center + radius)

    @classmethod
    def point(cls, value: int) -> Self:
        """
        :param value: The only integer in the range.
        :return: The range that holds only *value*.
        """
        return cls(value, value)


# %% symbolic functions


@symbolic_function
def doubled(number: int) -> int:
    """
    :param number: An integer.
    :return: Twice *number*.
    """
    return 2 * number


@symbolic_function
def repeated(number: int) -> list[int]:
    """
    :param number: An integer.
    :return: A list that holds *number* twice.
    """
    return [number, number]


@symbolic_function
def unannotated_doubled(number):
    """
    :param number: An integer.
    :return: Twice *number*.
    """
    return 2 * number


@symbolic_function
def unconstrained_doubled(number: int) -> Any:
    """
    :param number: An integer.
    :return: Twice *number*, annotated as any value.
    """
    return 2 * number


# %% symbolic callable classes


@dataclass(eq=False)
class IsEven(Predicate):
    """
    Whether an integer is even.
    """

    number: int
    """
    The integer that is tested.
    """

    def __call__(self) -> bool:
        return self.number % 2 == 0

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["number"]), Verb("be"), Noun("even"))


@dataclass(eq=False)
class Halved(SymbolicFunction):
    """
    An integer divided by two, rounded down.
    """

    number: int
    """
    The integer that is halved.
    """

    def __call__(self) -> int:
        return self.number // 2

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return Noun(fields["number"])


@dataclass(eq=False)
class UnannotatedHalved(SymbolicFunction):
    """
    An integer divided by two, rounded down, whose :meth:`__call__` has no return
    annotation.
    """

    number: int
    """
    The integer that is halved.
    """

    def __call__(self):
        return self.number // 2

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return Noun(fields["number"])
