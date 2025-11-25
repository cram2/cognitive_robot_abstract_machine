from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import TYPE_CHECKING

from .failures import (
    NegativeQuantificationError,
    QuantificationConsistencyError,
    GreaterThanExpectedNumberOfSolutions,
    LessThanExpectedNumberOfSolutions,
)

if TYPE_CHECKING:
    from .symbolic import An, ResultQuantifier


@dataclass
class ResultQuantificationConstraint(ABC):
    """
    A base class that represents a constraint for quantification.
    """

    @abstractmethod
    def assert_satisfaction(
        self, number_of_solutions: int, quantifier: ResultQuantifier, done: bool
    ) -> None:
        """
        Check if the constraint is satisfied, if not, raise a QuantificationNotSatisfiedError exception.

        :param number_of_solutions: The current number of solutions.
        :param quantifier: The quantifier expression of the query.
        :param done: Whether all results have been found.
        :raises: QuantificationNotSatisfiedError: If the constraint is not satisfied.
        """
        ...

    @abstractmethod
    def __repr__(self): ...


@dataclass
class SingleValueQuantificationConstraint(ResultQuantificationConstraint, ABC):
    """
    A class that represents a single value constraint on the result quantification.
    """

    value: int
    """
    The exact value of the constraint.
    """

    def __post_init__(self):
        if self.value < 0:
            raise NegativeQuantificationError()


@dataclass
class Exactly(SingleValueQuantificationConstraint):
    """
    A class that represents an exact constraint on the result quantification.
    """

    def __repr__(self):
        return f"n=={self.value}"

    def assert_satisfaction(
        self, number_of_solutions: int, quantifier: ResultQuantifier, done: bool
    ) -> None:
        if number_of_solutions > self.value:
            raise GreaterThanExpectedNumberOfSolutions(quantifier, self.value)
        elif done and number_of_solutions < self.value:
            raise LessThanExpectedNumberOfSolutions(
                quantifier, self.value, number_of_solutions
            )


@dataclass
class AtLeast(SingleValueQuantificationConstraint):
    """
    A class that specifies a minimum number of results as a quantification constraint.
    """

    def __repr__(self):
        return f"n>={self.value}"

    def assert_satisfaction(
        self, number_of_solutions: int, quantifier: ResultQuantifier, done: bool
    ) -> None:
        if done and number_of_solutions < self.value:
            raise LessThanExpectedNumberOfSolutions(
                quantifier, self.value, number_of_solutions
            )


@dataclass
class AtMost(SingleValueQuantificationConstraint):
    """
    A class that specifies a maximum number of results as a quantification constraint.
    """

    def __repr__(self):
        return f"n<={self.value}"

    def assert_satisfaction(
        self, number_of_solutions: int, quantifier: ResultQuantifier, done: bool
    ) -> None:
        if number_of_solutions > self.value:
            raise GreaterThanExpectedNumberOfSolutions(quantifier, self.value)


@dataclass
class Range(ResultQuantificationConstraint):
    """
    A class that represents a range constraint on the result quantification.
    """

    at_least: AtLeast
    """
    The minimum value of the range.
    """
    at_most: AtMost
    """
    The maximum value of the range.
    """

    def __post_init__(self):
        """
        Validate quantification constraints are consistent.
        """
        if self.at_most.value < self.at_least.value:
            raise QuantificationConsistencyError(
                message=f"at_most {self.at_most} cannot be less than at_least {self.at_least}."
            )

    def assert_satisfaction(
        self, number_of_solutions: int, quantifier: ResultQuantifier, done: bool
    ) -> None:
        self.at_least.assert_satisfaction(number_of_solutions, quantifier, done)
        self.at_most.assert_satisfaction(number_of_solutions, quantifier, done)

    def __repr__(self):
        return f"{self.at_least}<=n<={self.at_most}"
