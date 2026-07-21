"""
Result quantifiers and constraints for the Entity Query Language.

This module defines quantifiers that control how many results are acceptable (e.g.,
an/the) and the constraints used to evaluate result counts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Optional

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.exceptions import (
    InvalidQuantificationRangeError,
    NegativeQuantificationError,
    GreaterThanExpectedNumberOfSolutions,
    LessThanExpectedNumberOfSolutions,
    NoSolutionFound,
    MultipleSolutionFound,
)


@dataclass
class ResultQuantificationConstraint(ABC):
    """
    A base class that represents a constraint for quantification.
    """

    @abstractmethod
    def assert_satisfaction(
        self,
        number_of_solutions: int,
        quantified_expression: SymbolicExpression,
        done: bool,
    ) -> None:
        """
        Check if the constraint is satisfied, if not, raise a
        QuantificationNotSatisfiedError exception.

        :param number_of_solutions: The current number of solutions.
        :param quantified_expression: The query expression being quantified. This will be mentioned in errors.
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
        self,
        number_of_solutions: int,
        quantified_expression: SymbolicExpression,
        done: bool,
    ) -> None:
        if number_of_solutions > self.value:
            raise GreaterThanExpectedNumberOfSolutions(
                quantified_expression, self.value
            )
        elif done and number_of_solutions < self.value:
            raise LessThanExpectedNumberOfSolutions(
                quantified_expression, self.value, number_of_solutions
            )


@dataclass
class ExactlyOne(Exactly):
    """
    A definite single-result constraint. Unlike :class:`Exactly`, it raises the definite-result
    errors (:class:`NoSolutionFound` / :class:`MultipleSolutionFound`) directly, so a definite query
    (``the``) enforces exactly one result without translating generic quantification errors.
    """

    value: int = 1
    """
    The required number of results, always one.
    """

    def assert_satisfaction(
        self,
        number_of_solutions: int,
        quantified_expression: SymbolicExpression,
        done: bool,
    ) -> None:
        if number_of_solutions > self.value:
            raise MultipleSolutionFound(quantified_expression)
        if done and number_of_solutions < self.value:
            raise NoSolutionFound(quantified_expression)


@dataclass
class AtLeast(SingleValueQuantificationConstraint):
    """
    A class that specifies a minimum number of results as a quantification constraint.
    """

    def __repr__(self):
        return f"n>={self.value}"

    def assert_satisfaction(
        self,
        number_of_solutions: int,
        quantified_expression: SymbolicExpression,
        done: bool,
    ) -> None:
        if done and number_of_solutions < self.value:
            raise LessThanExpectedNumberOfSolutions(
                quantified_expression, self.value, number_of_solutions
            )


@dataclass
class AtMost(SingleValueQuantificationConstraint):
    """
    A class that specifies a maximum number of results as a quantification constraint.
    """

    def __repr__(self):
        return f"n<={self.value}"

    def assert_satisfaction(
        self,
        number_of_solutions: int,
        quantified_expression: SymbolicExpression,
        done: bool,
    ) -> None:
        if number_of_solutions > self.value:
            raise GreaterThanExpectedNumberOfSolutions(
                quantified_expression, self.value
            )


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
            raise InvalidQuantificationRangeError(self.at_least, self.at_most)

    def assert_satisfaction(
        self,
        number_of_solutions: int,
        quantified_expression: SymbolicExpression,
        done: bool,
    ) -> None:
        self.at_least.assert_satisfaction(
            number_of_solutions, quantified_expression, done
        )
        self.at_most.assert_satisfaction(
            number_of_solutions, quantified_expression, done
        )

    def __repr__(self):
        return f"{self.at_least}<=n<={self.at_most}"


@dataclass
class ResultQuantifier(ABC):
    """
    Marker for the kind of result quantification a query requests (e.g. :class:`An`, :class:`The`).

    A query stores the requested kind, and its quantification pipeline stage enforces that kind's
    default result-count constraint.
    """

    @classmethod
    def _default_constraint_(cls) -> Optional[ResultQuantificationConstraint]:
        """
        :return: The result-count constraint this quantifier kind enforces by default, applied by the
            query's quantification pipeline stage when no explicit constraint is given.
        """
        return None


@dataclass
class An(ResultQuantifier):
    """Quantifier that accepts all matching results."""


@dataclass
class The(ResultQuantifier):
    """Quantifier that requires exactly one result."""

    @classmethod
    def _default_constraint_(cls) -> ResultQuantificationConstraint:
        return ExactlyOne()
