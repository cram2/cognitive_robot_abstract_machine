"""
Result quantifiers and constraints for the Entity Query Language.

This module defines quantifiers that control how many results are acceptable (e.g., an/the) and the
constraints used to evaluate result counts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Optional, Iterable, Union as TypingUnion, Dict

from ..core.base_expressions import (
    UnaryExpression,
    DerivedExpression,
    SymbolicExpression,
    Bindings,
    OperationResult,
)
from ..failures import (
    NegativeQuantificationError,
    QuantificationConsistencyError,
    GreaterThanExpectedNumberOfSolutions,
    LessThanExpectedNumberOfSolutions,
    UnsupportedNegation,
    NoSolutionFound,
    MultipleSolutionFound,
)
from ..utils import T


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


@dataclass(eq=False)
class ResultQuantifier(UnaryExpression, DerivedExpression, ABC):
    """
    Base for quantifiers that return concrete results from entity/set queries
    (e.g., An, The).
    """

    _quantification_constraint_: Optional[ResultQuantificationConstraint] = None
    """
    The quantification constraint that must be satisfied by the result quantifier if present.
    """

    @property
    def _original_expression_(self) -> SymbolicExpression:
        return self._child_

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[T]:

        result_count = 0
        values = self._child_._evaluate_(parent=self)
        for value in values:
            result_count += 1
            self._assert_satisfaction_of_quantification_constraints_(
                result_count, done=False
            )
            yield OperationResult(value.bindings, False, self)
        self._assert_satisfaction_of_quantification_constraints_(
            result_count, done=True
        )

    def _assert_satisfaction_of_quantification_constraints_(
        self, result_count: int, done: bool
    ):
        """
        Assert the satisfaction of quantification constraints.

        :param result_count: The current count of results
        :param done: Whether all results have been processed
        :raises QuantificationNotSatisfiedError: If the quantification constraints are not satisfied.
        """
        if self._quantification_constraint_:
            self._quantification_constraint_.assert_satisfaction(
                result_count, self, done
            )

    def _invert_(self):
        raise UnsupportedNegation(self.__class__)

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        if self._quantification_constraint_:
            name += f"({self._quantification_constraint_})"
        return name


@dataclass(eq=False, repr=False)
class An(ResultQuantifier):
    """Quantifier that yields all matching results one by one."""

    ...


@dataclass(eq=False, repr=False)
class The(ResultQuantifier):
    """
    Quantifier that expects exactly one result; raises MultipleSolutionFound if more, and NoSolutionFound if none.
    """

    _quantification_constraint_: ResultQuantificationConstraint = field(
        init=False, default_factory=lambda: Exactly(1)
    )

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[TypingUnion[T, Dict[TypingUnion[T, SymbolicExpression], T]]]:
        """
        Evaluates the query object descriptor with the given bindings and yields the results.

        :raises MultipleSolutionFound: If more than one result is found.
        :raises NoSolutionFound: If no result is found.
        """
        try:
            yield from super()._evaluate__(sources)
        except LessThanExpectedNumberOfSolutions:
            raise NoSolutionFound(self)
        except GreaterThanExpectedNumberOfSolutions:
            raise MultipleSolutionFound(self)
