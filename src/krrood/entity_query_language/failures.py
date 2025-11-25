"""
This module defines some custom exception types used by the entity_query_language package.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Type, Any

from ..utils import DataclassException

if TYPE_CHECKING:
    from .symbolic import SymbolicExpression, ResultQuantifier


@dataclass
class QuantificationNotSatisfiedError(DataclassException, ABC):
    """
    Represents a custom exception where the quantification constraints are not satisfied.

    This exception is used to indicate errors related to the quantification
    of the query results.
    """

    expression: ResultQuantifier
    """
    The result quantifier expression where the error occurred.
    """
    expected_number: int
    """
    Expected number of solutions (i.e, quantification constraint value).
    """


@dataclass
class GreaterThanExpectedNumberOfSolutions(QuantificationNotSatisfiedError):
    """
    Represents an error when the number of solutions exceeds the
    expected threshold.
    """

    def __post_init__(self):
        self.message = f"More than {self.expected_number} solutions found for the expression {self.expression}."
        super().__post_init__()


@dataclass
class LessThanExpectedNumberOfSolutions(QuantificationNotSatisfiedError):
    """
    Represents an error that occurs when the number of solutions found
    is lower than the expected number.
    """

    found_number: int
    """
    The number of solutions found.
    """

    def __post_init__(self):
        self.message = (
            f"Found {self.found_number} solutions which is less than the expected {self.expected_number} "
            f"solutions for the expression {self.expression}."
        )
        super().__post_init__()


@dataclass
class MultipleSolutionFound(GreaterThanExpectedNumberOfSolutions):
    """
    Raised when a query unexpectedly yields more than one solution where a single
    result was expected.
    """

    expected_number: int = 1


@dataclass
class NoSolutionFound(LessThanExpectedNumberOfSolutions):
    """
    Raised when a query does not yield any solution.
    """

    expected_number: int = 1
    found_number: int = 0


@dataclass
class UsageError(DataclassException):
    """
    Raised when there is an incorrect usage of the entity query language API.
    """

    ...


@dataclass
class UnsupportedOperation(UsageError):
    """
    Raised when an operation is not supported by the entity query language API.
    """

    ...


@dataclass
class UnSupportedOperand(UnsupportedOperation):
    """
    Raised when an operand is not supported by the operation.
    """

    operation: Type[SymbolicExpression]
    """
    The operation used.
    """
    unsupported_operand: Any
    """
    The operand that is not supported by the operation.
    """

    def __post_init__(self):
        self.message = (
            f"{self.unsupported_operand} cannot be used as an operand for {self.operation} operations."
        )
        super().__post_init__()


@dataclass
class UnsupportedNegation(UnsupportedOperation):
    """
    Raised when negating quantifiers.
    """

    operation_type: Type[SymbolicExpression]
    """
    The type of the operation that is being negated.
    """

    def __post_init__(self):
        self.message = (
            f"Symbolic NOT operations on {self.operation_type} types"
            f" operands are not allowed, you can negate the conditions instead,"
            f" as negating them is most likely not what you want"
            f" because it is ambiguous and can be very expensive to compute."
            f"To Negate Conditions do:"
            f" `not_(condition)` instead of `not_(an(entity(..., condition)))`."
        )
        super().__post_init__()


@dataclass
class QuantificationSpecificationError(UsageError):
    """
    Raised when the quantification constraints specified on the query results are invalid or inconsistent.
    """


@dataclass
class QuantificationConsistencyError(QuantificationSpecificationError):
    """
    Raised when the quantification constraints specified on the query results are inconsistent.
    """

    ...


@dataclass
class NegativeQuantificationError(QuantificationConsistencyError):
    """
    Raised when the quantification constraints specified on the query results have a negative value.
    """

    message: str = f"ResultQuantificationConstraint must be a non-negative integer."


@dataclass
class InvalidEntityType(UsageError):
    """
    Raised when an invalid entity type is given to the quantification operation.
    """

    invalid_entity_type: Type
    """
    The invalid entity type.
    """

    def __post_init__(self):
        self.message = (
            f"The entity type {self.invalid_entity_type} is not valid. It must be a subclass of QueryObjectDescriptor class."
            f"e.g. Entity, or SetOf"
        )
        super().__post_init__()
