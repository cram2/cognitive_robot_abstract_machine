from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    List,
    Tuple,
    Union,
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from krrood.symbolic_math.symbolic_math import FloatVariable


class SymbolicMathError(Exception):
    pass


@dataclass
class UnsupportedOperationError(SymbolicMathError, TypeError):
    """
    Represents an error for unsupported operations between incompatible types.

    This class is derived from `SymbolicMathError` and `TypeError` and is specifically
    designed to handle cases where an operation is attempted between two arguments
    that are of incompatible types. It stores details about the operation and the
    involved arguments, and provides an error message that highlights the problematic
    types.
    """

    operation: str
    """The name of the operation that was attempted (e.g., '+', '-', etc.)."""
    left: Any
    """The first argument involved in the operation."""
    right: Any
    """The second argument involved in the operation."""

    def __post_init__(self):
        super().__init__(
            f"unsupported operand type(s) for {self.operation}: '{self.left.__class__.__name__}' and '{self.right.__class__.__name__}'"
        )


@dataclass
class WrongDimensionsError(SymbolicMathError):
    """
    Represents an error for mismatched dimensions.
    """

    expected_dimensions: Tuple[int, int]
    actual_dimensions: Tuple[int, int]

    def __post_init__(self):
        msg = f"Expected {self.expected_dimensions} dimensions, but got {self.actual_dimensions}."
        super().__init__(msg)


@dataclass
class NotScalerError(WrongDimensionsError):
    """
    Exception raised for errors when a non-scalar input is provided.
    """

    expected_dimensions: Tuple[int, int] = field(default=(1, 1), init=False)


@dataclass
class NotSquareMatrixError(WrongDimensionsError):
    """
    Represents an error raised when an operation requires a square matrix but the input is not.
    """

    actual_dimensions: Tuple[int, int]

    def __post_init__(self):
        msg = f"Expected a square matrix, but got {self.actual_dimensions} dimensions."
        super().__init__(msg)


@dataclass
class HasFreeVariablesError(SymbolicMathError):
    """
    Raised when an operation can't be performed on an expression with free variables.
    """

    variables: List[FloatVariable]

    def __post_init__(self):
        msg = f"Operation can't be performed on expression with free variables: {self.variables}."
        super().__init__(msg)


class ExpressionEvaluationError(SymbolicMathError):
    """
    Represents an exception raised during the evaluation of a symbolic mathematical expression.
    """


@dataclass
class WrongNumberOfArgsError(ExpressionEvaluationError):
    """
    This error is specifically used in expression evaluation scenarios where a certain number of arguments
    are required and the actual number provided is incorrect.
    """

    expected_number_of_args: int
    actual_number_of_args: int

    def __post_init__(self):
        msg = f"Expected {self.expected_number_of_args} arguments, but got {self.actual_number_of_args}."
        super().__init__(msg)


@dataclass
class DuplicateVariablesError(SymbolicMathError):
    """
    Raised when duplicate variables are found in an operation that requires unique variables.
    """

    variables: List[FloatVariable]

    def __post_init__(self):
        msg = f"Operation failed due to duplicate variables: {self.variables}. All variables must be unique."
        super().__init__(msg)
