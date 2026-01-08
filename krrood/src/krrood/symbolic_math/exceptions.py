from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    List,
    Tuple,
    TYPE_CHECKING,
    Any,
)

from ..utils import DataclassException

if TYPE_CHECKING:
    from krrood.symbolic_math.symbolic_math import FloatVariable


@dataclass
class SymbolicMathError(DataclassException):
    """
    Represents an error specifically related to symbolic mathematics operations.
    """


@dataclass
class UnsupportedOperationError(SymbolicMathError, TypeError):
    """
    Represents an error for unsupported operations between incompatible types.
    """

    operation: str
    """The name of the operation that was attempted (e.g., '+', '-', etc.)."""
    left: Any
    """The first argument involved in the operation."""
    right: Any
    """The second argument involved in the operation."""
    message: str = field(init=False)

    def __post_init__(self):
        self.message = f"unsupported operand type(s) for {self.operation}: '{self.left.__class__.__name__}' and '{self.right.__class__.__name__}'"
        super().__post_init__()


@dataclass
class WrongDimensionsError(SymbolicMathError):
    """
    Represents an error for mismatched dimensions.
    """

    expected_dimensions: Tuple[int, int] | str
    actual_dimensions: Tuple[int, int]
    message: str = field(init=False)

    def __post_init__(self):
        self.message = f"Expected {self.expected_dimensions} dimensions, but got {self.actual_dimensions}."
        super().__post_init__()


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

    expected_dimensions: Tuple[int, int] = field(default="square", init=False)
    actual_dimensions: Tuple[int, int]


@dataclass
class HasFreeVariablesError(SymbolicMathError):
    """
    Raised when an operation can't be performed on an expression with free variables.
    """

    variables: List[FloatVariable]
    message: str = field(init=False)

    def __post_init__(self):
        self.message = f"Operation can't be performed on expression with free variables: {self.variables}."
        super().__post_init__()


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
    message: str = field(init=False)

    def __post_init__(self):
        self.message = f"Expected {self.expected_number_of_args} arguments, but got {self.actual_number_of_args}."
        super().__post_init__()


@dataclass
class DuplicateVariablesError(SymbolicMathError):
    """
    Raised when duplicate variables are found in an operation that requires unique variables.
    """

    variables: List[FloatVariable]
    message: str = field(init=False)

    def __post_init__(self):
        self.message = f"Operation failed due to duplicate variables: {self.variables}. All variables must be unique."
        super().__post_init__()
