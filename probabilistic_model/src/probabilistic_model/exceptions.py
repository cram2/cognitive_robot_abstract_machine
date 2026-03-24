from dataclasses import dataclass
from typing import Any
from krrood.utils import DataclassException


@dataclass
class IntractableError(DataclassException):
    """
    Exception raised when an inference is intractable for a model.
    For instance, the mode of a non-deterministic model.
    """

    model: Any

    def __post_init__(self):
        self.message = f"Inference is intractable for {self.model}."


@dataclass
class UndefinedOperationError(DataclassException):
    """
    Exception raised when an operation is not defined for a model.
    For instance, invoking the CDF of a model that contains symbolic variables.
    """

    model: Any

    def __post_init__(self):
        self.message = f"Operation is not defined for {self.model}."

@dataclass
class ShapeMismatchError(DataclassException, ValueError):
    """
    Exception raised when the shape of two objects does not match.
    """

    received_shape: Any
    """
    The first object to compare.
    """

    expected_shape: Any
    """
    The second object to compare.
    """

    def __post_init__(self):
        self.message = f"Expected shape {self.expected_shape}, received shape {self.received_shape}"
