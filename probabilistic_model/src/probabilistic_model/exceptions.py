from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, TYPE_CHECKING

from krrood.exceptions import DataclassException
from random_events.variable import Variable

if TYPE_CHECKING:
    from random_events.product_algebra import Event

    from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class IntractableError(DataclassException):
    """
    Exception raised when an inference is intractable for a model.

    For instance, the mode of a non-deterministic model.
    """

    model: ProbabilisticModel

    def error_message(self) -> str:
        return f"Inference is intractable for {self.model}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UndefinedOperationError(DataclassException):
    """
    Exception raised when an operation is not defined for a model.

    For instance, invoking the CDF of a model that contains symbolic variables.
    """

    model: ProbabilisticModel

    def error_message(self) -> str:
        return f"Operation is not defined for {self.model}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ProbabilisticCircuitRequiredError(DataclassException):
    """
    Exception raised when a distribution is asked for something whose answer only a
    probabilistic circuit can represent.
    """

    model: ProbabilisticModel
    """
    The distribution that was asked.
    """

    def error_message(self) -> str:
        return (
            f"{self.model} cannot answer this with a distribution of its own, since the "
            f"answer needs a probabilistic circuit."
        )

    def suggest_correction(self) -> str:
        return "Wrap the distribution into a probabilistic circuit and ask the circuit."


@dataclass
class EventIsNotABoxError(ProbabilisticCircuitRequiredError):
    """
    Exception raised when a model is asked to confine itself to something other than a
    single box.

    A box is one simple interval per variable. Anything wider leaves a shape no single
    truncated distribution describes.
    """

    event: Event
    """
    What it was asked to confine itself to.
    """

    def error_message(self) -> str:
        return f"{self.event} is not one box, so {self.model} cannot be confined to it."

    def suggest_correction(self) -> str:
        return (
            "Confine it to one simple interval per variable, or wrap the distribution "
            "into a probabilistic circuit and confine the circuit."
        )


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

    def error_message(self) -> str:
        return f"Expected shape {self.expected_shape}, received shape {self.received_shape}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class VariableNotInDistributionError(DataclassException):
    """
    Exception raised when a variable is named that a distribution is not over.

    A distribution's mean and covariance are laid out by its own variables, so a
    variable outside them has no index to be read from or written to.
    """

    variable: Variable
    """
    The variable that was named.
    """

    variables: List[Variable]
    """
    The variables the distribution is over.
    """

    def error_message(self) -> str:
        return (
            f"{self.variable} is not one of the variables "
            f"{[str(variable) for variable in self.variables]}."
        )

    def suggest_correction(self) -> str:
        return "Name one of the variables the distribution is over."
