from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, List, Type, TYPE_CHECKING

from krrood.exceptions import DataclassException
from random_events.variable import Variable

if TYPE_CHECKING:
    from random_events.product_algebra import Event, SimpleEvent

    from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class UnboundedEventError(DataclassException):
    """
    Raised when a uniform measure is asked for over an event that does not bound one of
    its continuous variables from below and above.
    """

    variable: Variable
    """
    The variable without bounds.
    """

    def error_message(self) -> str:
        return (
            f"The event does not bound '{self.variable.name}' from below and above, so "
            f"no uniform measure over it exists."
        )

    def suggest_correction(self) -> str:
        return f"Bound '{self.variable.name}' from below and above in the event."


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


@dataclass
class NonContinuousVariableError(DataclassException, ValueError):
    """
    Exception raised when a model that only fits continuous variables is given others.
    """

    variables: List[Variable]
    """
    The variables that are not continuous.
    """

    def error_message(self) -> str:
        return (
            f"The variables {[str(variable) for variable in self.variables]} are not "
            f"continuous."
        )

    def suggest_correction(self) -> str:
        return "Fit them with a model that supports discrete variables."


@dataclass
class NoClosedFormError(DataclassException, NotImplementedError):
    """
    Exception raised when a query has no answer that the model or layer that was asked
    can represent in closed form.
    """

    asked_type: Type
    """
    The type of the model or layer that was asked.
    """

    query: Callable
    """
    The method that was queried.
    """

    def error_message(self) -> str:
        return f"{self.asked_type.__name__}.{self.query.__name__} has no closed form."

    def suggest_correction(self) -> str:
        return "Ask the query before truncating, or approximate it by sampling."


@dataclass
class InvalidMomentOrderError(DataclassException, ValueError):
    """
    Exception raised when a moment is asked for an order that is not a whole number or
    is negative.
    """

    order: Any
    """
    The order that was asked for.
    """

    def error_message(self) -> str:
        return f"There is no moment of order {self.order}."

    def suggest_correction(self) -> str:
        return "Ask for an order that is a whole number and not negative."


@dataclass
class EventMixesPointsAndIntervalsError(DataclassException, ValueError):
    """
    Exception raised when an event is to be measured uniformly while one of its
    continuous variables takes points and intervals at once.

    A point has no width, so there is no weight that puts it in proportion to an
    interval.
    """

    event: SimpleEvent
    """
    The event whose variable mixes them.
    """

    def error_message(self) -> str:
        return (
            f"A continuous variable of {self.event} takes points and intervals at "
            f"once."
        )

    def suggest_correction(self) -> str:
        return "Make every continuous variable either points only or intervals only."


@dataclass
class MixtureOverDifferentVariablesError(DataclassException, ValueError):
    """
    Exception raised when a mixture is built of components that are not over the same
    variables.

    A mixture is only smooth if every component models the same variables, so a query
    over a mixture of differing components has no well-defined answer.
    """

    variables_per_component: List[List[Variable]]
    """
    The variables of every component, in the order the components were given.
    """

    def error_message(self) -> str:
        return (
            "A mixture needs components over the same variables, received "
            f"{[[str(variable) for variable in variables] for variables in self.variables_per_component]}."
        )

    def suggest_correction(self) -> str:
        return "Marginalize every component to the variables the mixture is over."


@dataclass
class ProductOverSharedVariablesError(DataclassException, ValueError):
    """
    Exception raised when a product is built of factors that share a variable.

    A product is only decomposable if its factors are over disjoint variables, so a
    shared variable would be counted twice.
    """

    shared_variables: List[Variable]
    """
    The variables that more than one factor is over.
    """

    def error_message(self) -> str:
        return (
            "A product needs factors over disjoint variables, but "
            f"{[str(variable) for variable in self.shared_variables]} "
            "appear in more than one factor."
        )

    def suggest_correction(self) -> str:
        return "Marginalize the factors to disjoint sets of variables."


@dataclass
class EmptyCompositionError(DataclassException, ValueError):
    """
    Exception raised when a mixture or a product is built of no components at all.
    """

    composition: Callable
    """
    The method that was called.
    """

    def error_message(self) -> str:
        return f"{self.composition.__name__} needs at least one component."

    def suggest_correction(self) -> str:
        return "Leave the composition out instead of building it of nothing."


@dataclass
class NumberOfWeightsMismatchError(DataclassException, ValueError):
    """
    Exception raised when a mixture does not get exactly one weight per component.
    """

    number_of_weights: int
    """
    The number of weights given.
    """

    number_of_components: int
    """
    The number of components to mix.
    """

    def error_message(self) -> str:
        return (
            f"Got {self.number_of_weights} weights for "
            f"{self.number_of_components} components."
        )

    def suggest_correction(self) -> str:
        return "Pass exactly one weight per component."
