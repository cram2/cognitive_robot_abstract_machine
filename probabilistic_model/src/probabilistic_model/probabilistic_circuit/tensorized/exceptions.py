from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Type

from krrood.exceptions import DataclassException


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


@dataclass
class UndefinedCumulativeDistributionError(DataclassException, TypeError):
    """
    Exception raised when the cumulative distribution function of a layer over an
    unordered variable is queried.
    """

    layer_type: Type
    """
    The type of the layer.
    """

    def error_message(self) -> str:
        return (
            f"{self.layer_type.__name__} has no cumulative distribution function, "
            f"since the states of its variable are not ordered."
        )

    def suggest_correction(self) -> str:
        return "Query the cumulative distribution only over ordered variables."
