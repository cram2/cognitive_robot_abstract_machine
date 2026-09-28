from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
class NonPositiveScaleError(DataclassException, ValueError):
    """
    Exception raised when a layer has a scale that is not positive.
    """

    scale: np.ndarray
    """
    The scales that are not positive.
    """

    def error_message(self) -> str:
        return f"The scales {self.scale.tolist()} are not positive."

    def suggest_correction(self) -> str:
        return "Give every node a positive scale."
