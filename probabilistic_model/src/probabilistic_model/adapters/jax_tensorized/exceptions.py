from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from krrood.exceptions import DataclassException
from random_events.variable import Variable


@dataclass
class StatesAreNotColumnIndicesError(DataclassException):
    """
    Raised when the states of a discrete variable cannot be the columns of the
    probability table of a discrete layer of the ``jax`` package.

    That layer looks the probability of a value up in the column with the value as its
    index, so the states of an integer variable have to be non-negative and the states
    of a symbolic variable have to be the numbers from zero to the size of its domain.
    """

    variable: Variable = field(kw_only=True)
    """
    The variable of the discrete layer.
    """

    states: np.ndarray = field(kw_only=True)
    """
    The states that are not column indices.
    """

    def error_message(self) -> str:
        return (
            f"The states {self.states.tolist()} of {self.variable.name} are not the "
            f"column indices of a probability table."
        )

    def suggest_correction(self) -> str:
        return (
            "Use an integer variable with non-negative states or a symbolic variable "
            "whose domain elements hash to 0, 1, 2, ..., such as an IntEnum."
        )
