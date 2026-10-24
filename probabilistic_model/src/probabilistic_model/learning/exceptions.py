"""
Exceptions of the methods that fit circuits on data.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from random_events.variable import Variable
from typing_extensions import TYPE_CHECKING

from probabilistic_model.exceptions import ProbabilisticModelException

if TYPE_CHECKING:
    from probabilistic_model.probabilistic_model import ProbabilisticModel


# %% exceptions
@dataclass
class UnsupportedLeafDistributionError(ProbabilisticModelException, TypeError):
    """
    Exception raised when a leaf distribution cannot be learned from weighted rows.
    """

    distribution: ProbabilisticModel
    """
    The distribution that cannot be learned.
    """

    def error_message(self) -> str:
        return (
            f"Leaf distributions of type {type(self.distribution).__name__} cannot be "
            f"learned from weighted rows."
        )

    def suggest_correction(self) -> str:
        return "Use untruncated Gaussian, discrete or symbolic leaf distributions."


@dataclass
class StartingCircuitVariablesMismatchError(ProbabilisticModelException, ValueError):
    """
    Exception raised when a fit is given other variables than those of the circuit it
    starts from.
    """

    circuit_variables: frozenset[Variable]
    """
    The variables of the starting circuit.
    """

    given_variables: frozenset[Variable]
    """
    The variables the fit was given.
    """

    def error_message(self) -> str:
        return (
            f"The fit was given the variables "
            f"{sorted(variable.name for variable in self.given_variables)}, but the "
            f"starting circuit is over "
            f"{sorted(variable.name for variable in self.circuit_variables)}."
        )

    def suggest_correction(self) -> str:
        return (
            "Give no variables, or the variables of the starting circuit, or start from "
            "a circuit over the given variables."
        )
