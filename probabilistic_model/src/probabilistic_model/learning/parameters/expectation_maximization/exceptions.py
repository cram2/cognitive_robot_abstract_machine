"""
Exceptions of learning the parameters of a circuit by expectation maximization.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from random_events.variable import Variable

from probabilistic_model.exceptions import ProbabilisticModelException


# %% exceptions


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class UnsupportedLayerError(ProbabilisticModelException, TypeError):
    """
    Exception raised when tensorized expectation maximization is asked to learn a
    circuit with a layer whose parameters it cannot learn.
    """

    layer_type: type
    """
    The type of the layer that cannot be learned.
    """

    def error_message(self) -> str:
        return (
            f"Expectation maximization cannot learn the parameters of a "
            f"{self.layer_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            "Learn circuits whose layers with parameters are sum and Gaussian layers, "
            "or train the circuit by gradient descent."
        )
