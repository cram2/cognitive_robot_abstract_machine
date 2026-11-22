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


@dataclass(kw_only=True)
class NotEnoughMemoryError(ProbabilisticModelException, MemoryError):
    """
    Exception raised when not even a batch of one row fits in the free memory of a
    device.
    """

    free_bytes: int
    """
    The bytes still free on the device.
    """

    fixed_bytes: float
    """
    The bytes a training step allocates whatever the size of its batch.
    """

    bytes_per_row: float
    """
    The bytes a training step allocates for every row of its batch.
    """

    def error_message(self) -> str:
        return (
            f"A training step needs {self.fixed_bytes:.0f} bytes and "
            f"{self.bytes_per_row:.0f} bytes per row, but only {self.free_bytes} bytes "
            f"are free."
        )

    def suggest_correction(self) -> str:
        return "Free memory on the device, or train a smaller circuit."


@dataclass(kw_only=True)
class UnmeasurableStepMemoryError(ProbabilisticModelException, ValueError):
    """
    Exception raised when a training step on more rows did not measure more memory, so
    that no memory per row can be told.
    """

    smaller_rows: int
    """
    The rows of the smaller probe batch.
    """

    bytes_per_row: float
    """
    The memory per row the two probes gave.
    """

    def error_message(self) -> str:
        return (
            f"Probing training steps on {self.smaller_rows} and "
            f"{2 * self.smaller_rows} rows gave {self.bytes_per_row} bytes per row."
        )

    def suggest_correction(self) -> str:
        return "Probe on more rows, so that the memory of the rows exceeds the noise."


@dataclass(kw_only=True)
class UnmeasurableHostStepMemoryError(ProbabilisticModelException, ValueError):
    """
    Exception raised when the memory of a torch training step on the CPU is asked for.
    """

    def error_message(self) -> str:
        return (
            "The memory of a torch training step on the CPU cannot be measured: torch "
            "keeps freed host memory resident, so the memory of the process does not "
            "show what a step allocates."
        )

    def suggest_correction(self) -> str:
        return "Choose the batch size on the CPU by hand, or train on a CUDA device."


@dataclass(kw_only=True)
class UnscheduledOptimizerError(ProbabilisticModelException, ValueError):
    """
    Exception raised when a learning rate schedule is given for an optax optimizer whose
    learning rate cannot be changed during training.
    """

    def error_message(self) -> str:
        return (
            "The learning rate of the optimizer cannot be changed between steps, so a "
            "learning rate schedule cannot reduce it."
        )

    def suggest_correction(self) -> str:
        return (
            "Build the optimizer with optax.inject_hyperparams, for example "
            "optax.inject_hyperparams(optax.adam)(learning_rate=1e-2)."
        )


@dataclass(kw_only=True)
class NoRowsToFitError(ProbabilisticModelException, ValueError):
    """
    Exception raised when a circuit is to be fitted on a dataframe without rows.
    """

    def error_message(self) -> str:
        return "The dataframe to fit on has no rows."

    def suggest_correction(self) -> str:
        return "Fit on at least one row."


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
