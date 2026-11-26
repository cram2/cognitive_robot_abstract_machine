"""
Exceptions of learning the parameters of a circuit on mini-batches.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from probabilistic_model.exceptions import ProbabilisticModelException


# %% exceptions


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
