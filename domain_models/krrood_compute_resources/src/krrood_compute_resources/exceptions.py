"""
Exceptions of the compute resources model.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from krrood.exceptions import DataclassException

# %% memory


@dataclass(kw_only=True)
class MissingMemoryFieldError(DataclassException, LookupError):
    """
    Exception raised when a Linux memory file lacks a field that is read from it.
    """

    memory_file: str
    """
    The memory file that was read.
    """

    memory_field: str
    """
    The memory field that was not found in it.
    """

    def error_message(self) -> str:
        return f"{self.memory_file} has no field {self.memory_field}."

    def suggest_correction(self) -> str:
        return "Run on a Linux kernel that reports this field."


# %% division of a machine


@dataclass(kw_only=True)
class EmptyMachineDivisionError(DataclassException, ValueError):
    """
    Exception raised when a machine is divided into no shares at all.
    """

    def error_message(self) -> str:
        return "A machine division needs at least one share."

    def suggest_correction(self) -> str:
        return "Give at least one share of the GPU or of the processor cores."


@dataclass(kw_only=True)
class MissingGraphicsProcessorError(DataclassException, LookupError):
    """
    Exception raised when shares of the GPU are asked of a machine without one.
    """

    graphics_processor_shares: int
    """
    The number of shares of the GPU that were asked for.
    """

    def error_message(self) -> str:
        return (
            f"{self.graphics_processor_shares} shares of the GPU were asked for, but "
            f"the machine has no GPU."
        )

    def suggest_correction(self) -> str:
        return (
            "Give the machine the memory of its GPU, or divide it into shares of the "
            "processor cores only."
        )


# %% workers


@dataclass(kw_only=True)
class NotEnoughMemoryForWorkers(DataclassException):
    """
    Raised when not even one worker of a pool fits into the free memory.
    """

    memory_per_worker: int
    """
    How many bytes each worker needs.
    """

    free_bytes: int
    """
    How many bytes the workers may use.
    """

    def error_message(self) -> str:
        return (
            f"A worker of {self.memory_per_worker / 1e9:.1f} GB needs more than the "
            f"{self.free_bytes / 1e9:.1f} GB the workers may use."
        )

    def suggest_correction(self) -> str:
        return "Raise the memory budget, or free memory first."
