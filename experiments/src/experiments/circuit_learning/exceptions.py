"""
Exceptions of the circuit learning benchmark.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from krrood.exceptions import DataclassException

from experiments.circuit_learning.records import BenchmarkRun

# %% exceptions


@dataclass(kw_only=True)
class NoSlotForRunError(DataclassException, RuntimeError):
    """
    Exception raised when a run needs a kind of share of the machine of which the
    benchmark has none, so it could never start.
    """

    runs: list[BenchmarkRun]
    """
    The runs that could never start.
    """

    def error_message(self) -> str:
        keys = ", ".join(run.key for run in self.runs)
        return f"No share of the machine can carry out the runs {keys}."

    def suggest_correction(self) -> str:
        return (
            "Give the benchmark at least one share of the GPU for runs on the GPU and "
            "one share of the processor cores for runs on the processor cores."
        )
