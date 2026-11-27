"""
How much memory one training step needs beyond what is already in use, so that mini-
batches can be sized to fit the free memory of a device.
"""

from __future__ import annotations

# %% imports
import math
from dataclasses import dataclass

from probabilistic_model.learning.parameters.exceptions import (
    UnmeasurableStepMemoryError,
)


# %% memory of a training step


@dataclass(kw_only=True)
class ProbeMeasurement:
    """
    The memory one training step needed on a batch of a given number of rows.
    """

    rows: int
    """
    The number of rows of the batch.
    """

    step_bytes: int
    """
    The bytes the step allocated beyond what was in use before it.
    """


@dataclass(kw_only=True)
class StepMemory:
    """
    The memory one training step allocates beyond what is already in use, as a fixed
    part and a part that grows with the rows of the batch.
    """

    fixed_bytes: float
    """
    The bytes a step allocates whatever the size of its batch.
    """

    bytes_per_row: float
    """
    The bytes a step allocates for every row of its batch.
    """

    @classmethod
    def from_two_probes(
        cls, *, smaller: ProbeMeasurement, larger: ProbeMeasurement
    ) -> StepMemory:
        """
        :param smaller: The step on the batch with fewer rows.
        :param larger: The step on the batch with more rows.
        :return: The line through both measurements.
        :raises UnmeasurableStepMemoryError: If the larger batch did not need more
            memory.
        """
        bytes_per_row = (larger.step_bytes - smaller.step_bytes) / (
            larger.rows - smaller.rows
        )
        if bytes_per_row <= 0:
            raise UnmeasurableStepMemoryError(
                smaller_rows=smaller.rows, bytes_per_row=bytes_per_row
            )
        return cls(
            fixed_bytes=smaller.step_bytes - bytes_per_row * smaller.rows,
            bytes_per_row=bytes_per_row,
        )

    def rows_fitting_in(self, *, free_bytes: int) -> int:
        """
        :param free_bytes: The memory a step may allocate.
        :return: The most rows a batch can have for its step to fit.
        """
        return math.floor((free_bytes - self.fixed_bytes) / self.bytes_per_row)
