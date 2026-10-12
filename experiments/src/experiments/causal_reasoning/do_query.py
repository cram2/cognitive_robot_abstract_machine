"""
What asking a do() question of a circuit yields, whatever the question was about.

A question marks one cause and one effect, and the effect is read off every region of
the cause twice: once by conditioning alone and once with backdoor adjustment. These are
the types that hold the two answers side by side.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Tuple


@dataclass(frozen=True)
class CauseRegion:
    """
    One region of the cause, with how likely the effect is on it.
    """

    description: str
    """
    The values of the cause this region covers.
    """

    probability: float
    """
    How much of the training population the region accounts for.
    """

    conditioned_probability: float
    """
    How likely the effect is given the region, read off the circuit with no adjustment.
    """

    adjusted_probability: float
    """
    How likely the effect is under an intervention setting the cause to the region,
    adjusted for the question's confounders.
    """

    @property
    def shift_from_adjusting(self) -> float:
        """
        How far adjusting moves the answer away from plain conditioning.
        """
        return self.adjusted_probability - self.conditioned_probability


@dataclass(frozen=True)
class DoQueryAnswer:
    """
    What one question yielded on one fitted circuit.
    """

    asked: str
    """
    The question, in words.
    """

    training_example_count: int
    """
    How many examples the circuit was fitted on.
    """

    regions: Tuple[CauseRegion, ...]
    """
    One entry per region of the cause the grounded circuit's support covers.
    """

    @property
    def largest_shift_from_adjusting(self) -> float:
        """
        The largest distance between a conditioned and an adjusted answer over the
        regions: how much the confounders mattered at all.
        """
        return max(abs(region.shift_from_adjusting) for region in self.regions)
