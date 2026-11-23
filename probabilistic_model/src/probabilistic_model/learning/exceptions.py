"""
Exceptions shared by the methods that fit circuits on data.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

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
