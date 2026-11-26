"""
Exceptions of learning the structure of a circuit from data.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from probabilistic_model.exceptions import ProbabilisticModelException


# %% exceptions


@dataclass(kw_only=True)
class NoRowsToFitError(ProbabilisticModelException, ValueError):
    """
    Exception raised when a circuit is to be fitted on a dataframe without rows.
    """

    def error_message(self) -> str:
        return "The dataframe to fit on has no rows."

    def suggest_correction(self) -> str:
        return "Fit on at least one row."
