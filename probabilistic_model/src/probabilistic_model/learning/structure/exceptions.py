"""
Exceptions of learning the structure of a circuit from data.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from random_events.variable import Variable

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


@dataclass(kw_only=True)
class UnsupportedStructureVariableError(ProbabilisticModelException, TypeError):
    """
    Exception raised when a structure learner meets a variable it has no input
    distribution for.
    """

    variable: Variable
    """
    The variable.
    """

    def error_message(self) -> str:
        return (
            f"The variable {self.variable.name} of type {type(self.variable).__name__} "
            f"has no input distribution in a learned structure."
        )

    def suggest_correction(self) -> str:
        return "Use symbolic or continuous variables."
