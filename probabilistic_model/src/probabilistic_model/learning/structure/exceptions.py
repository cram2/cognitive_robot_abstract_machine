"""
Exceptions of learning the structure of a circuit from data.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from random_events.variable import Variable

from probabilistic_model.exceptions import ProbabilisticModelException

if TYPE_CHECKING:
    from probabilistic_model.learning.structure.mutual_information import (
        MutualInformationEstimator,
    )


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


@dataclass(kw_only=True)
class UnsupportedMutualInformationVariableError(ProbabilisticModelException, TypeError):
    """
    Exception raised when an estimator of the mutual information meets a variable it
    cannot estimate for.
    """

    variable: Variable
    """
    The variable.
    """

    estimator: MutualInformationEstimator
    """
    The estimator.
    """

    def error_message(self) -> str:
        return (
            f"{type(self.estimator).__name__} cannot estimate the mutual information of "
            f"the variable {self.variable.name} of type {type(self.variable).__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            "Use an estimator that supports every kind of variable, such as the "
            "binned or the nearest neighbour one."
        )
