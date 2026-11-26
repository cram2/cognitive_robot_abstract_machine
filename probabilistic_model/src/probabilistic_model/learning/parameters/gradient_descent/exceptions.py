"""
Exceptions of learning the parameters of a circuit by gradient descent.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

from probabilistic_model.exceptions import ProbabilisticModelException


# %% exceptions


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
