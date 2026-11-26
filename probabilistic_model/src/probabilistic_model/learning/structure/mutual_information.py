"""
The empirical mutual information between every pair of variables of a dataset, which
Chow-Liu trees are built on.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import numpy as np
from random_events.variable import Continuous, Variable
from sortedcontainers import SortedSet


# %% discretized data


@dataclass(kw_only=True)
class DiscretizedData:
    """
    The rows of a dataset with every value replaced by the index of its state: the value
    itself for a discrete variable, the quantile bin for a continuous one.
    """

    variables: SortedSet
    """
    The variables, one per column.
    """

    data: np.ndarray
    """
    The rows, encoded as the circuits read them.
    """

    number_of_bins: int = 10
    """
    The number of equally populated bins a continuous variable is cut into.
    """

    states: np.ndarray = field(init=False)
    """
    The state index of every value, shape (#rows, #variables).
    """

    number_of_states: np.ndarray = field(init=False)
    """
    The number of states of every variable.
    """

    def __post_init__(self) -> None:
        """
        Replaces every value of the data by the index of its state.
        """
        columns = [
            self.state_column(variable=variable, values=self.data[:, index])
            for index, variable in enumerate(self.variables)
        ]
        self.states = np.stack(columns, axis=1).astype(np.int64)
        self.number_of_states = np.maximum(self.states.max(axis=0) + 1, 1)

    def state_column(self, *, variable: Variable, values: np.ndarray) -> np.ndarray:
        """
        :param variable: The variable of the column.
        :param values: The values of the column.
        :return: The state index of every value.
        """
        if isinstance(variable, Continuous):
            edges = np.unique(
                np.quantile(values, np.linspace(0, 1, self.number_of_bins + 1)[1:-1])
            )
            return np.searchsorted(edges, values, side="right")
        return np.unique(values, return_inverse=True)[1]


# %% mutual information


@dataclass(kw_only=True)
class PairwiseMutualInformation:
    """
    The mutual information in nats between the empirical distributions of every pair of
    columns of discretized data.
    """

    discretized: DiscretizedData
    """
    The data.
    """

    def compute(self) -> np.ndarray:
        """
        :return: The symmetric matrix of mutual information, zero on the diagonal.
        """
        states = self.discretized.states
        number_of_rows = len(states)
        offsets = np.concatenate([[0], np.cumsum(self.discretized.number_of_states)])
        one_hot = np.zeros((number_of_rows, int(offsets[-1])), dtype=np.float32)
        one_hot[
            np.repeat(np.arange(number_of_rows), states.shape[1]),
            (states + offsets[:-1]).ravel(),
        ] = 1.0
        joint = (one_hot.T @ one_hot).astype(float) / number_of_rows
        marginal = np.diag(joint)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                joint > 0, joint * np.log(joint / np.outer(marginal, marginal)), 0.0
            )
        number_of_variables = len(self.discretized.number_of_states)
        state_of_variable = np.zeros((int(offsets[-1]), number_of_variables))
        state_of_variable[
            np.arange(int(offsets[-1])),
            np.repeat(
                np.arange(number_of_variables), self.discretized.number_of_states
            ),
        ] = 1.0
        result = state_of_variable.T @ terms @ state_of_variable
        np.fill_diagonal(result, 0.0)
        return result
