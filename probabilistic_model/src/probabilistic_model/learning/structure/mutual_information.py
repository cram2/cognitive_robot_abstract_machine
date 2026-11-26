"""
The empirical mutual information between every pair of variables of a dataset, which
Chow-Liu trees are built on.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from random_events.variable import Continuous
from sortedcontainers import SortedSet
from typing_extensions import Self

from probabilistic_model.causal_structure.discretization import Discretizer


# %% mutual information


@dataclass(kw_only=True)
class PairwiseMutualInformation:
    """
    The mutual information in nats between the empirical distributions of every pair of
    columns of discretized data.
    """

    levels: npt.NDArray[np.int64]
    """
    The level of every value, shape (#rows, #variables), counted from zero per variable.
    """

    @classmethod
    def from_rows(
        cls,
        *,
        variables: SortedSet,
        data: npt.NDArray[np.float64],
        number_of_bins: int,
    ) -> Self:
        """
        :param variables: The variables, one per column.
        :param data: The rows, encoded as the circuits read them.
        :param number_of_bins: The number of quantile bins a continuous variable with
            many values is cut into.
        :return: The mutual information of the rows, every value of a symbolic variable
            its own level.
        """
        frame = pd.DataFrame(
            {
                variable.name: (
                    data[:, index]
                    if isinstance(variable, Continuous)
                    else data[:, index].astype(str)
                )
                for index, variable in enumerate(variables)
            }
        )
        levels = Discretizer(bin_count=number_of_bins).fit_transform(frame)
        return cls(levels=levels.to_numpy(dtype=np.int64))

    def compute(self) -> npt.NDArray[np.float64]:
        """
        :return: The symmetric matrix of mutual information, zero on the diagonal.
        """
        number_of_rows, number_of_variables = self.levels.shape
        number_of_levels = self.levels.max(axis=0) + 1
        offsets = np.concatenate([[0], np.cumsum(number_of_levels)])
        one_hot = np.zeros((number_of_rows, int(offsets[-1])), dtype=np.float32)
        one_hot[
            np.repeat(np.arange(number_of_rows), number_of_variables),
            (self.levels + offsets[:-1]).ravel(),
        ] = 1.0
        joint = (one_hot.T @ one_hot).astype(float) / number_of_rows
        marginal = np.diag(joint)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(
                joint > 0, joint * np.log(joint / np.outer(marginal, marginal)), 0.0
            )
        level_of_variable = np.zeros((int(offsets[-1]), number_of_variables))
        level_of_variable[
            np.arange(int(offsets[-1])),
            np.repeat(np.arange(number_of_variables), number_of_levels),
        ] = 1.0
        result = level_of_variable.T @ terms @ level_of_variable
        np.fill_diagonal(result, 0.0)
        return result
