"""
Estimators of the mutual information between every pair of variables of a dataset, which
Chow-Liu trees are built on.
"""

from __future__ import annotations

# %% imports
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from random_events.variable import Continuous
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sortedcontainers import SortedSet

from probabilistic_model.causal_structure.discretization import Discretizer
from probabilistic_model.learning.structure.exceptions import (
    UnsupportedMutualInformationVariableError,
)


# %% estimators


@dataclass(kw_only=True)
class MutualInformationEstimator(ABC):
    """
    A way of estimating the mutual information in nats between every pair of columns of
    a dataset.
    """

    @abstractmethod
    def estimate(
        self, *, variables: SortedSet, data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        :param variables: The variables, one per column.
        :param data: The rows, encoded as the circuits read them.
        :return: The symmetric matrix of mutual information, zero on the diagonal.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class BinnedMutualInformation(MutualInformationEstimator):
    """
    The mutual information of the empirical distribution of the rows discretized by the
    :class:`~probabilistic_model.causal_structure.discretization.Discretizer`.
    """

    number_of_bins: int = 10
    """
    The number of quantile bins a continuous variable with many values is cut into.
    """

    def discretize_rows(
        self, *, variables: SortedSet, data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int64]:
        """
        :param variables: The variables, one per column.
        :param data: The rows, encoded as the circuits read them.
        :return: The level of every value, counted from zero per variable; every state
            of a symbolic variable is a level of its own.
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
        levels = Discretizer(bin_count=self.number_of_bins).fit_transform(frame)
        return levels.to_numpy(dtype=np.int64)

    def estimate(
        self, *, variables: SortedSet, data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        levels = self.discretize_rows(variables=variables, data=data)
        number_of_rows, number_of_variables = levels.shape
        number_of_levels = levels.max(axis=0) + 1
        offsets = np.concatenate([[0], np.cumsum(number_of_levels)])
        one_hot = np.zeros((number_of_rows, int(offsets[-1])), dtype=np.float32)
        one_hot[
            np.repeat(np.arange(number_of_rows), number_of_variables),
            (levels + offsets[:-1]).ravel(),
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


@dataclass(kw_only=True)
class GaussianMutualInformation(MutualInformationEstimator):
    """
    The mutual information of the Gaussian with the correlations of the rows,
    :math:`-\\frac{1}{2} \\log(1 - \\rho^2)` for a pair of correlation :math:`\\rho`.

    It measures linear dependence only and is defined for continuous variables only.
    """

    def estimate(
        self, *, variables: SortedSet, data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        :raises UnsupportedMutualInformationVariableError: If a variable is not
            continuous.
        """
        for variable in variables:
            if not isinstance(variable, Continuous):
                raise UnsupportedMutualInformationVariableError(
                    variable=variable, estimator=self
                )
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.nan_to_num(np.corrcoef(data, rowvar=False))
        # a perfect correlation has infinite information; keep it the largest finite one
        squared = np.minimum(np.atleast_2d(correlation) ** 2, np.nextafter(1.0, 0.0))
        result = -0.5 * np.log1p(-squared)
        np.fill_diagonal(result, 0.0)
        return result


@dataclass(kw_only=True)
class NearestNeighbourMutualInformation(MutualInformationEstimator):
    """
    The nearest neighbour estimate of the mutual information
    :cite:`kraskov2004estimating` as :mod:`sklearn.feature_selection` computes it, for
    symbolic and continuous variables alike, averaged over both directions of every
    pair.
    """

    neighbours: int = 3
    """
    The number of neighbours of every row the estimate counts in.
    """

    seed: int = 0
    """
    The seed of the noise added to continuous values to break ties.
    """

    def estimate(
        self, *, variables: SortedSet, data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        symbolic = np.array(
            [not isinstance(variable, Continuous) for variable in variables]
        )
        columns = data.copy()
        for index in np.flatnonzero(symbolic).tolist():
            columns[:, index] = np.unique(data[:, index], return_inverse=True)[1]
        directed = np.stack(
            [
                (mutual_info_classif if symbolic[index] else mutual_info_regression)(
                    columns,
                    columns[:, index],
                    discrete_features=symbolic,
                    n_neighbors=self.neighbours,
                    random_state=self.seed,
                )
                for index in range(len(variables))
            ]
        )
        result = (directed + directed.T) / 2.0
        np.fill_diagonal(result, 0.0)
        return result
