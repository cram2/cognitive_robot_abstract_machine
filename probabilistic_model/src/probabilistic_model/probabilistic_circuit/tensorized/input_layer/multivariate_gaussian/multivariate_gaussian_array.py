from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from random_events.variable import Variable
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal
from typing_extensions import Self, Sequence

from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeScopeValues,
    NodeValues,
    SampleNodeValues,
    SampleScopeValues,
    ScopeValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.covariance_array import (
    CovarianceArray,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.hyperrectangle_array import (
    HyperrectangleArray,
)


@dataclass
class MultivariateGaussianArray:
    """
    Gaussians over the same variables, one per node of a layer.
    """

    mean: NodeScopeValues
    """
    The mean of every Gaussian.
    """

    covariance: CovarianceArray
    """
    The covariance of every Gaussian.
    """

    @classmethod
    def from_distributions(
        cls,
        distributions: Sequence[MultivariateGaussianDistribution],
        variables: Sequence[Variable],
    ) -> Self:
        """
        :param distributions: Gaussians over the same variables, in any order.
        :param variables: Those variables, in the order to lay the parameters out in.
        :return: The Gaussians, one per distribution.
        """
        means, covariances = [], []
        for distribution in distributions:
            order = [distribution.index_of(variable) for variable in variables]
            means.append(distribution.mean[order])
            covariances.append(distribution.covariance.marginal(order))
        return cls(np.array(means), CovarianceArray.from_covariances(covariances))

    @property
    def number_of_gaussians(self) -> int:
        """
        :return: How many Gaussians there are.
        """
        return len(self.mean)

    @property
    def dimension(self) -> int:
        """
        :return: How many variables every Gaussian is over.
        """
        return self.mean.shape[1]

    def log_density(self, values: SampleScopeValues) -> SampleNodeValues:
        """
        :param values: Points over the variables of the Gaussians.
        :return: The log-density of every Gaussian at every point, shape (#points,
            #Gaussians).
        """
        cholesky = np.linalg.cholesky(self.covariance.matrices)
        # whiten by every node at once: one matrix product of the points with the
        # stacked inverse Cholesky factors, shape (#points, #nodes * #variables)
        whitening = np.linalg.inv(cholesky)
        dimension = self.dimension
        stacked = whitening.reshape(-1, dimension)
        whitened_means = np.einsum("nij,nj->ni", whitening, self.mean).reshape(-1)
        whitened = (values @ stacked.T - whitened_means).reshape(
            len(values), self.number_of_gaussians, dimension
        )
        log_determinant = 2 * np.log(np.diagonal(cholesky, axis1=1, axis2=2)).sum(
            axis=1
        )
        return -0.5 * (
            np.einsum("snd,snd->sn", whitened, whitened)
            + dimension * math.log(2 * math.pi)
            + log_determinant
        )

    def probability_of_hyperrectangles(
        self, hyperrectangles: HyperrectangleArray
    ) -> NodeValues:
        """
        :param hyperrectangles: One hyperrectangle per Gaussian.
        :return: The probability of every hyperrectangle under its Gaussian.
        """
        return np.exp(self.log_probability_of_hyperrectangles(hyperrectangles))

    def log_probability_of_hyperrectangles(
        self, hyperrectangles: HyperrectangleArray
    ) -> NodeValues:
        """
        The probability of an axis-aligned hyperrectangle under a correlated Gaussian
        has no closed form, so it is integrated numerically by
        :mod:`scipy.stats.multivariate_normal` over the variables the hyperrectangle
        bounds. A hyperrectangle that bounds only one variable is answered in closed
        form for all Gaussians at once, in log space so that hyperrectangles far in the
        tails keep a probability above zero.

        :param hyperrectangles: One hyperrectangle per Gaussian.
        :return: The log-probability of every hyperrectangle under its Gaussian.
        """
        result = np.full(self.number_of_gaussians, -np.inf)
        possible = (hyperrectangles.lower < hyperrectangles.upper).all(axis=1)
        bounded = np.isfinite(hyperrectangles.lower) | np.isfinite(
            hyperrectangles.upper
        )

        for pattern in np.unique(bounded[possible], axis=0):
            nodes = possible & (bounded == pattern).all(axis=1)
            positions = np.flatnonzero(pattern)
            if len(positions) == 0:
                result[nodes] = 0.0
            elif len(positions) == 1:
                [position] = positions
                mean = self.mean[nodes, position]
                deviation = np.sqrt(self.covariance.variances[nodes, position])
                result[nodes] = self.log_standard_normal_probability_between(
                    (hyperrectangles.lower[nodes, position] - mean) / deviation,
                    (hyperrectangles.upper[nodes, position] - mean) / deviation,
                )
            else:
                for node in np.flatnonzero(nodes):
                    probability = multivariate_normal(
                        self.mean[node, positions],
                        self.covariance.matrices[node][np.ix_(positions, positions)],
                    ).cdf(
                        hyperrectangles.upper[node, positions],
                        lower_limit=hyperrectangles.lower[node, positions],
                    )
                    with np.errstate(divide="ignore"):
                        result[node] = np.log(np.clip(probability, 0.0, 1.0))
        return np.minimum(result, 0.0)

    @staticmethod
    def log_standard_normal_probability_between(
        lower: NodeValues, upper: NodeValues
    ) -> NodeValues:
        """
        :param lower: The lower end of one interval per Gaussian, in standard deviations
            from its mean.
        :param upper: The upper end of every interval, greater than its lower end.
        :return: The log-probability of every interval under the standard normal.
        """
        # an interval in the upper tail is mirrored into the lower tail, where the
        # log of the cumulative distribution is precise
        mirrored = lower > 0
        lower, upper = np.where(mirrored, -upper, lower), np.where(
            mirrored, -lower, upper
        )
        log_upper = log_ndtr(upper)
        difference = log_ndtr(lower) - log_upper
        # log(1 - exp(difference)), precise both near zero and far below it
        with np.errstate(divide="ignore"):
            log_one_minus = np.where(
                difference > -math.log(2),
                np.log(-np.expm1(difference)),
                np.log1p(-np.exp(difference)),
            )
        return log_upper + log_one_minus

    def marginal(self, positions: NodeIndices) -> Self:
        """
        :param positions: Positions of variables of the Gaussians.
        :return: The Gaussians over only those variables.
        """
        return type(self)(self.mean[:, positions], self.covariance.marginal(positions))

    def conditional(
        self, fixed: NodeIndices, free: NodeIndices, values: ScopeValues
    ) -> Self:
        """
        :param fixed: The positions of the variables held at a value.
        :param free: The positions of the other variables.
        :param values: What the fixed variables are held at.
        :return: The Gaussians over the free variables, conditioned on the fixed ones.
        """
        free_with_fixed = self.covariance.between(free, fixed)
        gain = np.swapaxes(
            np.linalg.solve(
                self.covariance.between(fixed, fixed),
                np.swapaxes(free_with_fixed, 1, 2),
            ),
            1,
            2,
        )
        return type(self)(
            self.mean[:, free]
            + np.einsum("nij,nj->ni", gain, values - self.mean[:, fixed]),
            CovarianceArray.from_matrices(
                self.covariance.between(free, free)
                - gain @ np.swapaxes(free_with_fixed, 1, 2)
            ),
        )
