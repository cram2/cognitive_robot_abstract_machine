"""
Gaussians stacked into arrays answer like the single distributions they were made of.
"""

from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import closed, reals
from scipy.stats import multivariate_normal, norm

from probabilistic_model.distributions.multivariate_gaussian import (
    Covariance,
    MultivariateGaussianDistribution,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.hyperrectangle_array import (
    HyperrectangleArray,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.multivariate_gaussian_array import (
    MultivariateGaussianArray,
)

# %% fixtures


@pytest.fixture
def distributions(x, y, z) -> list[MultivariateGaussianDistribution]:
    """
    Two correlated Gaussians over *x*, *y* and *z*.
    """
    return [
        MultivariateGaussianDistribution(
            variables=(x, y, z),
            mean=np.array([0.0, 1.0, -1.0]),
            covariance=Covariance.from_matrix(
                [[1.0, 0.6, 0.1], [0.6, 2.0, -0.3], [0.1, -0.3, 0.5]]
            ),
        ),
        MultivariateGaussianDistribution(
            variables=(x, y, z),
            mean=np.array([2.0, -1.0, 0.5]),
            covariance=Covariance.from_matrix(
                [[0.5, -0.2, 0.0], [-0.2, 0.8, 0.4], [0.0, 0.4, 3.0]]
            ),
        ),
    ]


@pytest.fixture
def gaussians(distributions, x, y, z) -> MultivariateGaussianArray:
    """
    :func:`distributions` stacked into an array.
    """
    return MultivariateGaussianArray.from_distributions(distributions, [x, y, z])


def hyperrectangles_of(*intervals) -> HyperrectangleArray:
    """
    :return: The hyperrectangle one interval per variable spans, once per Gaussian.
    """
    return HyperrectangleArray.of_simple_intervals(
        [interval.simple_sets[0] for interval in intervals]
    ).broadcast_to(2)


# %% Gaussian arrays


def test_from_distributions_lays_the_parameters_out_in_the_given_order(
    distributions, x, y, z
):
    reordered = MultivariateGaussianArray.from_distributions(distributions, [z, x, y])
    order = [2, 0, 1]
    for index, distribution in enumerate(distributions):
        np.testing.assert_array_equal(reordered.mean[index], distribution.mean[order])
        np.testing.assert_array_equal(
            reordered.covariance.matrices[index],
            distribution.covariance.matrix[np.ix_(order, order)],
        )


def test_log_density_is_the_log_likelihood_of_every_distribution(
    gaussians, distributions
):
    points = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [2.5, 1.0, -1.0]])
    expected = np.stack(
        [distribution.log_likelihood(points) for distribution in distributions], axis=1
    )
    np.testing.assert_allclose(gaussians.log_density(points), expected)


def test_probability_of_hyperrectangles(gaussians, distributions):
    hyperrectangles = hyperrectangles_of(closed(-1.0, 1.0), closed(-2.0, 0.5), reals())
    expected = [
        multivariate_normal(
            distribution.mean[:2], distribution.covariance.matrix[:2, :2]
        ).cdf([1.0, 0.5], lower_limit=[-1.0, -2.0])
        for distribution in distributions
    ]
    np.testing.assert_allclose(
        gaussians.probability_of_hyperrectangles(hyperrectangles), expected, atol=1e-4
    )


def test_log_probability_of_hyperrectangles_far_in_the_tail(gaussians, distributions):
    lower, upper = 9.0, 10.0
    hyperrectangles = hyperrectangles_of(closed(lower, upper), reals(), reals())
    expected = []
    for distribution in distributions:
        mean = distribution.mean[0]
        deviation = np.sqrt(distribution.covariance.matrix[0, 0])
        expected.append(
            np.log(norm.sf(lower, mean, deviation) - norm.sf(upper, mean, deviation))
        )
    np.testing.assert_allclose(
        gaussians.log_probability_of_hyperrectangles(hyperrectangles), expected
    )


def test_marginal_is_the_marginal_of_every_distribution(gaussians, distributions, x, z):
    marginal = gaussians.marginal(np.array([0, 2]))
    for index, distribution in enumerate(distributions):
        expected = distribution.marginal([x, z])
        np.testing.assert_allclose(marginal.mean[index], expected.mean)
        np.testing.assert_allclose(
            marginal.covariance.matrices[index], expected.covariance.matrix
        )


def test_conditional_is_the_conditional_of_every_distribution(
    gaussians, distributions, x, y, z
):
    conditional = gaussians.conditional(
        np.array([1]), np.array([0, 2]), np.array([0.3])
    )
    for index, distribution in enumerate(distributions):
        expected = distribution.log_conditional({y: 0.3}).distribution
        expected = expected.marginal([x, z])
        np.testing.assert_allclose(conditional.mean[index], expected.mean)
        np.testing.assert_allclose(
            conditional.covariance.matrices[index], expected.covariance.matrix
        )
