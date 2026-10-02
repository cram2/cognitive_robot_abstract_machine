import unittest

import numpy as np
from random_events.interval import closed, reals
from random_events.variable import Continuous
from scipy.stats import multivariate_normal

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

x, y, z = Continuous("x"), Continuous("y"), Continuous("z")

DISTRIBUTIONS = [
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

POINTS = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [2.5, 1.0, -1.0]])


class MultivariateGaussianArrayTestCase(unittest.TestCase):
    """
    Gaussians stacked into arrays answer like the single distributions they were made
    of.
    """

    def setUp(self):
        self.gaussians = MultivariateGaussianArray.from_distributions(
            DISTRIBUTIONS, [x, y, z]
        )

    def test_from_distributions_lays_the_parameters_out_in_the_given_order(self):
        reordered = MultivariateGaussianArray.from_distributions(
            DISTRIBUTIONS, [z, x, y]
        )
        order = [2, 0, 1]
        for index, distribution in enumerate(DISTRIBUTIONS):
            np.testing.assert_array_equal(
                reordered.mean[index], distribution.mean[order]
            )
            np.testing.assert_array_equal(
                reordered.covariance.matrices[index],
                distribution.covariance.matrix[np.ix_(order, order)],
            )

    def test_log_density_is_the_log_likelihood_of_every_distribution(self):
        expected = np.stack(
            [distribution.log_likelihood(POINTS) for distribution in DISTRIBUTIONS],
            axis=1,
        )
        np.testing.assert_allclose(self.gaussians.log_density(POINTS), expected)

    def test_probability_of_hyperrectangles(self):
        hyperrectangles = HyperrectangleArray.of_simple_intervals(
            [
                interval.simple_sets[0]
                for interval in (closed(-1.0, 1.0), closed(-2.0, 0.5), reals())
            ]
        ).broadcast_to(2)
        expected = [
            multivariate_normal(
                distribution.mean[:2], distribution.covariance.matrix[:2, :2]
            ).cdf([1.0, 0.5], lower_limit=[-1.0, -2.0])
            for distribution in DISTRIBUTIONS
        ]
        np.testing.assert_allclose(
            self.gaussians.probability_of_hyperrectangles(hyperrectangles),
            expected,
            atol=1e-4,
        )

    def test_marginal_is_the_marginal_of_every_distribution(self):
        marginal = self.gaussians.marginal(np.array([0, 2]))
        for index, distribution in enumerate(DISTRIBUTIONS):
            expected = distribution.marginal([x, z])
            np.testing.assert_allclose(marginal.mean[index], expected.mean)
            np.testing.assert_allclose(
                marginal.covariance.matrices[index], expected.covariance.matrix
            )

    def test_conditional_is_the_conditional_of_every_distribution(self):
        conditional = self.gaussians.conditional(
            np.array([1]), np.array([0, 2]), np.array([0.3])
        )
        for index, distribution in enumerate(DISTRIBUTIONS):
            expected, _ = distribution.log_conditional({y: 0.3})
            gaussian = expected.marginal([x, z])
            np.testing.assert_allclose(conditional.mean[index], gaussian.mean)
            np.testing.assert_allclose(
                conditional.covariance.matrices[index], gaussian.covariance.matrix
            )


if __name__ == "__main__":
    unittest.main()
