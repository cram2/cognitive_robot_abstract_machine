import unittest

import numpy as np
from random_events.interval import closed, reals
from random_events.variable import Continuous
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


class MultivariateGaussianArrayTestCase(unittest.TestCase):
    """
    Gaussians stacked into arrays answer like the single distributions they were made
    of.
    """

    def setUp(self):
        self.x, self.y, self.z = Continuous("x"), Continuous("y"), Continuous("z")
        self.distributions = [
            MultivariateGaussianDistribution(
                variables=(self.x, self.y, self.z),
                mean=np.array([0.0, 1.0, -1.0]),
                covariance=Covariance.from_matrix(
                    [[1.0, 0.6, 0.1], [0.6, 2.0, -0.3], [0.1, -0.3, 0.5]]
                ),
            ),
            MultivariateGaussianDistribution(
                variables=(self.x, self.y, self.z),
                mean=np.array([2.0, -1.0, 0.5]),
                covariance=Covariance.from_matrix(
                    [[0.5, -0.2, 0.0], [-0.2, 0.8, 0.4], [0.0, 0.4, 3.0]]
                ),
            ),
        ]
        self.points = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [2.5, 1.0, -1.0]])
        self.gaussians = MultivariateGaussianArray.from_distributions(
            self.distributions, [self.x, self.y, self.z]
        )

    def test_from_distributions_lays_the_parameters_out_in_the_given_order(self):
        reordered = MultivariateGaussianArray.from_distributions(
            self.distributions, [self.z, self.x, self.y]
        )
        order = [2, 0, 1]
        for index, distribution in enumerate(self.distributions):
            np.testing.assert_array_equal(
                reordered.mean[index], distribution.mean[order]
            )
            np.testing.assert_array_equal(
                reordered.covariance.matrices[index],
                distribution.covariance.matrix[np.ix_(order, order)],
            )

    def test_log_density_is_the_log_likelihood_of_every_distribution(self):
        expected = np.stack(
            [
                distribution.log_likelihood(self.points)
                for distribution in self.distributions
            ],
            axis=1,
        )
        np.testing.assert_allclose(self.gaussians.log_density(self.points), expected)

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
            for distribution in self.distributions
        ]
        np.testing.assert_allclose(
            self.gaussians.probability_of_hyperrectangles(hyperrectangles),
            expected,
            atol=1e-4,
        )

    def test_log_probability_of_hyperrectangles_far_in_the_tail(self):
        lower, upper = 9.0, 10.0
        hyperrectangles = HyperrectangleArray.of_simple_intervals(
            [
                interval.simple_sets[0]
                for interval in (closed(lower, upper), reals(), reals())
            ]
        ).broadcast_to(2)
        expected = []
        for distribution in self.distributions:
            mean = distribution.mean[0]
            deviation = np.sqrt(distribution.covariance.matrix[0, 0])
            expected.append(
                np.log(
                    norm.sf(lower, mean, deviation) - norm.sf(upper, mean, deviation)
                )
            )
        np.testing.assert_allclose(
            self.gaussians.log_probability_of_hyperrectangles(hyperrectangles),
            expected,
        )

    def test_marginal_is_the_marginal_of_every_distribution(self):
        marginal = self.gaussians.marginal(np.array([0, 2]))
        for index, distribution in enumerate(self.distributions):
            expected = distribution.marginal([self.x, self.z])
            np.testing.assert_allclose(marginal.mean[index], expected.mean)
            np.testing.assert_allclose(
                marginal.covariance.matrices[index], expected.covariance.matrix
            )

    def test_conditional_is_the_conditional_of_every_distribution(self):
        conditional = self.gaussians.conditional(
            np.array([1]), np.array([0, 2]), np.array([0.3])
        )
        for index, distribution in enumerate(self.distributions):
            expected, _ = distribution.log_conditional({self.y: 0.3})
            gaussian = expected.marginal([self.x, self.z])
            np.testing.assert_allclose(conditional.mean[index], gaussian.mean)
            np.testing.assert_allclose(
                conditional.covariance.matrices[index], gaussian.covariance.matrix
            )


if __name__ == "__main__":
    unittest.main()
