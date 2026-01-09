import unittest

from probabilistic_model.distributions.gaussian import *


class GaussianDistributionTestCase(unittest.TestCase):
    x = Continuous("x")
    distribution: GaussianDistribution = GaussianDistribution(x, location=2, scale=2)

    def test_domain(self):
        self.assertEqual(self.distribution.univariate_support, reals())

    def test_probability_of_domain(self):
        self.assertEqual(self.distribution.probability(self.distribution.support), 1)

    def test_mode(self):
        mode, likelihood = self.distribution.univariate_log_mode()
        self.assertEqual(mode, singleton(2))

    def test_sample(self):
        samples = self.distribution.sample(100)
        self.assertEqual(len(samples), 100)
        likelihoods = self.distribution.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))

    def test_conditional_simple_intersection(self):
        event = SimpleEvent(
            {self.distribution.variable: closed(1, 2)}
        ).as_composite_set()
        conditional, probability = self.distribution.truncated(event)
        self.assertIsInstance(conditional, TruncatedGaussianDistribution)
        cdf = self.distribution.cdf(np.array([1, 2]).reshape(-1, 1))
        self.assertAlmostEqual(probability, cdf[1] - cdf[0])
        self.assertEqual(conditional.lower, 1)
        self.assertEqual(conditional.upper, 2)

    def test_raw_moment(self):
        self.assertEqual(self.distribution.raw_moment(0), 1)
        self.assertEqual(self.distribution.raw_moment(1), self.distribution.location)
        self.assertEqual(
            self.distribution.raw_moment(2),
            self.distribution.location**2 + self.distribution.scale**2,
        )

    def test_centered_moment(self):
        expectation = self.distribution.moment(
            VariableMap({self.distribution.variable: 1}),
            VariableMap({self.distribution.variable: 0}),
        )
        self.assertEqual(expectation[self.distribution.variable], 2)
        variance = self.distribution.moment(
            VariableMap({self.distribution.variable: 2}), expectation
        )
        self.assertEqual(variance[self.distribution.variable], 4)

        third_order_moment = self.distribution.moment(
            VariableMap({self.distribution.variable: 3}), expectation
        )
        self.assertEqual(third_order_moment[self.distribution.variable], 0)

        fourth_order_moment = self.distribution.moment(
            VariableMap({self.distribution.variable: 4}), expectation
        )
        self.assertEqual(fourth_order_moment[self.distribution.variable], 48)

    def test_moment_with_different_center_than_expectation(self):
        center = VariableMap({self.distribution.variable: 2.5})

        expectation = self.distribution.moment(
            VariableMap({self.distribution.variable: 1}), center
        )
        self.assertEqual(expectation[self.distribution.variable], -0.5)
        variance = self.distribution.moment(
            VariableMap({self.distribution.variable: 2}), center
        )
        self.assertEqual(variance[self.distribution.variable], 4.25)

        third_order_moment = self.distribution.moment(
            VariableMap({self.distribution.variable: 3}), center
        )
        self.assertEqual(third_order_moment[self.distribution.variable], -6.125)

    def test_likelihood_shape(self):
        samples = self.distribution.sample(100)
        self.assertEqual(samples.shape, (100, 1))
        likelihoods = self.distribution.log_likelihood(samples)
        self.assertEqual(likelihoods.shape, (100,))

    def test_plot(self):
        fig = go.Figure(self.distribution.plot())
        self.assertIsNotNone(fig)
        # fig.show()

    def test_serialization(self):
        serialized = self.distribution.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertEqual(self.distribution, deserialized)
        self.assertIsInstance(deserialized, GaussianDistribution)

    def test_variance(self):
        variance = self.distribution.variance([self.x])
        self.assertEqual(
            variance[self.distribution.variable], self.distribution.scale**2
        )


class TruncatedGaussianDistributionTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    distribution: TruncatedGaussianDistribution

    def setUp(self):
        self.distribution = TruncatedGaussianDistribution(
            self.x, SimpleInterval(-2, 2, Bound.CLOSED, Bound.CLOSED), 2, 2.0
        )

    def test_init(self):
        self.assertEqual(self.distribution.location, 2.0)

    def test_normalization_constant(self):
        normal_distribution = GaussianDistribution(
            self.x, location=self.distribution.location, scale=self.distribution.scale
        )
        self.assertAlmostEqual(
            normal_distribution.probability(self.distribution.support),
            self.distribution.normalizing_constant,
        )

    def test_cdf(self):
        cdf = self.distribution.cdf(np.array([0, 3, -3]).reshape(-1, 1))
        self.assertAlmostEqual(cdf[0], 0.285, places=3)
        self.assertEqual(cdf[1], 1)
        self.assertEqual(cdf[2], 0)

    def test_mode(self):
        mode, likelihood = self.distribution.univariate_log_mode()
        self.assertEqual(mode, singleton(2))
        self.assertEqual(
            likelihood,
            self.distribution.log_likelihood_without_bounds_check(np.array([[2]])),
        )

    def test_raw_moment(self):
        expectation = self.distribution.moment(
            VariableMap({self.x: 1}), VariableMap({self.x: 0})
        )
        self.assertAlmostEqual(expectation[self.x], 0.5544205, places=7)

    def test_centered_moment(self):
        gauss_distribution: GaussianDistribution = GaussianDistribution(
            self.x, location=0, scale=1
        )
        beta = (
            self.distribution.upper - self.distribution.location
        ) / self.distribution.scale
        alpha = (
            self.distribution.lower - self.distribution.location
        ) / self.distribution.scale
        center = VariableMap({self.x: self.distribution.location})
        expectation = self.distribution.moment(VariableMap({self.x: 1}), center)
        likelihood_alpha, likelihood_beta = gauss_distribution.likelihood(
            np.array([[alpha], [beta]])
        )
        offset_term = (
            -self.distribution.scale
            * (likelihood_beta - likelihood_alpha)
            / self.distribution.normalizing_constant
        )
        self.assertAlmostEqual(
            expectation[self.distribution.variable], 0 + offset_term, places=7
        )

    def test_moment_with_different_center_than_expectation(self):
        center = VariableMap({self.distribution.variable: 2.5})
        expectation = self.distribution.moment(
            VariableMap({self.distribution.variable: 1}),
            VariableMap({self.distribution.variable: 0}),
        )
        first_moment = self.distribution.moment(
            VariableMap({self.distribution.variable: 1}), center
        )
        self.assertAlmostEqual(
            first_moment[self.distribution.variable],
            expectation[self.distribution.variable]
            - center[self.distribution.variable],
            places=7,
        )
        second_moment = self.distribution.moment(
            VariableMap({self.distribution.variable: 2}), center
        )
        second_raw_moment = self.distribution.moment(
            VariableMap({self.distribution.variable: 2}),
            VariableMap({self.distribution.variable: 0}),
        )
        self.assertAlmostEqual(
            second_moment[self.distribution.variable],
            second_raw_moment[self.distribution.variable]
            + center[self.distribution.variable] ** 2
            - 2
            * center[self.distribution.variable]
            * expectation[self.distribution.variable],
            places=7,
        )

    def test_conditional_simple_intersection(self):
        event = SimpleEvent({self.x: closed(1, 2)}).as_composite_set()
        conditional, probability = self.distribution.truncated(event)
        self.assertIsInstance(conditional, TruncatedGaussianDistribution)
        cdf = self.distribution.cdf(np.array([1, 2]).reshape(-1, 1))
        self.assertEqual(probability, cdf[1] - cdf[0])
        self.assertEqual(conditional.lower, 1)
        self.assertEqual(conditional.upper, 2)

    def test_conditional_on_mode(self):
        mode, _ = self.distribution.mode()
        conditional, probability = self.distribution.truncated(mode)
        self.assertIsNone(conditional)

        point_value = (
            mode.simple_sets[0][self.distribution.variable].simple_sets[0].lower
        )
        conditional, probability = self.distribution.log_conditional(
            {self.distribution.variable: point_value}
        )
        self.assertIsInstance(conditional, DiracDeltaDistribution)
        self.assertTrue(probability > -np.inf)

    def test_copy(self):
        copy = self.distribution.__copy__()
        self.assertEqual(self.distribution, copy)
        copy.interval = copy.interval.intersection_with(
            SimpleInterval(-1, 1, Bound.CLOSED, Bound.CLOSED)
        )
        self.assertNotEqual(self.distribution, copy)

    def test_sample(self):
        samples = self.distribution.rejection_sample(100)
        self.assertEqual(samples.shape, (100, 1))
        likelihoods = self.distribution.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))

    def test_plot(self):
        fig = go.Figure(data=self.distribution.plot())
        self.assertIsNotNone(fig)
        # fig.show()

    def test_serialization(self):
        serialized = self.distribution.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertEqual(self.distribution, deserialized)
        self.assertIsInstance(deserialized, GaussianDistribution)


class TruncatedGaussianDistributionJapaneseManTestCase(unittest.TestCase):
    distribution = GaussianDistribution(Continuous("x"), location=0, scale=1)
    example_2: TruncatedGaussianDistribution
    example_3: TruncatedGaussianDistribution

    @classmethod
    def setUpClass(cls):
        cls.example_2, _ = cls.distribution.truncated(
            SimpleEvent(
                {cls.distribution.variable: closed(0.5, np.inf)}
            ).as_composite_set()
        )

        cls.example_3, _ = cls.distribution.truncated(
            SimpleEvent({cls.distribution.variable: closed(-1, 1)}).as_composite_set()
        )

    def test_raw_expectation_example_2(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 1})
        result = self.example_2.moment(order, center)
        expectation = result[self.distribution.variable]
        self.assertAlmostEqual(expectation, 1.14, delta=0.01)

    def test_raw_expectation_example_3(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 1})
        result = self.example_3.moment(order, center)
        expectation = result[self.distribution.variable]
        self.assertAlmostEqual(expectation, 0, delta=0.01)

    def test_raw_second_moment_example_2(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 2})
        result = self.example_2.moment(order, center)
        raw_moment = result[self.distribution.variable]
        self.assertAlmostEqual(raw_moment, 1.57, delta=0.01)

    def test_raw_second_moment_example_3(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 2})
        result = self.example_3.moment(order, center)
        raw_moment = result[self.distribution.variable]
        self.assertAlmostEqual(raw_moment, 0.291, delta=0.01)

    def test_raw_third_moment_example_2(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 3})
        result = self.example_2.moment(order, center)
        raw_moment = result[self.distribution.variable]
        self.assertAlmostEqual(raw_moment, 2.57, delta=0.01)

    def test_raw_third_moment_example_3(self):
        center = VariableMap({self.distribution.variable: 0})
        order = VariableMap({self.distribution.variable: 3})
        result = self.example_3.moment(order, center)
        raw_moment = result[self.distribution.variable]
        self.assertAlmostEqual(raw_moment, 0, delta=0.01)


class TruncatedGaussianSamplingTestCase(unittest.TestCase):
    x = Continuous("x")

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)

    def test_with_center_in_truncation(self):
        model = TruncatedGaussianDistribution(self.x, SimpleInterval(-3, 5), 1, 2)
        samples = model.robert_rejection_sample(1000).reshape(-1, 1)
        self.assertEqual(len(samples), 1000)
        likelihoods = model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))
        self.assertAlmostEqual(
            model.expectation(model.variables)[self.x], samples.mean(), delta=0.1
        )

    def test_with_center_higher_truncation(self):
        model = TruncatedGaussianDistribution(self.x, SimpleInterval(3, 10), 1, 2)
        samples = model.robert_rejection_sample(1000).reshape(-1, 1)
        self.assertEqual(len(samples), 1000)
        likelihoods = model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))
        self.assertAlmostEqual(
            model.expectation(model.variables)[model.variable],
            samples.mean(),
            delta=0.1,
        )

    def test_with_center_lower_truncation(self):
        model = TruncatedGaussianDistribution(self.x, SimpleInterval(-6, -2), 1, 2)
        samples = model.robert_rejection_sample(1000).reshape(-1, 1)
        self.assertEqual(len(samples), 1000)
        likelihoods = model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))
        self.assertAlmostEqual(
            model.expectation(model.variables)[model.variable],
            samples.mean(),
            delta=0.1,
        )

    def test_compare_rejection_sampling_with_robert_sampling(self):
        model = TruncatedGaussianDistribution(self.x, SimpleInterval(9, 11), 0, 1)
        with self.assertRaises(RecursionError):
            model.rejection_sample(50)

    def test_sampling_with_infinite_bounds_smaller_0(self):
        model = TruncatedGaussianDistribution(
            self.x, SimpleInterval(-1, float("inf")), 0, 1
        )
        samples = model.sample(1000)

        self.assertEqual(len(samples), 1000)
        likelihoods = model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))
        self.assertAlmostEqual(
            model.expectation(model.variables)[model.variable],
            np.array(samples).mean(),
            delta=0.1,
        )

    def test_sampling_with_infinite_bounds_greater_0(self):
        model = TruncatedGaussianDistribution(
            self.x, SimpleInterval(1, float("inf")), 0, 1
        )
        samples = model.sample(1000)
        likelihoods = model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))
        self.assertAlmostEqual(
            model.expectation(model.variables)[model.variable],
            np.array(samples).mean(),
            delta=0.1,
        )

    def test_sampling_with_infinite_bounds_and_flipping(self):
        model = TruncatedGaussianDistribution(
            self.x, SimpleInterval(-float("inf"), -1), 0, 1
        )
        samples = model.sample(1000)
        likelihoods = model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))
        self.assertAlmostEqual(
            model.expectation(model.variables)[model.variable],
            np.array(samples).mean(),
            delta=0.1,
        )

    def test_non_standard_sampling(self):
        model = TruncatedGaussianDistribution(
            self.x, SimpleInterval(-np.inf, -0.1), 0.5, 2
        )
        samples = model.robert_rejection_sample(1000).reshape(-1, 1)
        self.assertAlmostEqual(max(samples), -0.1, delta=0.1)
        likelihoods = model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))
        self.assertAlmostEqual(
            model.expectation(model.variables)[model.variable],
            samples.mean(),
            delta=0.1,
        )

    def test_with_zero_in_bound(self):
        model = TruncatedGaussianDistribution(
            self.x, SimpleInterval(-0.62, 0.0), 0.0, 0.5
        )
        samples = model.sample(1000)
        self.assertEqual(len(samples), 1000)
        mean = samples.mean()
        expectation = model.expectation(model.variables)[model.variable]
        self.assertAlmostEqual(mean, expectation, delta=0.1)

    def test_with_far_right_interval(self):
        model = TruncatedGaussianDistribution(self.x, SimpleInterval(11, np.inf), 0, 1)
        samples = model.sample(1000)
        self.assertEqual(len(samples), 1000)


if __name__ == "__main__":
    unittest.main()
