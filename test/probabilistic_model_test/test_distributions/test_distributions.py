import unittest
from enum import IntEnum

from krrood.adapters.json_serializer import to_json, from_json

from probabilistic_model.distributions.distributions import *
from probabilistic_model.utils import MissingDict


class TestEnum(IntEnum):
    A = 0
    B = 1
    C = 2


class IntegerDistributionTestCase(unittest.TestCase):
    x = Integer("x")
    model: IntegerDistribution

    def setUp(self):
        probabilities = MissingDict(float)
        probabilities[1] = 4 / 20
        probabilities[2] = 5 / 20
        probabilities[4] = 11 / 20
        self.model = IntegerDistribution(variable=self.x, probabilities=probabilities)

    def test_likelihood(self):
        pdf = self.model.likelihood(np.array([1, 2, 3, 4]).reshape(-1, 1))
        self.assertAlmostEqual(pdf[0], 4 / 20)
        self.assertAlmostEqual(pdf[1], 5 / 20)
        self.assertAlmostEqual(pdf[2], 0)
        self.assertAlmostEqual(pdf[3], 11 / 20)

    def test_probability(self):
        event = SimpleEvent.from_data({self.x: closed(1, 3)}).as_composite_set()
        self.assertEqual(self.model.probability(event), 9 / 20)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertAlmostEqual(likelihood, 11 / 20)
        self.assertEqual(mode, SimpleEvent.from_data({self.x: singleton(4)}).as_composite_set())

    def test_conditional(self):
        event = SimpleEvent.from_data({self.x: closed(0, 1) | closed(3, 4)}).as_composite_set()
        conditional, probability = self.model.truncated(event)
        self.assertEqual(probability, 15 / 20)
        self.assertAlmostEqual(conditional.probabilities[1], 4 / 15)
        self.assertAlmostEqual(conditional.probabilities[4], 11 / 15)

    def test_conditional_impossible(self):
        event = SimpleEvent.from_data({self.x: open(0, 1)}).as_composite_set()

        conditional, probability = self.model.truncated(event)
        self.assertIsNone(conditional)
        self.assertEqual(probability, 0)

    def test_sample(self):
        samples = self.model.sample(100)
        likelihoods = self.model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))

    def test_copy(self):
        copied = self.model.__copy__()
        self.assertEqual(copied, self.model)
        copied.probabilities = MissingDict(float)
        self.assertNotEqual(copied, self.model)

    def test_fit(self):
        data = [1, 2, 2, 2]
        self.model.fit(data)
        self.assertEqual(self.model.probabilities[1], [1 / 4])
        self.assertEqual(self.model.probabilities[2], [3 / 4])

    def test_domain(self):
        support = self.model.univariate_support
        self.assertEqual(support, singleton(1) | singleton(2) | singleton(4))

    def test_domain_if_weights_are_zero(self):
        distribution = IntegerDistribution(variable=self.x, probabilities=MissingDict(float))
        self.assertTrue(distribution.univariate_support.is_empty())

    def test_plot(self):
        fig = go.Figure(self.model.plot(), self.model.plotly_layout())
        # fig.show()

    def test_serialization(self):
        serialized = to_json(self.model)
        deserialized = from_json(serialized)
        self.assertIsInstance(deserialized, DiscreteDistribution)
        self.assertEqual(deserialized, self.model)

    def test_cdf(self):
        cdf = self.model.cumulative_distribution_function(np.array([0, 1, 2, 3, 4]).reshape(-1, 1))
        self.assertAlmostEqual(cdf[0], 0)
        self.assertAlmostEqual(cdf[1], 4 / 20)
        self.assertAlmostEqual(cdf[2], 9 / 20)
        self.assertAlmostEqual(cdf[3], 9 / 20)
        self.assertAlmostEqual(cdf[4], 1)

    def test_bernoulli(self):
        y = Integer("y")

        bern = BernoulliDistribution(y, p=0.7)
        self.assertAlmostEqual(bern.p, 0.7)

        like = bern.likelihood(np.array([0, 1]).reshape(-1, 1))
        self.assertAlmostEqual(like[0], 0.3)
        self.assertAlmostEqual(like[1], 0.7)

        cdf = bern.cdf(np.array([0, 1, 2]).reshape(-1, 1))
        self.assertAlmostEqual(cdf[0], 1 - 0.7)
        self.assertAlmostEqual(cdf[1], 1)
        self.assertAlmostEqual(cdf[2], 1)

        mode_event, mode_like = bern.mode()
        self.assertAlmostEqual(mode_like, 0.7)
        self.assertEqual(mode_event, SimpleEvent({y: singleton(1)}).as_composite_set())

        fair = BernoulliDistribution(y, p=0.5)
        fair_mode_event, fair_mode_like = fair.mode()
        self.assertAlmostEqual(fair_mode_like, 0.5)
        self.assertEqual(
            fair_mode_event,
            SimpleEvent({y: singleton(0) | singleton(1)}).as_composite_set(),
        )

        samples = bern.sample(2000)
        self.assertTrue(set(int(v[0]) for v in samples).issubset({0, 1}))
        empirical_mean = float(np.mean(samples))
        self.assertTrue(abs(empirical_mean - 0.7) < 0.1)

        serialized = bern.to_json()
        restored = SubclassJSONSerializer.from_json(serialized)
        self.assertIsInstance(restored, DiscreteDistribution)
        self.assertEqual(restored, bern)


class SymbolicDistributionTestCase(unittest.TestCase):
    x = Symbolic(name="x", domain=Set.from_iterable(TestEnum))
    model: SymbolicDistribution

    def setUp(self):
        probabilities = MissingDict(float)
        probabilities[hash(TestEnum.A)] = 7 / 20
        probabilities[hash(TestEnum.B)] = 13 / 20
        self.model = SymbolicDistribution(variable=self.x, probabilities=probabilities)

    def test_sample(self):
        samples = self.model.sample(100)
        likelihoods = self.model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 13 / 20)
        self.assertEqual(mode, SimpleEvent.from_data({self.x: TestEnum.B}).as_composite_set())

    def test_plot(self):
        fig = go.Figure(self.model.plot(), self.model.plotly_layout())
        # fig.show()

    def test_probability(self):
        event = SimpleEvent.from_data({self.x: (TestEnum.A, TestEnum.C)}).as_composite_set()
        self.assertEqual(self.model.probability(event), 7 / 20)

    def test_support(self):
        support = self.model.univariate_support
        self.assertEqual(
            support,
            Set.from_simple_sets(
                *[SetElement.from_data(TestEnum.A, self.x.domain.simple_sets[0].all_elements),
                SetElement.from_data(TestEnum.B, self.x.domain.simple_sets[0].all_elements)]
            ),
        )

    def test_fit(self):
        data = [TestEnum.A, TestEnum.B, TestEnum.B, TestEnum.B]
        self.model.fit_from_indices(data)

        e_1 = SimpleEvent.from_data({self.x: TestEnum.A}).as_composite_set()
        self.assertEqual(self.model.probability(e_1), 1 / 4)

        e_2 = SimpleEvent.from_data({self.x: TestEnum.B}).as_composite_set()
        self.assertEqual(self.model.probability(e_2), 3 / 4)

        event = e_1 | e_2

        prob = self.model.probability(event)
        self.assertEqual(prob, 1)


class DiracDeltaDistributionTestCase(unittest.TestCase):
    x = Continuous("x")
    model: DiracDeltaDistribution

    def setUp(self):
        self.model = DiracDeltaDistribution(variable=self.x, location=0, density_cap=2)

    def test_likelihood(self):
        pdf = self.model.likelihood(np.array([0, 1]).reshape(-1, 1))
        self.assertEqual(pdf[0], 2)
        self.assertEqual(pdf[1], 0)

    def test_cdf(self):
        cdf = self.model.cumulative_distribution_function(np.array([-1, 0, 1]).reshape(-1, 1))
        self.assertEqual(cdf[0], 0)
        self.assertEqual(cdf[1], 1)
        self.assertEqual(cdf[2], 1)

    def test_probability(self):
        event = SimpleEvent.from_data({self.x: closed(0, 1) | closed(1.5, 2)}).as_composite_set()
        self.assertEqual(self.model.probability(event), 1)

    def test_probability_0(self):
        event = SimpleEvent.from_data(
            {self.x: open_closed(0 + self.model.tolerance, 1)}
        ).as_composite_set()
        self.assertEqual(self.model.probability(event), 0.0)

    def test_conditional(self):
        event = SimpleEvent.from_data({self.model.variable: closed(-1, 2)}).as_composite_set()
        conditional, probability = self.model.truncated(event)
        self.assertEqual(conditional, self.model)
        self.assertEqual(probability, 1)

    def test_conditional_impossible(self):
        event = SimpleEvent.from_data({self.model.variable: closed(1, 2)}).as_composite_set()
        conditional, probability = self.model.truncated(event)
        self.assertIsNone(conditional)
        self.assertEqual(0, probability)

    def test_mode(self):
        mode, log_likelihood = self.model.univariate_log_mode()
        self.assertEqual(mode, singleton(0))
        self.assertEqual(log_likelihood, np.log(2))

    def test_sample(self):
        samples = self.model.sample(100)
        likelihoods = self.model.likelihood(samples)
        self.assertTrue(all(likelihoods == 2))

    def test_expectation(self):
        self.assertEqual(self.model.expectation([self.x])[self.x], 0)

    def test_variance(self):
        self.assertEqual(self.model.variance([self.x])[self.x], 0)

    def test_higher_order_moment(self):
        center = self.model.expectation([self.x])
        order = VariableMap({self.x: 3})
        self.assertEqual(self.model.moment(order, center)[self.x], 0)

    def test_serialization(self):
        serialized = to_json(self.model)
        deserialized = from_json(serialized)
        self.assertIsInstance(deserialized, DiracDeltaDistribution)
        self.assertEqual(deserialized, self.model)

    def test_plot(self):
        fig = go.Figure(self.model.plot(), self.model.plotly_layout())  # fig.show()


if __name__ == "__main__":
    unittest.main()
