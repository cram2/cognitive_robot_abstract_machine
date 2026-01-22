import unittest
from enum import IntEnum

from probabilistic_model.distributions.distributions import *
from probabilistic_model.utils import MissingDict
from random_events.utils import SubclassJSONSerializer


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
        self.model = IntegerDistribution(self.x, probabilities)

    def test_likelihood(self):
        pdf = self.model.likelihood(np.array([1, 2, 3, 4]).reshape(-1, 1))
        self.assertAlmostEqual(pdf[0], 4 / 20)
        self.assertAlmostEqual(pdf[1], 5 / 20)
        self.assertAlmostEqual(pdf[2], 0)
        self.assertAlmostEqual(pdf[3], 11 / 20)

    def test_probability(self):
        event = SimpleEvent({self.x: closed(1, 3)}).as_composite_set()
        self.assertEqual(self.model.probability(event), 9 / 20)

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertAlmostEqual(likelihood, 11 / 20)
        self.assertEqual(mode, SimpleEvent({self.x: singleton(4)}).as_composite_set())

    def test_conditional(self):
        event = SimpleEvent({self.x: closed(0, 1) | closed(3, 4)}).as_composite_set()
        conditional, probability = self.model.truncated(event)
        self.assertEqual(probability, 15 / 20)
        self.assertAlmostEqual(conditional.probabilities[1], 4 / 15)
        self.assertAlmostEqual(conditional.probabilities[4], 11 / 15)

    def test_conditional_impossible(self):
        event = SimpleEvent({self.x: open(0, 1)}).as_composite_set()

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
        distribution = IntegerDistribution(self.x, MissingDict(float))
        self.assertTrue(distribution.univariate_support.is_empty())

    def test_plot(self):
        fig = go.Figure(self.model.plot(), self.model.plotly_layout())
        # fig.show()

    def test_serialization(self):
        serialized = self.model.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertIsInstance(deserialized, DiscreteDistribution)
        self.assertEqual(deserialized, self.model)

    def test_cdf(self):
        cdf = self.model.cdf(np.array([0, 1, 2, 3, 4]).reshape(-1, 1))
        self.assertAlmostEqual(cdf[0], 0)
        self.assertAlmostEqual(cdf[1], 4 / 20)
        self.assertAlmostEqual(cdf[2], 9 / 20)
        self.assertAlmostEqual(cdf[3], 9 / 20)
        self.assertAlmostEqual(cdf[4], 1)


class SymbolicDistributionTestCase(unittest.TestCase):
    x = Symbolic("x", Set.from_iterable(TestEnum))
    model: SymbolicDistribution

    def setUp(self):
        probabilities = MissingDict(float)
        probabilities[hash(TestEnum.A)] = 7 / 20
        probabilities[hash(TestEnum.B)] = 13 / 20
        self.model = SymbolicDistribution(self.x, probabilities)

    def test_sample(self):
        samples = self.model.sample(100)
        likelihoods = self.model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 13 / 20)
        self.assertEqual(mode, SimpleEvent({self.x: TestEnum.B}).as_composite_set())

    def test_plot(self):
        fig = go.Figure(self.model.plot(), self.model.plotly_layout())
        # fig.show()

    def test_probability(self):
        event = SimpleEvent({self.x: (TestEnum.A, TestEnum.C)}).as_composite_set()
        self.assertEqual(self.model.probability(event), 7 / 20)

    def test_support(self):
        support = self.model.univariate_support
        self.assertEqual(
            support,
            Set(
                SetElement(TestEnum.A, self.x.domain.simple_sets[0].all_elements),
                SetElement(TestEnum.B, self.x.domain.simple_sets[0].all_elements),
            ),
        )

    def test_fit(self):
        data = [TestEnum.A, TestEnum.B, TestEnum.B, TestEnum.B]
        self.model.fit_from_indices(data)

        e_1 = SimpleEvent({self.x: TestEnum.A}).as_composite_set()
        self.assertEqual(self.model.probability(e_1), 1 / 4)

        e_2 = SimpleEvent({self.x: TestEnum.B}).as_composite_set()
        self.assertEqual(self.model.probability(e_2), 3 / 4)

        event = e_1 | e_2

        prob = self.model.probability(event)
        self.assertEqual(prob, 1)


class DiracDeltaDistributionTestCase(unittest.TestCase):
    x = Continuous("x")
    model: DiracDeltaDistribution

    def setUp(self):
        self.model = DiracDeltaDistribution(self.x, 0, 2)

    def test_likelihood(self):
        pdf = self.model.likelihood(np.array([0, 1]).reshape(-1, 1))
        self.assertEqual(pdf[0], 2)
        self.assertEqual(pdf[1], 0)

    def test_cdf(self):
        cdf = self.model.cdf(np.array([-1, 0, 1]).reshape(-1, 1))
        self.assertEqual(cdf[0], 0)
        self.assertEqual(cdf[1], 1)
        self.assertEqual(cdf[2], 1)

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0, 1) | closed(1.5, 2)}).as_composite_set()
        self.assertEqual(self.model.probability(event), 1)

    def test_probability_0(self):
        event = SimpleEvent(
            {self.x: open_closed(0 + self.model.tolerance, 1)}
        ).as_composite_set()
        self.assertEqual(self.model.probability(event), 0.0)

    def test_conditional(self):
        event = SimpleEvent({self.model.variable: closed(-1, 2)}).as_composite_set()
        conditional, probability = self.model.truncated(event)
        self.assertEqual(conditional, self.model)
        self.assertEqual(probability, 1)

    def test_conditional_impossible(self):
        event = SimpleEvent({self.model.variable: closed(1, 2)}).as_composite_set()
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
        serialized = self.model.to_json()
        deserialized = SubclassJSONSerializer.from_json(serialized)
        self.assertIsInstance(deserialized, DiracDeltaDistribution)
        self.assertEqual(deserialized, self.model)

    def test_plot(self):
        fig = go.Figure(self.model.plot(), self.model.plotly_layout())  # fig.show()


if __name__ == "__main__":
    unittest.main()
