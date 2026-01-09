import itertools
import unittest
from enum import IntEnum

import numpy as np
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic

from probabilistic_model.distributions.multinomial import MultinomialDistribution


class XEnum(IntEnum):
    A = 0
    B = 1


class YEnum(IntEnum):
    A = 0
    B = 1
    C = 2


class ZEnum(IntEnum):
    A = 0
    B = 1
    C = 2
    D = 3


class MultinomialConstructionTestCase(unittest.TestCase):
    x: Symbolic
    y: Symbolic
    z: Symbolic

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.x = Symbolic("X", Set.from_iterable(XEnum))
        cls.y = Symbolic("Y", Set.from_iterable(YEnum))
        cls.z = Symbolic("Z", Set.from_iterable(ZEnum))

    def test_creation_with_probabilities(self):
        distribution = MultinomialDistribution(
            [self.x, self.y, self.z],
            np.random.rand(
                len(self.x.domain.simple_sets),
                len(self.y.domain.simple_sets),
                len(self.z.domain.simple_sets),
            ),
        )
        self.assertTrue(distribution)

    def test_creation_without_probabilities(self):
        distribution = MultinomialDistribution([self.x])
        self.assertTrue(np.allclose(1.0 / 2.0, distribution.probabilities))

    def test_creation_with_invalid_probabilities_shape(self):
        probabilities = np.array([[0.1, 0.1], [0.2, 0.2]])
        with self.assertRaises(ValueError):
            distribution = MultinomialDistribution([self.x, self.y], probabilities)

    def test_copy(self):
        distribution_1 = MultinomialDistribution(
            [self.x, self.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]])
        )
        distribution_2 = distribution_1.__copy__()
        self.assertEqual(distribution_1, distribution_2)
        distribution_2.probabilities = np.zeros_like(distribution_2.probabilities)
        self.assertNotEqual(distribution_2, distribution_1)

    def test_to_tabulate(self):
        distribution = MultinomialDistribution(
            [self.x, self.y, self.z],
            np.random.rand(
                len(self.x.domain.simple_sets),
                len(self.y.domain.simple_sets),
                len(self.z.domain.simple_sets),
            ),
        )
        table = distribution.to_tabulate()
        self.assertTrue(table)  # print(tabulate.tabulate(table, headers="firstrow"))

    def test_to_str(self):
        distribution = MultinomialDistribution([self.x, self.y, self.z])
        self.assertTrue(str(distribution))


class MultinomialInferenceTestCase(unittest.TestCase):
    x = Symbolic("X", Set.from_iterable(XEnum))
    y = Symbolic("Y", Set.from_iterable(YEnum))
    z = Symbolic("Z", Set.from_iterable(ZEnum))
    random_distribution: MultinomialDistribution
    random_distribution_mass: float
    crafted_distribution: MultinomialDistribution
    crafted_distribution_mass: float

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        cls.random_distribution = MultinomialDistribution(
            [cls.x, cls.y, cls.z],
            np.random.rand(
                len(cls.x.domain.simple_sets),
                len(cls.y.domain.simple_sets),
                len(cls.z.domain.simple_sets),
            ),
        )
        cls.random_distribution_mass = cls.random_distribution.probabilities.sum()

        cls.crafted_distribution = MultinomialDistribution(
            [cls.x, cls.y], np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]])
        )
        cls.crafted_distribution_mass = cls.crafted_distribution.probabilities.sum()

    def test_normalize_random(self):
        self.random_distribution.normalize()
        self.assertAlmostEqual(self.random_distribution.probabilities.sum(), 1.0)

    def test_random_marginal_with_normalize(self):
        marginal = self.random_distribution.marginal([self.x, self.y])
        self.assertAlmostEqual(marginal.probabilities.sum(), 1)
        self.assertEqual(marginal.variables, (self.x, self.y))

    def test_crafted_marginal_with_normalize(self):
        marginal = self.crafted_distribution.marginal([self.x])
        self.assertAlmostEqual(marginal.probabilities.sum(), 1)
        self.assertAlmostEqual(
            marginal.probabilities[0], 0.6 / self.crafted_distribution_mass
        )
        self.assertAlmostEqual(
            marginal.probabilities[1], 1.2 / self.crafted_distribution_mass
        )

    def test_random_mode(self):
        mode, probability = self.random_distribution.mode()
        mode = mode.simple_sets[0]
        self.assertAlmostEqual(
            probability, self.random_distribution.probabilities.max()
        )
        self.assertEqual(mode["X"], self.x.make_value(XEnum.A))
        self.assertEqual(mode["Y"], self.y.make_value(YEnum.A))

    def test_crafted_mode(self):
        mode, probability = self.crafted_distribution.mode()
        mode = mode.simple_sets[0]
        self.assertEqual(probability, self.crafted_distribution.probabilities.max())
        self.assertEqual(mode["X"], self.x.make_value(XEnum.B))
        self.assertEqual(mode["Y"], self.x.make_value(YEnum.A))

    def test_likelihood(self):
        data = np.array([[XEnum.A, YEnum.A], [XEnum.B, YEnum.B]])
        likelihood = self.crafted_distribution.likelihood(data)
        self.assertEqual(likelihood.shape, (2,))
        self.assertAlmostEqual(likelihood[0], 0.1 / self.crafted_distribution_mass)
        self.assertAlmostEqual(likelihood[1], 0.4 / self.crafted_distribution_mass)

    def test_multiple_modes(self):
        distribution = MultinomialDistribution(
            [self.x, self.y],
            np.array([[0.1, 0.7, 0.3], [0.7, 0.4, 0.1]]),
        )
        mode, likelihood = distribution.mode()

        mode_by_hand = (
            SimpleEvent({self.x: XEnum.A, self.y: YEnum.B}).as_composite_set()
            | SimpleEvent({self.x: XEnum.B, self.y: YEnum.A}).as_composite_set()
        )

        self.assertEqual(likelihood, 0.7)
        self.assertEqual(len(mode.simple_sets), 2)
        self.assertEqual(mode, mode_by_hand)

    def test_crafted_probability(self):
        distribution = self.crafted_distribution
        distribution.normalize()
        event = SimpleEvent()
        self.assertAlmostEqual(distribution.probability(event.as_composite_set()), 1)

        event[self.x] = XEnum.A
        event._update_cpp_object()
        self.assertAlmostEqual(
            distribution.probability(event.as_composite_set()), 1 / 3
        )

        event[self.y] = (YEnum.A, YEnum.B)
        self.assertAlmostEqual(
            distribution.probability(event.as_composite_set()),
            0.3 / self.crafted_distribution_mass,
        )

    def test_random_probability(self):
        self.random_distribution.normalize()
        event = SimpleEvent()
        event.fill_missing_variables(self.random_distribution.variables)
        self.assertAlmostEqual(
            self.random_distribution.probability(event.as_composite_set()), 1
        )

        event[self.x] = XEnum.A
        self.assertLessEqual(
            self.random_distribution.probability(event.as_composite_set()), 1.0
        )

        event[self.y] = (YEnum.A, YEnum.B)
        self.assertLessEqual(
            self.random_distribution.probability(event.as_composite_set()), 1.0
        )

    def test_crafted_conditional(self):
        event = SimpleEvent({self.y: (YEnum.A, YEnum.B)})
        conditional, probability = self.crafted_distribution.truncated(
            event.as_composite_set()
        )
        self.assertAlmostEqual(conditional.probability(event.as_composite_set()), 1)

        impossible_event = SimpleEvent({self.y: YEnum.C})
        impossible_event.fill_missing_variables(self.crafted_distribution.variables)

        self.assertEqual(
            conditional.probability(impossible_event.as_composite_set()), 0.0
        )

    def test_as_probabilistic_circuit(self):
        self.random_distribution.normalize()
        circuit = self.random_distribution.as_probabilistic_circuit()

        for event in itertools.product(
            self.x.domain.simple_sets,
            self.y.domain.simple_sets,
            self.z.domain.simple_sets,
        ):
            event = SimpleEvent(zip([self.x, self.y, self.z], event)).as_composite_set()
            self.assertAlmostEqual(
                self.random_distribution.probability(event),
                circuit.probabilistic_circuit.probability(event),
            )

    def test_serialization(self):
        distribution = self.random_distribution
        serialized = distribution.to_json()
        deserialized = MultinomialDistribution.from_json(serialized)
        self.assertEqual(distribution, deserialized)


if __name__ == "__main__":
    unittest.main()
