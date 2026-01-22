import unittest
from enum import IntEnum

from matplotlib import pyplot as plt
from random_events.interval import *
from random_events.variable import Integer, Continuous

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import leaf
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import LeafUnit
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.distributions.distributions import (
    SymbolicDistribution,
    IntegerDistribution,
    DiscreteDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
from probabilistic_model.utils import MissingDict


class Animal(IntEnum):
    CAT = 0
    DOG = 1
    FISH = 2


class ContinuousDistributionTestCase(unittest.TestCase):
    variable = Continuous("x")
    leaf: LeafUnit

    def setUp(self):
        self.leaf = leaf(
            UniformDistribution(self.variable, closed(0, 1).simple_sets[0]),
            probabilistic_circuit=ProbabilisticCircuit(),
        )

    def test_conditional_from_simple_event(self):
        event = SimpleEvent({self.variable: closed(0.5, 2)}).as_composite_set()
        conditional, probability = self.leaf.probabilistic_circuit.truncated(event)
        self.assertEqual(len(list(conditional.nodes())), 1)
        self.assertEqual(probability, 0.5)
        self.assertEqual(
            conditional.root.distribution.univariate_support, closed(0.5, 1)
        )

    def test_conditional_from_singleton_event(self):
        event = SimpleEvent({self.variable: singleton(0.3)}).as_composite_set()
        conditional, probability = self.leaf.probabilistic_circuit.truncated(event)
        self.assertIsNone(conditional)

        conditional, probability = self.leaf.probabilistic_circuit.conditional(
            {self.variable: 0.3}
        )

        self.assertEqual(len(list(conditional.nodes())), 1)
        self.assertEqual(probability, 1.0)
        self.assertAlmostEqual(conditional.root.distribution.location, 0.3)

    def test_conditional_from_complex_event(self):
        interval = closed(0.0, 0.2) | closed(0.5, 1.0) | singleton(0.3)
        event = SimpleEvent({self.variable: interval})
        conditional, probability = self.leaf.probabilistic_circuit.truncated(
            event.as_composite_set()
        )

        self.assertEqual(len(list(conditional.nodes())), 3)
        self.assertEqual(len(list(conditional.edges())), 2)
        self.assertIsInstance(conditional.root, SumUnit)

    def test_conditional_with_none(self):
        event = SimpleEvent({self.variable: singleton(2)}).as_composite_set()
        conditional, probability = self.leaf.probabilistic_circuit.truncated(event)
        self.assertEqual(conditional, None)


class DiscreteDistributionTestCase(unittest.TestCase):
    symbol = Symbolic("animal", Set.from_iterable(Animal))
    integer = Integer("x")

    symbolic_distribution: ProbabilisticCircuit
    integer_distribution: ProbabilisticCircuit

    def setUp(self):
        symbolic_probabilities = MissingDict(
            float,
            {hash(Animal.CAT): 0.1, hash(Animal.DOG): 0.2, hash(Animal.FISH): 0.7},
        )
        self.symbolic_distribution = leaf(
            SymbolicDistribution(self.symbol, symbolic_probabilities),
            ProbabilisticCircuit(),
        ).probabilistic_circuit
        integer_probabilities = MissingDict(float, {0: 0.1, 1: 0.2, 2: 0.7})
        self.integer_distribution = leaf(
            IntegerDistribution(self.integer, integer_probabilities),
            ProbabilisticCircuit(),
        ).probabilistic_circuit

    def test_as_deterministic_sum(self):
        old_probs = self.symbolic_distribution.root.distribution.probabilities.values()
        new_root = self.symbolic_distribution.root.as_deterministic_sum()
        self.assertIsInstance(new_root, SumUnit)
        self.assertEqual(new_root, self.symbolic_distribution.root)
        self.assertEqual(len(new_root.subcircuits), 3)
        self.assertTrue(
            np.allclose(new_root.log_weights[::-1], np.log(np.array(list(old_probs))))
        )

    def test_from_deterministic_sum(self):
        self.integer_distribution.root.as_deterministic_sum()
        result = UnivariateDiscreteLeaf.from_mixture(self.integer_distribution)
        self.assertIsInstance(result, UnivariateDiscreteLeaf)
        self.assertIsInstance(result.distribution, IntegerDistribution)
        self.assertTrue(
            np.allclose(
                np.array(list(result.distribution.probabilities.values())),
                np.array([0.1, 0.2, 0.7]),
            )
        )
