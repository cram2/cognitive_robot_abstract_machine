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
    DiracDeltaDistribution,
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
            UniformDistribution(
                variable=self.variable, interval=closed(0, 1).simple_sets[0]
            ),
            probabilistic_circuit=ProbabilisticCircuit(),
        )

    def test_conditional_from_simple_event(self):
        event = SimpleEvent.from_data(
            {self.variable: closed(0.5, 2)}
        ).as_composite_set()
        conditional, probability = self.leaf.probabilistic_circuit.truncated(event)
        self.assertEqual(len(list(conditional.nodes())), 1)
        self.assertEqual(probability, 0.5)
        self.assertEqual(
            conditional.root.distribution.univariate_support, closed(0.5, 1)
        )

    def test_conditional_from_singleton_event(self):
        event = SimpleEvent.from_data(
            {self.variable: singleton(0.3)}
        ).as_composite_set()
        conditional, probability = self.leaf.probabilistic_circuit.truncated(event)
        self.assertIsNone(conditional)

        conditional, probability = self.leaf.probabilistic_circuit.conditional(
            {self.variable: 0.3}
        )

        self.assertEqual(len(list(conditional.nodes())), 1)
        self.assertEqual(probability, 1.0)
        self.assertAlmostEqual(conditional.root.distribution.location, 0.3)

    def test_probabilistic_circuit_singleton(self):
        x = Continuous("x")
        pc = leaf(
            UniformDistribution(variable=x, interval=SimpleInterval.from_data(0, 1)),
            ProbabilisticCircuit(),
        )

        event = SimpleEvent.from_data({x: singleton(0.5)}).as_composite_set()

        # singleton_allowed=False
        conditional, probability = pc.probabilistic_circuit.truncated(event)
        self.assertIsNone(conditional)

        # singleton_allowed=True
        conditional, probability = pc.probabilistic_circuit.truncated(
            event, singleton_allowed=True
        )
        self.assertIsNotNone(conditional)
        self.assertIsInstance(conditional.root.distribution, DiracDeltaDistribution)
        self.assertEqual(conditional.root.distribution.location, 0.5)
        self.assertAlmostEqual(probability, 1.0)  # f(0.5) = 1.0

    def test_probabilistic_circuit_complex_event_with_singleton(self):
        x = Continuous("x")
        # Sum of two uniforms: 0.5 * U(0,1) + 0.5 * U(2,3)
        pc = ProbabilisticCircuit()
        u1 = leaf(
            UniformDistribution(variable=x, interval=SimpleInterval.from_data(0, 1)), pc
        )
        u2 = leaf(
            UniformDistribution(variable=x, interval=SimpleInterval.from_data(2, 3)), pc
        )
        root = SumUnit(probabilistic_circuit=pc)
        root.add_subcircuit(u1, np.log(0.5))
        root.add_subcircuit(u2, np.log(0.5))

        # Event: [0.5, 0.6] OR {2.5}
        event = SimpleEvent.from_data(
            {x: closed(0.5, 0.6) | singleton(2.5)}
        ).as_composite_set()

        # singleton_allowed=True
        conditional, probability = pc.truncated(event, singleton_allowed=True)

        # Probabilities:
        # P([0.5, 0.6]) = 0.5 * 0.1 = 0.05
        # f(2.5) = 0.5 * 1.0 = 0.5
        # Total "probability" in truncated sense: 0.05 + 0.5 = 0.55
        self.assertAlmostEqual(probability, 0.55)

        # The result should be a SumUnit with two children:
        # 1. Truncated U(0,1) on [0.5, 0.6] -> U(0.5, 0.6) with weight 0.05/0.55
        # 2. DiracDelta(2.5) with weight 0.5/0.55

        self.assertIsInstance(conditional.root, SumUnit)
        self.assertEqual(len(conditional.root.subcircuits), 2)

        weights = np.exp(conditional.root.log_weights)
        self.assertTrue(any(np.isclose(weights, 0.05 / 0.55)))
        self.assertTrue(any(np.isclose(weights, 0.5 / 0.55)))

    def test_conditional_from_complex_event(self):
        interval = closed(0.0, 0.2) | closed(0.5, 1.0) | singleton(0.3)
        event = SimpleEvent.from_data({self.variable: interval})
        conditional, probability = self.leaf.probabilistic_circuit.truncated(
            event.as_composite_set()
        )

        self.assertEqual(len(list(conditional.nodes())), 3)
        self.assertEqual(len(list(conditional.edges())), 2)
        self.assertIsInstance(conditional.root, SumUnit)

    def test_conditional_with_none(self):
        event = SimpleEvent.from_data({self.variable: singleton(2)}).as_composite_set()
        conditional, probability = self.leaf.probabilistic_circuit.truncated(event)
        self.assertEqual(conditional, None)


class DiscreteDistributionTestCase(unittest.TestCase):
    symbol = Symbolic(name="animal", domain=Set.from_iterable(Animal))
    integer = Integer("x")

    symbolic_distribution: ProbabilisticCircuit
    integer_distribution: ProbabilisticCircuit

    def setUp(self):
        symbolic_probabilities = MissingDict(
            float,
            {hash(Animal.CAT): 0.1, hash(Animal.DOG): 0.2, hash(Animal.FISH): 0.7},
        )
        self.symbolic_distribution = leaf(
            SymbolicDistribution(
                variable=self.symbol, probabilities=symbolic_probabilities
            ),
            ProbabilisticCircuit(),
        ).probabilistic_circuit
        integer_probabilities = MissingDict(float, {0: 0.1, 1: 0.2, 2: 0.7})
        self.integer_distribution = leaf(
            IntegerDistribution(
                variable=self.integer, probabilities=integer_probabilities
            ),
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
