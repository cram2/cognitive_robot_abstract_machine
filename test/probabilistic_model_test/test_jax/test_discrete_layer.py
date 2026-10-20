import unittest
from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from enum import IntEnum

import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import numpy as np
from random_events.set import Set
from random_events.variable import Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import (
    DifferentiableDiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableSparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as NXProbabilisticCircuit,
    SumUnit,
    UnivariateDiscreteLeaf,
    leaf,
)
from probabilistic_model.utils import MissingDict


class Animal(IntEnum):
    CAT = 0
    DOG = 1
    FISH = 2


class DiscreteLayerTestCase(unittest.TestCase):
    model: DifferentiableDiscreteLayer
    x = Symbolic(name="x", domain=Set.from_iterable(Animal))

    @classmethod
    def setUpClass(cls):
        cls.model = DifferentiableDiscreteLayer(
            0, jnp.log(jnp.array([[0, 1, 2], [3, 4, 0]]))
        )
        cls.model.validate()

    def test_normalization(self):
        result = self.model.log_normalization_constant
        correct = jnp.log(jnp.array([3.0, 7.0]))
        self.assertTrue(jnp.allclose(result, correct, atol=1e-3))

    def test_log_likelihood(self):
        x = jnp.array([0.0])
        result = self.model.log_likelihood_of_nodes_single(x)
        correct = jnp.log(jnp.array([0.0, 3 / 7]))
        self.assertTrue(jnp.allclose(result, correct, atol=1e-3))

        x2 = jnp.array([0.0, 1.0, 2]).reshape(-1, 1)
        result = self.model.log_likelihood_of_nodes(x2)
        self.assertEqual(result.shape, (3, 2))
        correct = jnp.log(
            jnp.array([[0.0, 3.0 / 7.0], [1.0 / 3.0, 4.0 / 7.0], [2.0 / 3.0, 0.0]])
        )
        self.assertTrue(jnp.allclose(result, correct, atol=1e-3))

    def test_from_nx(self):

        nx_pc = NXProbabilisticCircuit()

        p1 = MissingDict(
            float, {hash(Animal.CAT): 0.0, hash(Animal.DOG): 1, hash(Animal.FISH): 2}
        )
        d1 = leaf(SymbolicDistribution(variable=self.x, probabilities=p1), nx_pc)

        p2 = MissingDict(
            float, {hash(Animal.CAT): 3, hash(Animal.DOG): 4, hash(Animal.FISH): 0}
        )
        d2 = leaf(SymbolicDistribution(variable=self.x, probabilities=p2), nx_pc)
        s = SumUnit(probabilistic_circuit=nx_pc)
        s.add_subcircuit(d2, np.log(0.5))
        s.add_subcircuit(d1, np.log(0.5))

        nx_pc = s.probabilistic_circuit

        jax_pc = CircuitRepresentations().convert(nx_pc, DifferentiableLayeredCircuit)
        discrete_layer = jax_pc.root.child_layers[0]

        self.assertIsInstance(discrete_layer, DifferentiableDiscreteLayer)
        self.assertEqual(discrete_layer.variable, 0)
        self.assertEqual(discrete_layer.log_probabilities.shape, (2, 3))

        self.assertTrue(
            jnp.allclose(discrete_layer.log_probabilities, self.model.log_probabilities)
        )

    def test_to_rx(self):
        mixture = DifferentiableSparseSumLayer(
            [self.model],
            [BCOO((jnp.zeros(2), jnp.array([[0, 0], [0, 1]])), shape=(1, 2))],
        )
        rx_circuit = CircuitRepresentations().convert(
            DifferentiableLayeredCircuit(SortedSet([self.x]), mixture),
            NXProbabilisticCircuit,
        )
        leaves = [
            node
            for node in rx_circuit.nodes()
            if isinstance(node, UnivariateDiscreteLeaf)
        ]
        self.assertEqual(len(leaves), self.model.number_of_nodes)
        for node in leaves:
            self.assertEqual(node.variable, self.x)
            distribution: SymbolicDistribution = node.distribution
            self.assertAlmostEqual(
                sum(distribution.probabilities.values()), 1.0, places=5
            )


if __name__ == "__main__":
    unittest.main()
