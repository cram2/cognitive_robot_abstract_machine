import random
import unittest
from enum import IntEnum

import optax
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from scipy.special import logsumexp

from probabilistic_model.learning.region_graph.region_graph import *
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    UnivariateDiscreteLeaf,
)

np.random.seed(420)
random.seed(420)


class Target(IntEnum):
    A = 0
    B = 1


class RandomRegionGraphTestCase(unittest.TestCase):
    variables = SortedSet([Continuous(str(i)) for i in range(4)])

    region_graph = RegionGraph(variables, partitions=2, depth=1, repetitions=2)
    region_graph = region_graph.create_random_region_graph()

    def test_region_graph(self):
        self.assertEqual(len(self.region_graph.nodes()), 19)

    def test_as_jpc(self):
        model = self.region_graph.as_probabilistic_circuit(input_units=10, sum_units=5)
        nx_model = model.to_nx()
        # fig = go.Figure(nx_model.plot_structure(), nx_model.plotly_layout_structure())
        # fig.show()

        self.assertEqual(
            len(list(node for node in nx_model.nodes() if isinstance(node, SumUnit))),
            21,
        )


class RandomRegionGraphLearningTestCase(unittest.TestCase):
    variables = SortedSet(
        [Continuous(str(i)) for i in range(8)]
        + [Symbolic("target", Set.from_iterable(Target))]
    )
    region_graph = RegionGraph(variables, partitions=2, depth=2, repetitions=2)
    region_graph = region_graph.create_random_region_graph()

    def test_learning(self):
        data = np.random.uniform(0, 1, (100, len(self.variables)))
        data = jnp.array(data)
        model = self.region_graph.as_probabilistic_circuit(input_units=5, sum_units=5)
        model.fit(data, epochs=10, optimizer=optax.adamw(0.01))
        nx_model = model.to_nx()
        for node in nx_model.nodes():
            if isinstance(node, SumUnit):
                self.assertAlmostEqual(logsumexp(node.log_weights), 0.0)
            elif isinstance(node, UnivariateDiscreteLeaf):
                self.assertAlmostEqual(sum(node.distribution.probabilities), 1.0)
            elif isinstance(node, UnivariateContinuousLeaf):
                distribution: GaussianDistribution = node.distribution
                self.assertGreater(distribution.scale, 0.0)


class ClassificationTestCase(unittest.TestCase):
    features = SortedSet([Continuous(f"x{i}") for i in range(4)])
    target = Symbolic("target", Set.from_iterable(Target))
    region_graph = RegionGraph(
        features, partitions=2, depth=1, repetitions=6, classes=2
    )
    region_graph = region_graph.create_random_region_graph()

    def test_classification(self):
        data = np.random.uniform(0, 1, (100, len(self.features)))
        data = jnp.array(data)
        labels = np.random.randint(2, size=len(data))
        labels = jnp.array(labels)
        model = self.region_graph.as_probabilistic_circuit(input_units=5, sum_units=5)
        self.assertIsInstance(model, ClassificationCircuit)
        self.assertEqual(model.root.number_of_nodes, 2)
        model.fit(data, labels=labels, epochs=10, optimizer=optax.adamw(0.01))
        pc = model.as_probabilistic_circuit(self.target)
        self.assertIsInstance(pc, JPC)
        self.assertEqual(pc.variables, self.features | SortedSet([self.target]))
        nx_pc = pc.to_nx()
        self.assertTrue(nx_pc.is_decomposable())

        p_target = nx_pc.marginal(SortedSet([self.target]))
        probabilities = {
            str(element): p_target.probability_of_simple_event(
                SimpleEvent({self.target: element})
            )
            for element in self.target.domain
        }
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
