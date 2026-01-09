import unittest

from matplotlib import pyplot as plt
from random_events.interval import closed
from random_events.variable import Continuous
from typing_extensions import Union

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
import plotly.graph_objects as go


class NormalizationTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")

    def test_normalization(self):
        pc = ProbabilisticCircuit()
        u1 = leaf(UniformDistribution(self.x, closed(0, 1).simple_sets[0]), pc)
        u2 = leaf(UniformDistribution(self.x, closed(3, 4).simple_sets[0]), pc)
        sum_unit = SumUnit(probabilistic_circuit=pc)
        sum_unit.add_subcircuit(u1, np.log(0.5))
        sum_unit.add_subcircuit(u2, np.log(0.3))
        sum_unit.normalize()
        self.assertAlmostEqual(sum_unit.log_weights[1], np.log(0.5 / 0.8))
        self.assertAlmostEqual(sum_unit.log_weights[0], np.log(0.3 / 0.8))
        traces = sum_unit.probabilistic_circuit.plot()
        self.assertGreater(len(traces), 0)
        # go.Figure(traces, sum_unit.probabilistic_circuit.plotly_layout()).show()


class SumUnitTestCase(unittest.TestCase):
    x: Continuous = Continuous("x")
    model: ProbabilisticCircuit

    def setUp(self):
        pc = ProbabilisticCircuit()
        u1 = leaf(UniformDistribution(self.x, closed(0, 1).simple_sets[0]), pc)
        u2 = leaf(UniformDistribution(self.x, closed(3, 4).simple_sets[0]), pc)

        model = SumUnit(probabilistic_circuit=pc)
        model.add_subcircuit(u1, np.log(0.6))
        model.add_subcircuit(u2, np.log(0.4))
        self.model = model.probabilistic_circuit

    def test_setup(self):
        self.assertEqual(len(list(self.model.nodes())), 3)
        self.assertEqual(len(self.model.edges()), 2)

    def test_variables(self):
        self.assertEqual(self.model.variables, SortedSet([self.x]))

    def test_latent_variable(self):
        self.assertEqual(len(self.model.root.latent_variable.domain.simple_sets), 2)

    def test_domain(self):
        domain = self.model.support
        domain_by_hand = SimpleEvent(
            {self.x: closed(0, 1) | closed(3, 4)}
        ).as_composite_set()
        self.assertEqual(domain, domain_by_hand)

    def test_weighted_subcircuits(self):
        weighted_subcircuits = self.model.root.log_weighted_subcircuits
        self.assertEqual(len(weighted_subcircuits), 2)
        self.assertEqual(
            [weighted_subcircuit[0] for weighted_subcircuit in weighted_subcircuits],
            [np.log(0.4), np.log(0.6)],
        )

    def test_likelihood(self):
        event = np.array([[0.5]])
        result = self.model.likelihood(event)
        self.assertEqual(result, 0.6)

    def test_probability(self):
        event = SimpleEvent({self.x: closed(0, 3.5)}).as_composite_set()
        result = self.model.probability(event)
        self.assertEqual(result, 0.8)

    def test_conditional(self):
        event = SimpleEvent({self.x: closed(0, 0.5)}).as_composite_set()
        result, probability = self.model.truncated(event)
        self.assertAlmostEqual(probability, 0.3)
        self.assertEqual(len(list(result.nodes())), 1)
        self.assertIsInstance(result.root, LeafUnit)

    def test_conditional_impossible(self):
        event = SimpleEvent({self.x: closed(5, 6)}).as_composite_set()
        result, probability = self.model.truncated(event)
        self.assertEqual(probability, 0.0)
        self.assertIsNone(result)

    def test_sample(self):
        samples = self.model.sample(100)
        likelihoods = self.model.likelihood(samples)
        self.assertTrue(all(likelihoods > 0))

    def test_moment(self):
        expectation = self.model.expectation(self.model.variables)
        self.assertEqual(expectation[self.x], 0.5 * 0.6 + 0.4 * 3.5)

    def test_marginal(self):
        marginal = self.model.marginal([self.x])
        self.assertEqual(len(marginal.edges()), len(self.model.edges()))
        self.assertEqual(len(marginal.nodes()), len(self.model.nodes()))

    def test_mode(self):
        mode, likelihood = self.model.mode()
        self.assertEqual(likelihood, 0.6)
        self.assertEqual(mode, SimpleEvent({self.x: closed(0, 1)}).as_composite_set())

    def test_serialization(self):
        event = SimpleEvent({self.x: closed(0, 0.5)}).as_composite_set()
        serialized = self.model.to_json()
        deserialized = ProbabilisticCircuit.from_json(serialized)
        self.assertEqual(self.model.probability(event), deserialized.probability(event))

    def test_conditional_inference(self):
        event = SimpleEvent({self.x: closed(0, 0.5)}).as_composite_set()
        result, probability = self.model.truncated(event)
        self.assertEqual(result.probability(event), 1)

    def test_deep_mount(self):
        pc = ProbabilisticCircuit()
        s1 = SumUnit(probabilistic_circuit=ProbabilisticCircuit())
        s2 = SumUnit(probabilistic_circuit=pc)
        s3 = SumUnit(probabilistic_circuit=pc)
        u1 = leaf(UniformDistribution(self.x, closed(0, 1).simple_sets[0]), pc)
        s2.probabilistic_circuit.add_nodes_from([s2, s3, u1])
        s2.probabilistic_circuit.add_edges_from([(s2, s3, 1.0), (s3, u1, 1.0)])
        s1.probabilistic_circuit.mount(s2)
        self.assertEqual(len(list(s1.probabilistic_circuit.nodes())), 4)

    def test_sample_not_equal(self):
        samples = self.model.sample(10)
        uniques = np.unique(samples, axis=0)
        self.assertEqual(len(samples), len(uniques))

    def test_determinism(self):
        self.assertTrue(self.model.is_deterministic())
