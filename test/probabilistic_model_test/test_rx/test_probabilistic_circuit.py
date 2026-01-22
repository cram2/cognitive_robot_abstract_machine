import copy
import unittest

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random_events.interval
from random_events.interval import closed, singleton
from sklearn.gaussian_process.kernels import Product

from probabilistic_model.distributions import (
    GaussianDistribution,
    DiracDeltaDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
from probabilistic_model.utils import MissingDict


class SymbolEnum(IntEnum):
    A = 0
    B = 1
    C = 2


class SmallCircuitTestCast(unittest.TestCase):
    """
    Integration test for all classes in probabilistic circuits.
    """

    x = Continuous("x")
    y = Continuous("y")

    model: ProbabilisticCircuit

    def setUp(self):
        model = ProbabilisticCircuit()
        sum1, sum2, sum3 = (
            SumUnit(probabilistic_circuit=model),
            SumUnit(probabilistic_circuit=model),
            SumUnit(probabilistic_circuit=model),
        )
        sum4, sum5 = SumUnit(probabilistic_circuit=model), SumUnit(
            probabilistic_circuit=model
        )
        prod1, prod2 = ProductUnit(probabilistic_circuit=model), ProductUnit(
            probabilistic_circuit=model
        )

        sum1.add_subcircuit(prod1, np.log(0.5))
        sum1.add_subcircuit(prod2, np.log(0.5))
        prod1.add_subcircuit(sum2)
        prod1.add_subcircuit(sum4)
        prod2.add_subcircuit(sum3)
        prod2.add_subcircuit(sum5)

        d_x1 = leaf(
            UniformDistribution(self.x, SimpleInterval(0, 1)),
            probabilistic_circuit=model,
        )
        d_x2 = leaf(
            UniformDistribution(self.x, SimpleInterval(2, 3)),
            probabilistic_circuit=model,
        )
        d_y1 = leaf(
            UniformDistribution(self.y, SimpleInterval(0, 1)),
            probabilistic_circuit=model,
        )
        d_y2 = leaf(
            UniformDistribution(self.y, SimpleInterval(3, 4)),
            probabilistic_circuit=model,
        )

        sum2.add_subcircuit(d_x1, np.log(0.8))
        sum2.add_subcircuit(d_x2, np.log(0.2))
        sum3.add_subcircuit(d_x1, np.log(0.7))
        sum3.add_subcircuit(d_x2, np.log(0.3))

        sum4.add_subcircuit(d_y1, np.log(0.5))
        sum4.add_subcircuit(d_y2, np.log(0.5))
        sum5.add_subcircuit(d_y1, np.log(0.1))
        sum5.add_subcircuit(d_y2, np.log(0.9))

        self.model = sum1.probabilistic_circuit

    def test_sampling(self):
        samples = self.model.sample(100)
        unique = np.unique(samples, axis=0)
        self.assertGreater(len(unique), 95)

    def test_truncation(self):
        event = SimpleEvent(
            {self.x: closed(0, 0.25) | closed(0.5, 0.75)}
        ).as_composite_set()
        conditional, prob = self.model.truncated(event)
        self.assertAlmostEqual(prob, 0.375)
        conditional.plot_structure()
        # plt.show()

    def test_plot(self):
        self.model.log_likelihood(np.array([[0.5, 0.5]]))
        color_map = {
            self.model.root: "red",
            self.model.root.subcircuits[0]: "blue",
            self.model.leaves[0]: "green",
        }
        self.model.plot_structure(
            color_map,
            plot_inference=True,
            inference_representation=lambda node: round(
                node.result_of_current_query[0].item(), 2
            ),
        )
        # plt.show()

    def test_translation(self):
        translation = {self.x: 5, self.y: 10}
        self.model.translate(translation)
        event = SimpleEvent(
            {self.x: closed(5, 5.25) | closed(5.5, 5.75)}
        ).as_composite_set()
        probability = self.model.probability(event)
        self.assertAlmostEqual(probability, 0.375)

    def test_copy(self):
        copied = self.model.__deepcopy__()
        self.assertTrue(copied is not self.model)
        self.assertTrue(copied.graph is not self.model.graph)
        self.assertTrue(copied.is_valid())
        self.assertEqual(len(copied.nodes()), len(self.model.nodes()))
        self.assertEqual(len(copied.graph.edges()), len(self.model.graph.edges()))


class SymbolicPlottingTestCase(unittest.TestCase):
    x = Symbolic("x", Set.from_iterable(SymbolEnum))
    model: ProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        probabilities = MissingDict(float)
        probabilities[int(SymbolEnum.A)] = 7 / 20
        probabilities[int(SymbolEnum.B)] = 13 / 20
        cls.model = ProbabilisticCircuit()
        l1 = UnivariateDiscreteLeaf(SymbolicDistribution(cls.x, probabilities))
        cls.model.add_node(l1)

    def test_plot(self):
        fig = go.Figure(self.model.plot(), self.model.plotly_layout())
        # fig.show()


class ConditioningWithOrphansTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")

    prod = ProductUnit(probabilistic_circuit=ProbabilisticCircuit())
    px = UnivariateContinuousLeaf(GaussianDistribution(x, 1, 1))
    py = UnivariateContinuousLeaf(GaussianDistribution(y, 1, 1))

    prod.add_subcircuit(px)
    prod.add_subcircuit(py)

    model = prod.probabilistic_circuit

    def test_conditioning(self):
        event = SimpleEvent(
            {
                self.x: closed(1.191472053527832, 1.2024999856948853),
                self.y: random_events.interval.open(1500.0009765625, np.inf)
                | random_events.interval.open(-np.inf, -1500.0009765625),
            }
        ).as_composite_set()

        result, probability = self.model.truncated(event)
        self.assertIsNone(result)


class DiracMixtureConditioningTestCase(unittest.TestCase):
    x = Continuous("x")

    model: ProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        cls.model = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=cls.model)
        root.add_subcircuit(
            leaf(UniformDistribution(cls.x, SimpleInterval(0, 1.0)), cls.model),
            np.log(0.5),
        )
        root.add_subcircuit(
            leaf(DiracDeltaDistribution(cls.x, 0.5, 2.0), cls.model), np.log(0.5)
        )

    def test_conditioning(self):
        event = SimpleEvent({self.x: closed(0.0, 1.0)}).as_composite_set()
        conditional, probability = self.model.truncated(event)
        self.assertAlmostEqual(probability, 1.0)

    def test_conditioning_without_dirac(self):
        event = SimpleEvent(
            {self.x: closed(0.0, 0.25) | closed(0.75, 1.0)}
        ).as_composite_set()

        conditional, probability = self.model.truncated(event)
        self.assertAlmostEqual(probability, 0.25)
        self.assertEqual(len(list(conditional.nodes())), 3)
        self.assertTrue(
            all(
                [
                    isinstance(node.distribution, UniformDistribution)
                    for node in conditional.leaves
                ]
            )
        )

    def test_conditioning_singleton(self):
        event = SimpleEvent({self.x: singleton(0.5)}).as_composite_set()

        conditional, probability = self.model.truncated(event)
        self.assertEqual(len(list(conditional.nodes())), 1)
        self.assertIsInstance(conditional.root.distribution, DiracDeltaDistribution)

        conditional, probability = self.model.conditional({self.x: 0.5})
        self.assertAlmostEqual(probability, 1.5)
        self.assertEqual(len(list(conditional.nodes())), 1)
        self.assertTrue(
            all(
                [
                    isinstance(node.distribution, DiracDeltaDistribution)
                    for node in conditional.leaves
                ]
            )
        )


class ConditioningTestCase(unittest.TestCase):
    x = Continuous("x")
    y = Continuous("y")

    model: ProbabilisticCircuit

    @classmethod
    def setUpClass(cls):
        model = ProbabilisticCircuit()
        s1 = SumUnit(probabilistic_circuit=model)
        p1 = ProductUnit(probabilistic_circuit=model)
        p2 = ProductUnit(probabilistic_circuit=model)

        u1 = leaf(UniformDistribution(cls.x, SimpleInterval(0, 1.0)), model)
        u2 = leaf(UniformDistribution(cls.x, SimpleInterval(0.0, 2)), model)
        u3 = leaf(UniformDistribution(cls.y, SimpleInterval(0, 1)), model)
        u4 = leaf(UniformDistribution(cls.y, SimpleInterval(0.0, 2)), model)

        s1.add_subcircuit(p1, np.log(0.5))
        s1.add_subcircuit(p2, np.log(0.5))

        p1.add_subcircuit(u1)
        p1.add_subcircuit(u3)

        p2.add_subcircuit(u2)
        p2.add_subcircuit(u4)
        cls.model = model

    def test_conditioning(self):
        p = {self.x: 0.5}

        marginal = self.model.marginal([self.y])

        model, _ = self.model.conditional(p)
        # model, _ = self.model.truncated(SimpleEvent({self.x: closed(0.3, 0.5)}).as_composite_set())

        conditioned_marginal = model.marginal([self.y])

        probability_event = SimpleEvent({self.y: closed(0.0, 1.0)}).as_composite_set()

        p_marginal = marginal.probability(probability_event)
        p_conditioned_marginal = conditioned_marginal.probability(probability_event)
        self.assertGreater(p_conditioned_marginal, p_marginal)

    def test_conditioning_with_symbolic(self):

        model = copy.deepcopy(self.model)

        s = Symbolic("s", Set.from_iterable(SymbolEnum))
        probabilities = MissingDict(float)
        probabilities[hash(SymbolEnum.A)] = 7 / 20
        probabilities[hash(SymbolEnum.B)] = 13 / 20

        old_root = model.root
        new_root = ProductUnit(probabilistic_circuit=model)
        p_s = leaf(SymbolicDistribution(s, probabilities), model)
        new_root.add_subcircuit(
            old_root,
        )
        new_root.add_subcircuit(
            p_s,
        )

        model.conditional({s: SymbolEnum.A})


if __name__ == "__main__":
    unittest.main()
