import math
import unittest
from enum import IntEnum

from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from sortedcontainers import SortedSet

from probabilistic_model.bayesian_network.bayesian_network import *
from probabilistic_model.distributions import UniformDistribution, SymbolicDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProductUnit,
)
from probabilistic_model.utils import MissingDict
from random_events.set import Set


class YEnum(IntEnum):
    ZERO = 0
    ONE = 1


class XEnum(IntEnum):
    ZERO = 0
    ONE = 1
    TWO = 2


class MinimalBayesianNetworkTestCase(unittest.TestCase):
    model: BayesianNetwork
    x: Symbolic = Symbolic("x", Set.from_iterable(XEnum))
    y: Symbolic = Symbolic("y", Set.from_iterable(YEnum))
    p_x: Root
    p_yx: ConditionalProbabilityTable

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        # create the root distribution for x
        self.p_x = Root(
            SymbolicDistribution(
                self.x,
                MissingDict(
                    float, zip([XEnum.ZERO, XEnum.ONE, XEnum.TWO], [0.5, 0.3, 0.2])
                ),
            ),
            bayesian_network=self.model,
        )

        self.p_yx = ConditionalProbabilityTable(bayesian_network=self.model)
        # create the truncated probability table for y
        self.p_yx.conditional_probability_distributions[XEnum.ZERO] = (
            SymbolicDistribution(self.y, MissingDict(float, zip([0, 1], [0.5, 0.5])))
        )
        self.p_yx.conditional_probability_distributions[XEnum.ONE] = (
            SymbolicDistribution(
                self.y, MissingDict(float, zip([YEnum.ZERO, YEnum.ONE], [0.3, 0.7]))
            )
        )
        self.p_yx.conditional_probability_distributions[XEnum.TWO] = (
            SymbolicDistribution(
                self.y, MissingDict(float, zip([YEnum.ZERO, YEnum.ONE], [0.1, 0.9]))
            )
        )

        # add the edge between x and y
        self.model.add_edge(self.p_x, self.p_yx)

    def test_setup(self):
        self.assertEqual(len(self.model.nodes()), 2)
        self.assertEqual(len(self.model.edges()), 1)

    def test_parents(self):
        self.assertEqual(self.p_yx.parent, self.p_x)

    def test_as_probabilistic_circuit(self):
        circuit = self.model.as_probabilistic_circuit()
        self.assertIsNotNone(circuit)


class ComplexBayesianNetworkTestCase(unittest.TestCase):
    model: BayesianNetwork
    x: Symbolic = Symbolic("x", Set.from_iterable(XEnum))
    y: Symbolic = Symbolic("y", Set.from_iterable(YEnum))
    z: Symbolic = Symbolic("z", Set.from_iterable(XEnum))
    a: Symbolic = Symbolic("a", Set.from_iterable(YEnum))

    d_x: Root

    def setUp(self):
        np.random.seed(69)

        self.model = BayesianNetwork()

        self.d_x = Root(
            SymbolicDistribution(
                self.x,
                MissingDict(
                    float, zip([XEnum.ZERO, XEnum.ONE, XEnum.TWO], [0.5, 0.3, 0.2])
                ),
            ),
            bayesian_network=self.model,
        )

        self.d_yx = ConditionalProbabilityTable(bayesian_network=self.model)
        self.d_zx = ConditionalProbabilityTable(bayesian_network=self.model)
        self.d_az = ConditionalProbabilityTable(bayesian_network=self.model)

        self.d_yx.conditional_probability_distributions[0] = SymbolicDistribution(
            self.y, MissingDict(float, zip([YEnum.ZERO, YEnum.ONE], [0.5, 0.5]))
        )
        self.d_yx.conditional_probability_distributions[1] = SymbolicDistribution(
            self.y, MissingDict(float, zip([YEnum.ZERO, YEnum.ONE], [0.6, 0.4]))
        )
        self.d_yx.conditional_probability_distributions[2] = SymbolicDistribution(
            self.y, MissingDict(float, zip([YEnum.ZERO, YEnum.ONE], [0.7, 0.3]))
        )

        self.d_zx.conditional_probability_distributions[0] = SymbolicDistribution(
            self.z, MissingDict(float, zip([XEnum.ZERO, XEnum.ONE], [0.1, 0.9]))
        )
        self.d_zx.conditional_probability_distributions[1] = SymbolicDistribution(
            self.z, MissingDict(float, zip([XEnum.ZERO, XEnum.ONE], [1, 0]))
        )
        self.d_zx.conditional_probability_distributions[2] = SymbolicDistribution(
            self.z, MissingDict(float, zip([XEnum.ZERO, XEnum.ONE], [0.8, 0.2]))
        )

        self.d_az.conditional_probability_distributions[0] = SymbolicDistribution(
            self.a, MissingDict(float, zip([YEnum.ZERO, YEnum.ONE], [0.3, 0.7]))
        )
        self.d_az.conditional_probability_distributions[1] = SymbolicDistribution(
            self.a, MissingDict(float, zip([YEnum.ZERO, YEnum.ONE], [0.6, 0.4]))
        )

        self.model.add_edge(self.d_x, self.d_yx)
        self.model.add_edge(self.d_x, self.d_zx)
        self.model.add_edge(self.d_zx, self.d_az)

    def test_as_probabilistic_circuit(self):
        circuit = self.model.as_probabilistic_circuit().simplify()
        # circuit.plot_structure()
        # plt.show()
        self.assertEqual(circuit.variables, SortedSet([self.x, self.y, self.z, self.a]))
        self.assertLess(
            len(circuit.edges()),
            math.prod([len(v.domain.simple_sets) for v in circuit.variables]),
        )
        assert circuit.is_valid()


class BayesianNetworkWithCircuitTestCase(unittest.TestCase):
    x: Symbolic = Symbolic("x", Set.from_iterable(YEnum))
    y: Continuous = Continuous("y")
    z: Continuous = Continuous("z")
    p_x: Root

    bayesian_network: BayesianNetwork

    def setUp(self):
        self.bayesian_network = BayesianNetwork()
        self.p_x = Root(
            SymbolicDistribution(
                self.x, MissingDict(float, zip([YEnum.ZERO, YEnum.ONE], [0.7, 0.3]))
            ),
            bayesian_network=self.bayesian_network,
        )

        self.p_yzx = ConditionalProbabilisticCircuit(
            bayesian_network=self.bayesian_network
        )

        pc1 = ProbabilisticCircuit()
        d1 = ProductUnit(probabilistic_circuit=pc1)
        d1.add_subcircuit(
            leaf(UniformDistribution(self.y, closed(0, 1).simple_sets[0]), pc1)
        )
        d1.add_subcircuit(
            leaf(UniformDistribution(self.z, closed(0, 1).simple_sets[0]), pc1)
        )

        pc2 = ProbabilisticCircuit()
        d2 = ProductUnit(probabilistic_circuit=pc2)
        d2.add_subcircuit(
            leaf(UniformDistribution(self.y, closed(0, 2).simple_sets[0]), pc2)
        )
        d2.add_subcircuit(
            leaf(UniformDistribution(self.z, closed(0, 3).simple_sets[0]), pc2)
        )

        self.p_yzx.conditional_probability_distributions[YEnum.ZERO] = pc1
        self.p_yzx.conditional_probability_distributions[YEnum.ONE] = pc2

        self.bayesian_network.add_nodes_from([self.p_x, self.p_yzx])
        self.bayesian_network.add_edge(self.p_x, self.p_yzx)

    def test_as_probabilistic_circuit(self):
        circuit = self.bayesian_network.as_probabilistic_circuit()
        self.assertEqual(
            circuit.probability(circuit.universal_simple_event().as_composite_set()),
            1.0,
        )
        event = SimpleEvent({self.x: (XEnum.ZERO, XEnum(1)), self.y: closed(1.5, 2)})
        self.assertAlmostEqual(
            0.3 * 0.25, circuit.probability(event.as_composite_set())
        )

    def test_plot(self):
        ...
        # self.bayesian_network.plot()


if __name__ == "__main__":
    unittest.main()
