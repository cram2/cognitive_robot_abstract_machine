import unittest

import numpy as np
from random_events.interval import closed
from random_events.variable import Continuous, Integer

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    IntegerDistribution,
)
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    CopiesNotAlignedError,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.dirac_delta_layer import (
    DiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import QueryCache
from probabilistic_model.probabilistic_circuit.tensorized.stacked_copies import (
    AlignedCopiesStacker,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    StructuralQuery,
)
from probabilistic_model.utils import MissingDict


class StackingConditionedCopiesTestCase(unittest.TestCase):
    """
    Copies of one circuit conditioned on different points share their structure, so they
    stack into one layer graph whose root holds one node per copy.
    """

    def setUp(self):
        self.x = Continuous("x")
        self.n = Integer("n")
        circuit = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=circuit)
        for weight, (lower, upper), probabilities in (
            (0.3, (0.0, 1.0), {0: 0.9, 1: 0.1}),
            (0.7, (2.0, 3.0), {0: 0.2, 1: 0.8}),
        ):
            product = ProductUnit(probabilistic_circuit=circuit)
            root.add_subcircuit(product, np.log(weight))
            product.add_subcircuit(
                leaf(
                    UniformDistribution(
                        variable=self.x, interval=closed(lower, upper).simple_sets[0]
                    ),
                    circuit,
                )
            )
            product.add_subcircuit(
                leaf(
                    IntegerDistribution(
                        variable=self.n,
                        probabilities=MissingDict(float, probabilities),
                    ),
                    circuit,
                )
            )
        self.layered = RustworkxCircuitToLayeredCircuitConverter.convert(circuit)
        self.events = np.array([[0.0, 0.5], [0.0, 2.5], [0.0, 1.5]])

    def conditioned_on(self, value: int):
        """
        :param value: A value of ``n``.
        :return: The layers of the circuit conditioned on it, without ``n``.
        """
        conditioned = self.layered.root.log_conditional_of_point(
            {self.n: value}, StructuralQuery(self.layered.variables), cache=QueryCache()
        )
        kept = np.array([variable == self.x for variable in self.layered.variables])
        return conditioned.layer.marginal(kept)

    def test_node_of_every_copy_is_that_copy(self):
        copies = [self.conditioned_on(0), self.conditioned_on(1)]
        stacked = AlignedCopiesStacker().stack(copies)
        stacked.layer.normalize()
        log_likelihoods = stacked.layer.log_likelihood_of_nodes(self.events)
        for index, copy in enumerate(copies):
            copy.normalize()
            np.testing.assert_allclose(
                log_likelihoods[:, index],
                copy.log_likelihood_of_nodes(self.events)[:, 0],
            )

    def test_layers_that_conditioning_leaves_unchanged_are_shared(self):
        copies = [self.conditioned_on(0), self.conditioned_on(1)]
        stacked = AlignedCopiesStacker().stack(copies)
        self.assertFalse(stacked.is_shared)
        [uniform_layer] = [
            layer
            for layer in stacked.layer.all_layers()
            if isinstance(layer, UniformLayer)
        ]
        [original_uniform_layer] = [
            layer for layer in self.layered.layers if isinstance(layer, UniformLayer)
        ]
        self.assertEqual(
            uniform_layer.number_of_nodes, original_uniform_layer.number_of_nodes
        )

    def test_equal_copies_are_one_shared_copy(self):
        stacked = AlignedCopiesStacker().stack(
            [self.layered.root, self.layered.root.__deepcopy__({})]
        )
        self.assertTrue(stacked.is_shared)
        self.assertEqual(stacked.layer.number_of_nodes, 1)

    def test_copies_of_different_structure_are_rejected(self):
        with self.assertRaises(CopiesNotAlignedError):
            AlignedCopiesStacker().stack(
                [self.layered.root, self.layered.root.child_layers[0]]
            )


class AttachingAChildLayerTestCase(unittest.TestCase):
    """
    Some nodes of a product layer multiply a node of a layer that is added later.
    """

    def test_only_the_given_nodes_multiply_the_new_child_layer(self):
        x, y = Continuous("x"), Continuous("y")
        uniform_layer = UniformLayer.from_distributions(
            0,
            [
                UniformDistribution(
                    variable=x, interval=closed(0.0, 1.0).simple_sets[0]
                ),
                UniformDistribution(
                    variable=x, interval=closed(0.0, 2.0).simple_sets[0]
                ),
            ],
        )
        product_layer = ProductLayer.node_wise_product_of([uniform_layer])
        point_mass = DiracDeltaLayer.from_distributions(
            1, [DiracDeltaDistribution(variable=y, location=3.0, density_cap=2.0)]
        )
        product_layer.attach_child_layer(point_mass, np.array([1]), np.array([0]))

        events = np.array([[0.5, 3.0]])
        np.testing.assert_allclose(
            product_layer.log_likelihood_of_nodes(events),
            [[np.log(1.0), np.log(0.5) + np.log(2.0)]],
        )
        np.testing.assert_array_equal(product_layer.variables, [0, 1])


class PointMassesLayerTestCase(unittest.TestCase):
    """
    Several point masses on one variable become one input layer with a node each.
    """

    def test_every_node_puts_its_mass_on_its_value(self):
        n = Integer("n")
        values = [0, 2]
        layer = LayeredProbabilisticCircuit.point_masses_layer(
            0,
            [
                IntegerDistribution(
                    variable=n, probabilities=MissingDict(float, {value: 1.0})
                )
                for value in values
            ],
        )
        np.testing.assert_array_equal(
            layer.log_likelihood_of_nodes(np.array([[value] for value in values])),
            [[0.0, -np.inf], [-np.inf, 0.0]],
        )


if __name__ == "__main__":
    unittest.main()
