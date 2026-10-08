import unittest

import jax.numpy as jnp
import numpy as np
from sortedcontainers import SortedSet

from probabilistic_model.adapters.circuit_representations import (
    CircuitRepresentations,
)
from probabilistic_model.adapters.exceptions import (
    CannotConvertError,
    NoConversionPathError,
)
from probabilistic_model.adapters.jax_tensorized.jax_to_tensorized import (
    DifferentiableLayeredCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.jax_tensorized.tensorized_to_jax import (
    LayeredCircuitToDifferentiableLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.rustworkx_tensorized.tensorized_to_rustworkx import (
    LayeredCircuitToRustworkxCircuitConverter,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from .differentiable_circuits import single_precision_tolerance, w, x, y, z


def rustworkx_mixture_of_products() -> RustworkxProbabilisticCircuit:
    """
    :return: A rustworkx circuit over x and y that mixes two products of Gaussians.
    """
    circuit = RustworkxProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for weight, location in [(0.3, -1.0), (0.7, 2.0)]:
        product = ProductUnit(probabilistic_circuit=circuit)
        for variable in [x, y]:
            product.add_subcircuit(
                leaf(
                    GaussianDistribution(variable=variable, location=location, scale=1.0),
                    circuit,
                )
            )
        root.add_subcircuit(product, np.log(weight))
    return circuit


class CircuitRepresentationsTestCase(unittest.TestCase):
    """
    A circuit converts into any representation that a chain of converters of whole
    circuits reaches.
    """

    representations = CircuitRepresentations()

    def test_rustworkx_reaches_differentiable_through_layered(self):
        self.assertEqual(
            self.representations.converters_between(
                RustworkxProbabilisticCircuit, DifferentiableLayeredCircuit
            ),
            [
                RustworkxCircuitToLayeredCircuitConverter,
                LayeredCircuitToDifferentiableLayeredCircuitConverter,
            ],
        )

    def test_differentiable_reaches_rustworkx_through_layered(self):
        self.assertEqual(
            self.representations.converters_between(
                DifferentiableLayeredCircuit, RustworkxProbabilisticCircuit
            ),
            [
                DifferentiableLayeredCircuitToLayeredCircuitConverter,
                LayeredCircuitToRustworkxCircuitConverter,
            ],
        )

    def test_rustworkx_circuit_converts_into_the_same_differentiable_distribution(self):
        circuit = rustworkx_mixture_of_products()
        differentiable = self.representations.convert(
            circuit, DifferentiableLayeredCircuit
        )
        self.assertIsInstance(differentiable, DifferentiableLayeredCircuit)
        events = np.random.normal(size=(20, 2))
        np.testing.assert_allclose(
            np.asarray(differentiable.log_likelihood(jnp.asarray(events))),
            circuit.log_likelihood(events),
            rtol=single_precision_tolerance,
            atol=single_precision_tolerance,
        )

    def test_differentiable_circuit_converts_into_the_same_rustworkx_distribution(
        self,
    ):
        circuit = (
            RegionGraph(SortedSet([w, x, y, z]), partitions=2, depth=1, repetitions=2)
            .create_random_region_graph()
            .as_probabilistic_circuit(input_units=3, sum_units=3)
        )
        rustworkx = self.representations.convert(circuit, RustworkxProbabilisticCircuit)
        self.assertIsInstance(rustworkx, RustworkxProbabilisticCircuit)
        events = np.random.normal(size=(20, 4))
        np.testing.assert_allclose(
            rustworkx.log_likelihood(events),
            np.asarray(circuit.log_likelihood(jnp.asarray(events))),
            rtol=single_precision_tolerance,
            atol=single_precision_tolerance,
        )

    def test_circuit_in_the_requested_representation_is_returned_as_it_is(self):
        circuit = rustworkx_mixture_of_products()
        self.assertIs(
            self.representations.convert(circuit, RustworkxProbabilisticCircuit),
            circuit,
        )

    def test_subclass_of_a_representation_converts_like_it(self):
        classification_circuit = (
            RegionGraph(
                SortedSet([w, x, y, z]), partitions=2, depth=1, repetitions=2, classes=2
            )
            .create_random_region_graph()
            .as_probabilistic_circuit()
        )
        self.assertEqual(
            self.representations.converters_between(
                type(classification_circuit), LayeredProbabilisticCircuit
            ),
            [DifferentiableLayeredCircuitToLayeredCircuitConverter],
        )
        with self.assertRaises(CannotConvertError):
            self.representations.convert(
                classification_circuit, LayeredProbabilisticCircuit
            )

    def test_unreachable_representation_is_refused(self):
        with self.assertRaises(NoConversionPathError):
            self.representations.convert(rustworkx_mixture_of_products(), SortedSet)


if __name__ == "__main__":
    unittest.main()
