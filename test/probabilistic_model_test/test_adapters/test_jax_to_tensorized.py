import unittest

import jax.numpy as jnp
import numpy as np
from random_events.interval import Bound
from scipy.special import logsumexp
from sortedcontainers import SortedSet

from probabilistic_model.adapters.exceptions import CannotConvertError
from probabilistic_model.adapters.jax_tensorized.converter import (
    JaxToTensorizedConverter,
)
from probabilistic_model.adapters.jax_tensorized.jax_to_tensorized import (
    DifferentiableLayeredCircuitToLayeredCircuitConverter,
    LayersToTensorized,
)
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import (
    DifferentiableDiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableDenseSumLayer,
    DifferentiableProductLayer,
    DifferentiableSparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableDiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ClassificationCircuit,
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.jax.uniform_layer import (
    DifferentiableUniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.dirac_delta_layer import (
    DiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    IntegerLayer,
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)

from .differentiable_circuits import (
    Color,
    Size,
    color,
    count,
    dense_log_weights,
    gaussian_layer,
    mixture_of_products,
    product_edges,
    region_graph_circuit,
    single_precision_tolerance,
    size,
    sparse_matrix,
    w,
    x,
    y,
    z,
)


def conversion_over(*variables) -> LayersToTensorized:
    """
    :param variables: The variables the converted layers index.
    :return: A conversion of differentiable layers into layered circuit layers.
    """
    return LayersToTensorized(JaxToTensorizedConverter, SortedSet(variables))


# %% inner layers


class SumLayerConversionTestCase(unittest.TestCase):
    """
    The weight matrices of a differentiable sum layer, one per child layer, become one
    normalized weight matrix whose columns are the nodes of all child layers in order.
    """

    def assert_normalized_weights(self, layer: SumLayer, unnormalized: np.ndarray):
        expected = unnormalized - logsumexp(unnormalized, axis=1, keepdims=True)
        np.testing.assert_allclose(dense_log_weights(layer), expected, atol=1e-6)

    def test_sparse_weights_are_normalized_over_all_child_layers(self):
        sum_layer = DifferentiableSparseSumLayer(
            [gaussian_layer(0), gaussian_layer(0)],
            [
                sparse_matrix([(0, 0, 0.0), (0, 1, 1.0), (1, 1, 2.0)], (2, 2)),
                sparse_matrix([(0, 0, 0.5), (1, 0, -1.0), (1, 1, 0.0)], (2, 2)),
            ],
        )
        layer = conversion_over(x).convert(sum_layer)
        self.assertIsInstance(layer, SumLayer)
        self.assert_normalized_weights(
            layer,
            np.array([[0.0, 1.0, 0.5, -np.inf], [-np.inf, 2.0, -1.0, 0.0]]),
        )

    def test_dense_weights_are_normalized_over_all_child_layers(self):
        first_weights = np.array([[0.0, 1.0], [2.0, 0.5]])
        second_weights = np.array([[-1.0, 0.0], [0.0, 0.0]])
        sum_layer = DifferentiableDenseSumLayer(
            [gaussian_layer(0), gaussian_layer(0)],
            [jnp.asarray(first_weights), jnp.asarray(second_weights)],
        )
        layer = conversion_over(x).convert(sum_layer)
        self.assertIsInstance(layer, SumLayer)
        self.assert_normalized_weights(
            layer, np.concatenate([first_weights, second_weights], axis=1)
        )

    def test_child_layers_keep_their_order(self):
        first, second = gaussian_layer(0), gaussian_layer(0)
        sum_layer = DifferentiableSparseSumLayer(
            [first, second],
            [
                sparse_matrix([(0, 0, 0.0)], (1, 2)),
                sparse_matrix([(0, 1, 0.0)], (1, 2)),
            ],
        )
        conversion = conversion_over(x)
        layer = conversion.convert(sum_layer)
        self.assertEqual(
            layer.child_layers, [conversion.convert(first), conversion.convert(second)]
        )


class ProductLayerConversionTestCase(unittest.TestCase):
    """
    The edges of a differentiable product layer keep their child node indices.
    """

    def test_edges_keep_their_child_nodes(self):
        edges = product_edges([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], (2, 2))
        product = DifferentiableProductLayer(
            [gaussian_layer(0), gaussian_layer(1)], edges
        )
        layer = conversion_over(x, y).convert(product)
        self.assertIsInstance(layer, ProductLayer)
        np.testing.assert_array_equal(layer.edges.toarray(), np.asarray(edges.todense()))


class LayerConversionTestCase(unittest.TestCase):
    """
    A conversion converts every layer once, so that a layer shared by several parents
    stays shared.
    """

    def test_a_layer_is_converted_once(self):
        conversion = conversion_over(x)
        shared = gaussian_layer(0)
        self.assertIs(conversion.convert(shared), conversion.convert(shared))


# %% input layers


class InputLayerConversionTestCase(unittest.TestCase):
    """
    Every differentiable input layer becomes the layered circuit layer of the same
    distributions.
    """

    def test_gaussian_scale_includes_the_minimum_scale(self):
        jax_layer = gaussian_layer(0)
        layer = conversion_over(x).convert(jax_layer)
        self.assertIsInstance(layer, GaussianLayer)
        np.testing.assert_allclose(layer.location, np.asarray(jax_layer.location))
        np.testing.assert_allclose(layer.scale, np.asarray(jax_layer.scale))

    def test_uniform_keeps_its_intervals_as_open_intervals(self):
        jax_layer = DifferentiableUniformLayer(0, jnp.array([[-1.0, 1.0], [0.0, 4.0]]))
        layer = conversion_over(x).convert(jax_layer)
        self.assertIsInstance(layer, UniformLayer)
        np.testing.assert_allclose(layer.interval, np.asarray(jax_layer.interval))
        self.assertTrue((layer.bounds == int(Bound.OPEN)).all())

    def test_dirac_delta_keeps_location_and_density_cap(self):
        jax_layer = DifferentiableDiracDeltaLayer(
            0, jnp.array([0.5, 1.5]), jnp.array([2.0, 3.0])
        )
        layer = conversion_over(x).convert(jax_layer)
        self.assertIsInstance(layer, DiracDeltaLayer)
        np.testing.assert_allclose(layer.location, np.asarray(jax_layer.location))
        np.testing.assert_allclose(layer.density_cap, np.asarray(jax_layer.density_cap))

    def test_discrete_layer_over_an_integer_variable_becomes_an_integer_layer(self):
        jax_layer = DifferentiableDiscreteLayer(
            0, jnp.log(jnp.array([[1.0, 2.0, 1.0], [0.0, 1.0, 3.0]]))
        )
        layer = conversion_over(count).convert(jax_layer)
        self.assertIsInstance(layer, IntegerLayer)
        np.testing.assert_array_equal(layer.states, np.arange(3))
        np.testing.assert_allclose(
            layer.table.dense_probabilities(),
            np.exp(np.asarray(jax_layer.normalized_log_probabilities)),
            rtol=single_precision_tolerance,
        )

    def test_symbolic_column_is_the_hash_of_the_domain_element(self):
        jax_layer = DifferentiableDiscreteLayer(
            0, jnp.log(jnp.array([[1.0, 2.0, 5.0], [4.0, 1.0, 3.0]]))
        )
        layer = conversion_over(size).convert(jax_layer)
        self.assertIsInstance(layer, SymbolicLayer)
        columns = SymbolicEncoding(size).hashes.astype(int)
        np.testing.assert_allclose(
            layer.table.dense_probabilities(),
            np.exp(np.asarray(jax_layer.normalized_log_probabilities))[:, columns],
            rtol=single_precision_tolerance,
        )

    def test_discrete_layer_over_a_continuous_variable_is_refused(self):
        jax_layer = DifferentiableDiscreteLayer(0, jnp.log(jnp.array([[0.5, 0.5]])))
        with self.assertRaises(CannotConvertError):
            conversion_over(x).convert(jax_layer)


# %% circuit


class DifferentiableCircuitConversionTestCase(unittest.TestCase):
    """
    A trained differentiable circuit converts into a layered circuit that expresses the
    same distribution, so that every query of the layered circuit answers for it.
    """

    def assert_same_log_likelihoods(
        self,
        jax_circuit: DifferentiableLayeredCircuit,
        numpy_circuit: LayeredProbabilisticCircuit,
        events: np.ndarray,
    ):
        expected = np.asarray(jax_circuit.log_likelihood(jnp.asarray(events)))
        np.testing.assert_allclose(
            numpy_circuit.log_likelihood(events),
            expected,
            rtol=single_precision_tolerance,
            atol=single_precision_tolerance,
        )

    def test_region_graph_circuit_has_the_same_log_likelihoods(self):
        jax_circuit = region_graph_circuit(SortedSet([w, x, y, z]))
        numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            jax_circuit
        )
        self.assert_same_log_likelihoods(
            jax_circuit, numpy_circuit, np.random.normal(size=(50, 4))
        )

    def test_region_graph_circuit_with_symbolic_variable_has_the_same_log_likelihoods(
        self,
    ):
        variables = SortedSet([w, x, y, color])
        jax_circuit = region_graph_circuit(variables)
        numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            jax_circuit
        )
        events = np.random.normal(size=(50, 4))
        events[:, variables.index(color)] = np.random.randint(0, len(Color), 50)
        self.assert_same_log_likelihoods(jax_circuit, numpy_circuit, events)

    def test_result_keeps_the_variables(self):
        jax_circuit = region_graph_circuit(SortedSet([w, x, y, z]))
        numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            jax_circuit
        )
        self.assertEqual(numpy_circuit.variables, jax_circuit.variables)

    def test_sum_weights_are_normalized(self):
        jax_circuit = mixture_of_products(gaussian_layer(0), gaussian_layer(1))
        numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            jax_circuit
        )
        np.testing.assert_allclose(
            numpy_circuit.root.log_normalization_constants, 0.0, atol=1e-12
        )

    def test_uniform_and_dirac_delta_layers_have_the_same_log_likelihoods(self):
        uniforms = DifferentiableUniformLayer(0, jnp.array([[-1.0, 1.0], [0.0, 4.0]]))
        dirac_deltas = DifferentiableDiracDeltaLayer(
            1, jnp.array([0.5, 1.5]), jnp.array([2.0, 3.0])
        )
        jax_circuit = mixture_of_products(uniforms, dirac_deltas)
        numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            jax_circuit
        )
        events = np.array([[0.5, 0.5], [0.5, 1.5], [3.0, 1.5], [-0.5, 0.5]])
        self.assert_same_log_likelihoods(jax_circuit, numpy_circuit, events)

    def test_integer_variable_has_the_same_log_likelihoods(self):
        integers = DifferentiableDiscreteLayer(
            0, jnp.log(jnp.array([[1.0, 2.0, 1.0], [0.0, 1.0, 3.0]]))
        )
        jax_circuit = mixture_of_products(integers, gaussian_layer(1))
        jax_circuit.variables = SortedSet([count, y])
        numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            jax_circuit
        )
        events = np.array([[0.0, 0.3], [1.0, -1.0], [2.0, 2.5]])
        self.assert_same_log_likelihoods(jax_circuit, numpy_circuit, events)

    def test_symbolic_variable_has_the_same_log_likelihoods(self):
        sizes = DifferentiableDiscreteLayer(
            0, jnp.log(jnp.array([[1.0, 2.0, 5.0], [4.0, 1.0, 3.0]]))
        )
        jax_circuit = mixture_of_products(sizes, gaussian_layer(1))
        jax_circuit.variables = SortedSet([size, y])
        numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            jax_circuit
        )
        events = np.array(
            [
                [hash(Size.SMALL), 0.3],
                [hash(Size.MEDIUM), -1.0],
                [hash(Size.LARGE), 2.5],
            ]
        )
        self.assert_same_log_likelihoods(jax_circuit, numpy_circuit, events)

    def test_shared_child_layer_stays_shared(self):
        shared = gaussian_layer(0)
        jax_circuit = mixture_of_products(shared, gaussian_layer(1))
        first_product = jax_circuit.root.child_layers[0]
        second_product = DifferentiableProductLayer(
            [shared, gaussian_layer(1)],
            product_edges([(0, 0, 1), (1, 0, 0)], (2, 1)),
        )
        jax_circuit.root = DifferentiableSparseSumLayer(
            [first_product, second_product],
            [
                sparse_matrix([(0, 0, 0.0), (0, 1, 0.0)], (1, 2)),
                sparse_matrix([(0, 0, 0.0)], (1, 1)),
            ],
        )
        numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            jax_circuit
        )
        first, second = numpy_circuit.root.child_layers
        self.assertIs(first.child_layers[0], second.child_layers[0])

    def test_classification_circuit_is_refused(self):
        classification_circuit = (
            RegionGraph(
                SortedSet([w, x, y, z]), partitions=2, depth=1, repetitions=2, classes=2
            )
            .create_random_region_graph()
            .as_probabilistic_circuit()
        )
        self.assertIsInstance(classification_circuit, ClassificationCircuit)
        with self.assertRaises(CannotConvertError):
            DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
                classification_circuit
            )


if __name__ == "__main__":
    unittest.main()
