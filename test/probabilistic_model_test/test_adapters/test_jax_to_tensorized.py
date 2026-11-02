"""
A trained differentiable circuit converts into a layered circuit that expresses the same
distribution, so that every query of the layered circuit answers for it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from random_events.interval import Bound
from random_events.variable import Variable
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
    dense_log_weights,
    gaussian_layer,
    mixture_of_products,
    region_graph_circuit,
    sparse_matrix,
)


def conversion_over(*variables: Variable) -> LayersToTensorized:
    """
    :param variables: The variables the converted layers index.
    :return: A conversion of differentiable layers into layered circuit layers.
    """
    return LayersToTensorized(JaxToTensorizedConverter, SortedSet(variables))


def assert_same_log_likelihoods(
    jax_circuit: DifferentiableLayeredCircuit,
    numpy_circuit: LayeredProbabilisticCircuit,
    events: np.ndarray,
    tolerance: float,
) -> None:
    """
    Assert that both circuits give the events the same log-likelihoods.
    """
    np.testing.assert_allclose(
        numpy_circuit.log_likelihood(events),
        np.asarray(jax_circuit.log_likelihood(jnp.asarray(events))),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.fixture
def discrete_jax_layer() -> DifferentiableDiscreteLayer:
    """
    Two unnormalized distributions over three states.
    """
    return DifferentiableDiscreteLayer(
        0, jnp.log(jnp.array([[1.0, 2.0, 5.0], [4.0, 1.0, 3.0]]))
    )


# %% inner layers


def assert_normalized_weights(layer: SumLayer, unnormalized: np.ndarray) -> None:
    """
    Assert that the weights of a layer are the normalized unnormalized weights.
    """
    expected = unnormalized - logsumexp(unnormalized, axis=1, keepdims=True)
    np.testing.assert_allclose(dense_log_weights(layer), expected, atol=1e-6)


def test_sparse_weights_are_normalized_over_all_child_layers(x):
    sum_layer = DifferentiableSparseSumLayer(
        [gaussian_layer(0), gaussian_layer(0)],
        [
            sparse_matrix([(0, 0, 0.0), (0, 1, 1.0), (1, 1, 2.0)], (2, 2)),
            sparse_matrix([(0, 0, 0.5), (1, 0, -1.0), (1, 1, 0.0)], (2, 2)),
        ],
    )
    layer = conversion_over(x).convert(sum_layer)
    assert isinstance(layer, SumLayer)
    assert_normalized_weights(
        layer, np.array([[0.0, 1.0, 0.5, -np.inf], [-np.inf, 2.0, -1.0, 0.0]])
    )


def test_dense_weights_are_normalized_over_all_child_layers(x):
    first_weights = np.array([[0.0, 1.0], [2.0, 0.5]])
    second_weights = np.array([[-1.0, 0.0], [0.0, 0.0]])
    sum_layer = DifferentiableDenseSumLayer(
        [gaussian_layer(0), gaussian_layer(0)],
        [jnp.asarray(first_weights), jnp.asarray(second_weights)],
    )
    layer = conversion_over(x).convert(sum_layer)
    assert isinstance(layer, SumLayer)
    assert_normalized_weights(
        layer, np.concatenate([first_weights, second_weights], axis=1)
    )


def test_child_layers_keep_their_order(x):
    first, second = gaussian_layer(0), gaussian_layer(0)
    sum_layer = DifferentiableSparseSumLayer(
        [first, second],
        [sparse_matrix([(0, 0, 0.0)], (1, 2)), sparse_matrix([(0, 1, 0.0)], (1, 2))],
    )
    conversion = conversion_over(x)
    layer = conversion.convert(sum_layer)
    assert layer.child_layers == [conversion.convert(first), conversion.convert(second)]


def test_product_edges_keep_their_child_nodes(x, y):
    edges = sparse_matrix([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], (2, 2))
    product = DifferentiableProductLayer([gaussian_layer(0), gaussian_layer(1)], edges)
    layer = conversion_over(x, y).convert(product)
    assert isinstance(layer, ProductLayer)
    np.testing.assert_array_equal(layer.edges.toarray(), np.asarray(edges.todense()))


def test_a_layer_is_converted_once(x):
    conversion = conversion_over(x)
    shared = gaussian_layer(0)
    assert conversion.convert(shared) is conversion.convert(shared)


# %% input layers


def test_gaussian_scale_includes_the_minimum_scale(x):
    jax_layer = gaussian_layer(0)
    layer = conversion_over(x).convert(jax_layer)
    assert isinstance(layer, GaussianLayer)
    np.testing.assert_allclose(layer.location, np.asarray(jax_layer.location))
    np.testing.assert_allclose(layer.scale, np.asarray(jax_layer.scale))


def test_uniform_keeps_its_intervals_as_open_intervals(x):
    jax_layer = DifferentiableUniformLayer(0, jnp.array([[-1.0, 1.0], [0.0, 4.0]]))
    layer = conversion_over(x).convert(jax_layer)
    assert isinstance(layer, UniformLayer)
    np.testing.assert_allclose(layer.interval, np.asarray(jax_layer.interval))
    assert (layer.bounds == int(Bound.OPEN)).all()


def test_dirac_delta_keeps_location_and_density_cap(x):
    jax_layer = DifferentiableDiracDeltaLayer(
        0, jnp.array([0.5, 1.5]), jnp.array([2.0, 3.0])
    )
    layer = conversion_over(x).convert(jax_layer)
    assert isinstance(layer, DiracDeltaLayer)
    np.testing.assert_allclose(layer.location, np.asarray(jax_layer.location))
    np.testing.assert_allclose(layer.density_cap, np.asarray(jax_layer.density_cap))


def test_discrete_layer_over_an_integer_variable_becomes_an_integer_layer(
    discrete_jax_layer, count, single_precision_tolerance
):
    layer = conversion_over(count).convert(discrete_jax_layer)
    assert isinstance(layer, IntegerLayer)
    np.testing.assert_array_equal(layer.states, np.arange(3))
    np.testing.assert_allclose(
        layer.table.dense_probabilities(),
        np.exp(np.asarray(discrete_jax_layer.normalized_log_probabilities)),
        rtol=single_precision_tolerance,
    )


def test_symbolic_column_is_the_hash_of_the_domain_element(
    discrete_jax_layer, size, single_precision_tolerance
):
    layer = conversion_over(size).convert(discrete_jax_layer)
    assert isinstance(layer, SymbolicLayer)
    columns = SymbolicEncoding(size).hashes.astype(int)
    np.testing.assert_allclose(
        layer.table.dense_probabilities(),
        np.exp(np.asarray(discrete_jax_layer.normalized_log_probabilities))[:, columns],
        rtol=single_precision_tolerance,
    )


def test_discrete_layer_over_a_continuous_variable_is_refused(x):
    jax_layer = DifferentiableDiscreteLayer(0, jnp.log(jnp.array([[0.5, 0.5]])))
    with pytest.raises(CannotConvertError):
        conversion_over(x).convert(jax_layer)


# %% circuits


def test_region_graph_circuit_has_the_same_variables_and_log_likelihoods(
    continuous_region_graph_circuit, single_precision_tolerance
):
    numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
        continuous_region_graph_circuit
    )
    assert numpy_circuit.variables == continuous_region_graph_circuit.variables
    assert_same_log_likelihoods(
        continuous_region_graph_circuit,
        numpy_circuit,
        np.random.normal(size=(50, 4)),
        single_precision_tolerance,
    )


def test_region_graph_circuit_with_symbolic_variable_has_the_same_log_likelihoods(
    w, x, y, color, single_precision_tolerance
):
    variables = SortedSet([w, x, y, color])
    jax_circuit = region_graph_circuit(variables)
    numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
        jax_circuit
    )
    events = np.random.normal(size=(50, 4))
    events[:, variables.index(color)] = np.random.randint(0, len(Color), 50)
    assert_same_log_likelihoods(
        jax_circuit, numpy_circuit, events, single_precision_tolerance
    )


def test_sum_weights_are_normalized(x, y):
    jax_circuit = mixture_of_products(
        gaussian_layer(0), gaussian_layer(1), SortedSet([x, y])
    )
    numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
        jax_circuit
    )
    np.testing.assert_allclose(
        numpy_circuit.root.log_normalization_constants, 0.0, atol=1e-12
    )


def test_uniform_and_dirac_delta_layers_have_the_same_log_likelihoods(
    x, y, single_precision_tolerance
):
    uniforms = DifferentiableUniformLayer(0, jnp.array([[-1.0, 1.0], [0.0, 4.0]]))
    dirac_deltas = DifferentiableDiracDeltaLayer(
        1, jnp.array([0.5, 1.5]), jnp.array([2.0, 3.0])
    )
    jax_circuit = mixture_of_products(uniforms, dirac_deltas, SortedSet([x, y]))
    numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
        jax_circuit
    )
    events = np.array([[0.5, 0.5], [0.5, 1.5], [3.0, 1.5], [-0.5, 0.5]])
    assert_same_log_likelihoods(
        jax_circuit, numpy_circuit, events, single_precision_tolerance
    )


def test_integer_variable_has_the_same_log_likelihoods(
    count, y, single_precision_tolerance
):
    integers = DifferentiableDiscreteLayer(
        0, jnp.log(jnp.array([[1.0, 2.0, 1.0], [0.0, 1.0, 3.0]]))
    )
    jax_circuit = mixture_of_products(
        integers, gaussian_layer(1), SortedSet([count, y])
    )
    numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
        jax_circuit
    )
    events = np.array([[0.0, 0.3], [1.0, -1.0], [2.0, 2.5]])
    assert_same_log_likelihoods(
        jax_circuit, numpy_circuit, events, single_precision_tolerance
    )


def test_symbolic_variable_has_the_same_log_likelihoods(
    discrete_jax_layer, size, y, single_precision_tolerance
):
    jax_circuit = mixture_of_products(
        discrete_jax_layer, gaussian_layer(1), SortedSet([size, y])
    )
    numpy_circuit = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
        jax_circuit
    )
    events = np.array(
        [[hash(Size.SMALL), 0.3], [hash(Size.MEDIUM), -1.0], [hash(Size.LARGE), 2.5]]
    )
    assert_same_log_likelihoods(
        jax_circuit, numpy_circuit, events, single_precision_tolerance
    )


def test_shared_child_layer_stays_shared(x, y):
    shared = gaussian_layer(0)
    jax_circuit = mixture_of_products(shared, gaussian_layer(1), SortedSet([x, y]))
    first_product = jax_circuit.root.child_layers[0]
    second_product = DifferentiableProductLayer(
        [shared, gaussian_layer(1)], sparse_matrix([(0, 0, 1), (1, 0, 0)], (2, 1))
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
    assert first.child_layers[0] is second.child_layers[0]


def test_classification_circuit_is_refused(classification_circuit):
    assert isinstance(classification_circuit, ClassificationCircuit)
    with pytest.raises(CannotConvertError):
        DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            classification_circuit
        )
