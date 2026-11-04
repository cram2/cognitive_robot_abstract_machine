"""
A layered circuit converts into a differentiable circuit with the same distribution, so
that a circuit learned in another way can be refined by gradient descent.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from random_events.interval import Bound
from random_events.set import Set
from random_events.variable import Symbolic, Variable
from scipy.sparse import coo_array
from sortedcontainers import SortedSet

from probabilistic_model.adapters.exceptions import CannotConvertError
from probabilistic_model.adapters.jax_tensorized.converter import (
    TensorizedToJaxConverter,
)
from probabilistic_model.adapters.jax_tensorized.exceptions import (
    StatesAreNotColumnIndicesError,
)
from probabilistic_model.adapters.jax_tensorized.jax_to_tensorized import (
    DifferentiableLayeredCircuitToLayeredCircuitConverter,
)
from probabilistic_model.adapters.jax_tensorized.tensorized_to_jax import (
    LayeredCircuitToDifferentiableLayeredCircuitConverter,
    LayersToDifferentiable,
)
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import (
    DifferentiableDiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import (
    DifferentiableGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableProductLayer,
    DifferentiableSparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.input_layer import (
    DifferentiableDiracDeltaLayer,
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
    TruncatedGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.uniform_layer import (
    UniformLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.row_grouped_sparse_array import (
    RowGroupedSparseArray,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)

from .differentiable_circuits import Size, dense_log_weights


def conversion_over(*variables: Variable) -> LayersToDifferentiable:
    """
    :param variables: The variables the converted layers index.
    :return: A conversion of layered circuit layers into differentiable layers.
    """
    return LayersToDifferentiable(TensorizedToJaxConverter, SortedSet(variables))


def gaussian_layer(variable: int) -> GaussianLayer:
    """
    :param variable: The index of the variable of the layer.
    :return: A Gaussian layer with two nodes over the variable.
    """
    return GaussianLayer(variable, np.array([0.0, 3.0]), np.array([1.0, 0.5]))


@pytest.fixture
def truncated_gaussian_layer() -> TruncatedGaussianLayer:
    """
    A Gaussian truncated to [-1, 1], which a differentiable circuit does not support.
    """
    return TruncatedGaussianLayer(
        0,
        np.array([[-1.0, 1.0]]),
        np.full((1, 2), int(Bound.CLOSED), dtype=np.int64),
        np.array([0.0]),
        np.array([1.0]),
    )


# %% inner layers


@pytest.fixture
def sum_layer() -> SumLayer:
    """
    Two mixtures over two Gaussian layers.
    """
    return SumLayer(
        [gaussian_layer(0), gaussian_layer(0)],
        RowGroupedSparseArray.from_entries(
            SparseEntries(
                np.log(np.array([1.0, 3.0, 2.0, 2.0])),
                np.array([0, 0, 1, 1]),
                np.array([0, 3, 1, 2]),
            ),
            (2, 4),
        ),
    )


def test_one_weight_matrix_per_child_layer(sum_layer, x):
    layer = conversion_over(x).convert(sum_layer)
    assert isinstance(layer, DifferentiableSparseSumLayer)
    assert [log_weights.shape for log_weights in layer.log_weights] == [
        (sum_layer.number_of_nodes, child_layer.number_of_nodes)
        for child_layer in sum_layer.child_layers
    ]


def test_weights_are_normalized(sum_layer, x, single_precision_tolerance):
    layer = conversion_over(x).convert(sum_layer)
    np.testing.assert_allclose(
        np.asarray(layer.normalized_weights.todense()),
        np.exp(dense_log_weights(sum_layer)),
        rtol=single_precision_tolerance,
    )


def test_product_edges_keep_their_child_nodes(x, y):
    product = ProductLayer(
        [gaussian_layer(0), gaussian_layer(1)],
        coo_array(
            (np.array([0, 1, 1, 0]), (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))),
            shape=(2, 2),
        ),
    )
    layer = conversion_over(x, y).convert(product)
    assert isinstance(layer, DifferentiableProductLayer)
    np.testing.assert_array_equal(
        np.asarray(layer.edges.todense()), product.edges.toarray()
    )


# %% input layers


def test_gaussian_has_no_minimum_scale(x):
    numpy_layer = gaussian_layer(0)
    layer = conversion_over(x).convert(numpy_layer)
    assert isinstance(layer, DifferentiableGaussianLayer)
    np.testing.assert_array_equal(np.asarray(layer.minimum_scale), 0.0)
    np.testing.assert_allclose(np.asarray(layer.scale), numpy_layer.scale)
    np.testing.assert_allclose(np.asarray(layer.location), numpy_layer.location)


def test_uniform_keeps_its_intervals(x):
    numpy_layer = UniformLayer(
        0,
        np.array([[-1.0, 1.0], [2.0, 5.0]]),
        np.full((2, 2), int(Bound.OPEN), dtype=np.int64),
    )
    layer = conversion_over(x).convert(numpy_layer)
    assert isinstance(layer, DifferentiableUniformLayer)
    np.testing.assert_allclose(np.asarray(layer.interval), numpy_layer.interval)


def test_dirac_delta_keeps_location_and_density_cap(x):
    numpy_layer = DiracDeltaLayer(0, np.array([0.5]), np.array([2.0]))
    layer = conversion_over(x).convert(numpy_layer)
    assert isinstance(layer, DifferentiableDiracDeltaLayer)
    np.testing.assert_allclose(np.asarray(layer.location), numpy_layer.location)
    np.testing.assert_allclose(np.asarray(layer.density_cap), numpy_layer.density_cap)


def test_symbolic_state_is_stored_in_the_column_of_its_hash(
    size, single_precision_tolerance
):
    numpy_layer = SymbolicLayer(
        0,
        np.arange(len(Size)),
        DenseProbabilityTable(np.log(np.array([[0.5, 0.2, 0.3]]))),
        SymbolicEncoding(size).hashes,
    )
    layer = conversion_over(size).convert(numpy_layer)
    assert isinstance(layer, DifferentiableDiscreteLayer)
    columns = SymbolicEncoding(size).hashes.astype(int)
    np.testing.assert_allclose(
        np.exp(np.asarray(layer.log_probabilities))[:, columns],
        numpy_layer.table.dense_probabilities(),
        rtol=single_precision_tolerance,
    )


def test_integer_state_is_stored_in_the_column_of_its_value(
    count, single_precision_tolerance
):
    numpy_layer = IntegerLayer(
        0, np.array([0, 2]), DenseProbabilityTable(np.log(np.array([[0.4, 0.6]])))
    )
    layer = conversion_over(count).convert(numpy_layer)
    assert isinstance(layer, DifferentiableDiscreteLayer)
    expected = np.zeros((1, numpy_layer.states.max() + 1))
    expected[:, numpy_layer.states] = numpy_layer.table.dense_probabilities()
    np.testing.assert_allclose(
        np.exp(np.asarray(layer.log_probabilities)),
        expected,
        rtol=single_precision_tolerance,
    )


def test_truncated_gaussian_layer_is_refused(truncated_gaussian_layer, x):
    with pytest.raises(CannotConvertError):
        conversion_over(x).convert(truncated_gaussian_layer)


def test_symbolic_states_that_are_not_column_indices_are_refused():
    letters = Symbolic(name="letter", domain=Set.from_iterable(["a", "b"]))
    symbols = SymbolicLayer(
        0,
        np.arange(2),
        DenseProbabilityTable(np.log(np.array([[0.5, 0.5]]))),
        SymbolicEncoding(letters).hashes,
    )
    with pytest.raises(StatesAreNotColumnIndicesError):
        conversion_over(letters).convert(symbols)


def test_negative_integer_states_are_refused(count):
    integers = IntegerLayer(
        0, np.array([-1, 0]), DenseProbabilityTable(np.log(np.array([[0.5, 0.5]])))
    )
    with pytest.raises(StatesAreNotColumnIndicesError):
        conversion_over(count).convert(integers)


# %% circuits


def test_every_supported_input_layer_has_the_same_variables_and_log_likelihoods(
    layered_circuit_over_color_and_x, single_precision_tolerance
):
    jax_circuit = LayeredCircuitToDifferentiableLayeredCircuitConverter.convert(
        layered_circuit_over_color_and_x
    )
    assert jax_circuit.variables == layered_circuit_over_color_and_x.variables
    events = np.array(
        [[0.0, 0.5], [1.0, 0.5], [2.0, -0.3], [1.0, 3.2], [0.0, 4.5], [2.0, 9.0]]
    )
    np.testing.assert_allclose(
        np.asarray(jax_circuit.log_likelihood(jnp.asarray(events))),
        layered_circuit_over_color_and_x.log_likelihood(events),
        rtol=single_precision_tolerance,
        atol=single_precision_tolerance,
    )


def test_round_trip_keeps_the_log_likelihoods(
    layered_circuit_over_color_and_x, single_precision_tolerance
):
    round_trip = DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
        LayeredCircuitToDifferentiableLayeredCircuitConverter.convert(
            layered_circuit_over_color_and_x
        )
    )
    events = np.array([[0.0, 0.5], [2.0, -0.3], [1.0, 3.2]])
    np.testing.assert_allclose(
        round_trip.log_likelihood(events),
        layered_circuit_over_color_and_x.log_likelihood(events),
        rtol=single_precision_tolerance,
    )


def test_differentiable_round_trip_keeps_the_log_likelihoods(
    continuous_region_graph_circuit, single_precision_tolerance
):
    round_trip = LayeredCircuitToDifferentiableLayeredCircuitConverter.convert(
        DifferentiableLayeredCircuitToLayeredCircuitConverter.convert(
            continuous_region_graph_circuit
        )
    )
    events = jnp.asarray(np.random.normal(size=(20, 4)))
    np.testing.assert_allclose(
        np.asarray(round_trip.log_likelihood(events)),
        np.asarray(continuous_region_graph_circuit.log_likelihood(events)),
        rtol=single_precision_tolerance,
        atol=single_precision_tolerance,
    )


def test_unsupported_layer_in_a_circuit_is_refused(truncated_gaussian_layer, x):
    circuit = LayeredProbabilisticCircuit(
        SortedSet([x]), SumLayer.mixture_of([truncated_gaussian_layer], [0.0])
    )
    with pytest.raises(CannotConvertError):
        LayeredCircuitToDifferentiableLayeredCircuitConverter.convert(circuit)
