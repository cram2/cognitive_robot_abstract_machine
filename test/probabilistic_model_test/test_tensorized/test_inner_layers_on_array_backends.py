"""
Tests that the numeric queries of sum and product layers, and of the layered circuits
built from them, run on every supported array backend and agree with numpy there.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from random_events.interval import closed, open_closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous
from scipy.special import logsumexp
from sortedcontainers import SortedSet
from typing_extensions import Any

from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    InnerLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.moment_query import (
    MomentQuery,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    EdgeMatrix,
    RowReduction,
    SparseEdgeMatrix,
    SparseEntries,
)

from .array_backend_comparison import assert_result_of_backend_equals

# %% a circuit built with the arrays of a backend


@dataclass
class MixtureOfProductsCase:
    """
    A circuit over two variables whose root mixes two mixtures of products and one of
    the products directly, so the products are shared by two sum layers.

    The sum nodes have different numbers of children and unnormalized weights.
    """

    x: Continuous
    """
    The first variable of the circuit.
    """

    y: Continuous
    """
    The second variable of the circuit.
    """

    @property
    def variables(self) -> SortedSet:
        """
        :return: The variables of the circuit.
        """
        return SortedSet([self.x, self.y])

    def circuit(
        self,
        namespace: ModuleType,
        edge_matrix_type: type[EdgeMatrix] = SparseEdgeMatrix,
    ) -> LayeredProbabilisticCircuit:
        """
        :param namespace: The array namespace to create the parameters with.
        :param edge_matrix_type: The type of matrix every inner layer stores its edges
            in.
        :return: The circuit with every parameter an array of that namespace.
        """
        leaf_x = GaussianLayer(
            0, namespace.asarray([-1.0, 2.0]), namespace.asarray([0.5, 1.5])
        )
        leaf_y = GaussianLayer(
            1, namespace.asarray([0.0, 1.0]), namespace.asarray([1.0, 2.0])
        )
        products = ProductLayer.node_wise_product_of([leaf_x, leaf_y])
        products.store_edges_as(edge_matrix_type)
        # node 0 mixes both products, node 1 only the second one
        mixtures = SumLayer.from_entries(
            [products],
            SparseEntries(
                namespace.asarray(np.log([0.3, 0.7, 1.0])),
                np.array([0, 0, 1]),
                np.array([0, 1, 1]),
            ),
            2,
            edge_matrix_type,
        )
        root = SumLayer.from_entries(
            [mixtures, products],
            SparseEntries(
                namespace.asarray(np.log([1.0, 0.4, 0.6])),
                np.array([0, 0, 0]),
                np.array([0, 1, 2]),
            ),
            1,
            edge_matrix_type,
        )
        return LayeredProbabilisticCircuit(self.variables, root)

    def events(self, namespace: ModuleType) -> Any:
        """
        :param namespace: An array namespace.
        :return: Values of both variables, an array of that namespace.
        """
        return namespace.asarray(
            np.array([[-1.0, 0.0], [0.5, 0.5], [2.0, 1.0], [3.0, -2.0]])
        )

    @property
    def simple_event(self) -> SimpleEvent:
        """
        :return: A box over both variables to compute the probability of.
        """
        return SimpleEvent.from_data(
            {self.x: closed(-0.5, 2.5), self.y: open_closed(-np.inf, 1.0)}
        )

    @property
    def moment_query(self) -> MomentQuery:
        """
        :return: The mean of the first variable and the second moment of the second one
            around a half.
        """
        return MomentQuery.from_maps(
            {self.x: 1, self.y: 2}, {self.x: 0.0, self.y: 0.5}, self.variables
        )


@pytest.fixture
def mixture_of_products_case() -> MixtureOfProductsCase:
    """
    The circuit whose queries are compared across backends.
    """
    return MixtureOfProductsCase(Continuous("x"), Continuous("y"))


# %% numeric queries agree with numpy on every backend


def test_log_likelihood_on_every_backend(
    mixture_of_products_case, edge_matrix_type, array_backend
):
    """
    The log-likelihood of a circuit with parameters of a backend, of events of that
    backend, is an array of that backend and agrees with numpy.
    """
    case = mixture_of_products_case
    assert_result_of_backend_equals(
        case.circuit(array_backend, edge_matrix_type).log_likelihood(
            case.events(array_backend)
        ),
        case.circuit(np).log_likelihood(case.events(np)),
        array_backend,
    )


def test_cumulative_distribution_function_on_every_backend(
    mixture_of_products_case, edge_matrix_type, array_backend
):
    """
    The cumulative distribution of a circuit with parameters of a backend is an array of
    that backend and agrees with numpy.
    """
    case = mixture_of_products_case
    assert_result_of_backend_equals(
        case.circuit(array_backend, edge_matrix_type).cumulative_distribution_function(
            case.events(array_backend)
        ),
        case.circuit(np).cumulative_distribution_function(case.events(np)),
        array_backend,
    )


def test_probability_of_simple_event_of_nodes_on_every_backend(
    mixture_of_products_case, edge_matrix_type, array_backend
):
    """
    The probability of a simple event under the root of a circuit with parameters of a
    backend is an array of that backend and agrees with numpy.
    """
    case = mixture_of_products_case
    assert_result_of_backend_equals(
        case.circuit(
            array_backend, edge_matrix_type
        ).root.probability_of_simple_event_of_nodes(case.simple_event, case.variables),
        case.circuit(np).root.probability_of_simple_event_of_nodes(
            case.simple_event, case.variables
        ),
        array_backend,
    )


def test_moment_of_nodes_on_every_backend(
    mixture_of_products_case, edge_matrix_type, array_backend
):
    """
    The moments of the root of a circuit with parameters of a backend are an array of
    that backend and agree with numpy.
    """
    case = mixture_of_products_case
    assert_result_of_backend_equals(
        case.circuit(array_backend, edge_matrix_type).root.moment_of_nodes(
            case.moment_query, case.variables
        ),
        case.circuit(np).root.moment_of_nodes(case.moment_query, case.variables),
        array_backend,
    )


# %% numeric queries compile with jax


def test_log_likelihood_compiles_with_jax(mixture_of_products_case, edge_matrix_type):
    """
    The log-likelihood of a circuit with jax parameters runs under :func:`jax.jit` and
    agrees with numpy.
    """
    case = mixture_of_products_case
    with jax.enable_x64(True):
        circuit = case.circuit(jnp, edge_matrix_type)
        result = jax.jit(circuit.log_likelihood)(case.events(jnp))
        assert_result_of_backend_equals(
            result, case.circuit(np).log_likelihood(case.events(np)), jnp
        )


def test_cumulative_distribution_function_compiles_with_jax(
    mixture_of_products_case, edge_matrix_type
):
    """
    The cumulative distribution of a circuit with jax parameters runs under
    :func:`jax.jit` and agrees with numpy.
    """
    case = mixture_of_products_case
    with jax.enable_x64(True):
        circuit = case.circuit(jnp, edge_matrix_type)
        result = jax.jit(circuit.cumulative_distribution_function)(case.events(jnp))
        assert_result_of_backend_equals(
            result,
            case.circuit(np).cumulative_distribution_function(case.events(np)),
            jnp,
        )


def test_a_circuit_that_answered_a_query_can_be_deep_copied(
    mixture_of_products_case, edge_matrix_type, array_backend
):
    """
    A circuit whose layers have answered a query, and so know the namespace of their
    parameters, still deep copies into a circuit with the same answers.
    """
    case = mixture_of_products_case
    circuit = case.circuit(array_backend, edge_matrix_type)
    expected = np.asarray(circuit.log_likelihood(case.events(array_backend)))
    assert_result_of_backend_equals(
        copy.deepcopy(circuit).log_likelihood(case.events(array_backend)),
        expected,
        array_backend,
    )


# %% every edge matrix type answers alike


def test_storing_the_edges_in_another_matrix_keeps_the_log_likelihood(
    mixture_of_products_case, edge_matrix_type
):
    """
    Storing the edges of every inner layer in a matrix of another type changes neither
    the edges nor the log-likelihood.
    """
    case = mixture_of_products_case
    circuit = case.circuit(np).store_edges_as(edge_matrix_type)
    assert all(
        type(layer.edges) is edge_matrix_type
        for layer in circuit.layers
        if isinstance(layer, InnerLayer)
    )
    np.testing.assert_allclose(
        circuit.log_likelihood(case.events(np)),
        case.circuit(np).log_likelihood(case.events(np)),
        rtol=1e-12,
    )


def test_a_truncated_circuit_keeps_the_type_of_its_edge_matrices(
    mixture_of_products_case, edge_matrix_type
):
    """
    Truncating a circuit rebuilds its inner layers with edge matrices of the type they
    had, and the truncation agrees with that of the sparse circuit.
    """
    case = mixture_of_products_case
    event = case.simple_event.as_composite_set()
    truncation = case.circuit(np, edge_matrix_type).truncated(event)
    expected = case.circuit(np).truncated(event)
    assert all(
        type(layer.edges) is edge_matrix_type
        for layer in truncation.distribution.layers
        if isinstance(layer, InnerLayer)
    )
    assert truncation.probability == pytest.approx(expected.probability, rel=1e-12)
    np.testing.assert_allclose(
        truncation.distribution.log_likelihood(case.events(np)),
        expected.distribution.log_likelihood(case.events(np)),
        rtol=1e-12,
    )


# %% row reductions on every backend


def reduced_by_definition(
    values: np.ndarray, edges: EdgeMatrix, reduction: RowReduction
) -> np.ndarray:
    """
    :param values: One value per column, the columns in the last axis.
    :param edges: The edges that connect the rows to the columns.
    :param reduction: How the values along the edges of a row are combined.
    :return: The values along the edges of every row combined one row at a time.
    """
    combine = {
        RowReduction.SUM: np.sum,
        RowReduction.PRODUCT: np.prod,
        RowReduction.LOG_SUM_EXP: logsumexp,
    }[reduction]
    return np.stack(
        [
            combine(values[..., edges.columns[edges.rows == row]], axis=-1)
            for row in range(edges.shape[0])
        ],
        axis=-1,
    )


@pytest.mark.parametrize("reduction", list(RowReduction))
def test_reduce_by_row_on_every_backend(
    uneven_rows, edge_matrix_type, reduction, array_backend
):
    """
    Reducing the values of the columns along the edges of every row gives an array of
    the backend of the values, and the identity of the reduction for a row without
    edges.
    """
    edges = edge_matrix_type.from_edge_positions(
        uneven_rows.rows, uneven_rows.columns, uneven_rows.shape
    )
    values = np.array(
        [[0.5, -1.0, 2.0, 0.25, -0.5, 1.5], [1.0, 2.0, -3.0, 0.5, 0.75, -0.25]]
    )
    assert_result_of_backend_equals(
        edges.reduce_by_row(
            edges.values_of_columns(array_backend.asarray(values)), reduction
        ),
        reduced_by_definition(values, edges, reduction),
        array_backend,
    )
