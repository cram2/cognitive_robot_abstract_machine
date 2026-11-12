"""
Layers and circuits that the tests of the conversions between differentiable circuits
and layered circuits share.
"""

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from random_events.interval import Bound
from random_events.variable import Continuous, Symbolic
from scipy.sparse import coo_array
from sortedcontainers import SortedSet

from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.jax.gaussian_layer import (
    DifferentiableGaussianLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableInputLayer,
    DifferentiableProductLayer,
    DifferentiableSparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
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
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
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


class Color(IntEnum):
    """
    The categories of a symbolic variable.
    """

    RED = 0
    GREEN = 1
    BLUE = 2


class Size(IntEnum):
    """
    Members whose order of declaration differs from the order of their hashes.
    """

    LARGE = 2
    SMALL = 0
    MEDIUM = 1


# %% sparse matrices


def sparse_matrix(
    entries: list[tuple[int, int, float]], shape: tuple[int, int]
) -> BCOO:
    """
    :param entries: The ``(row, column, value)`` of every stored entry.
    :param shape: The shape of the matrix.
    :return: The sparse matrix.
    """
    rows, columns, values = zip(*entries)
    return BCOO((jnp.array(values), jnp.array(list(zip(rows, columns)))), shape=shape)


def dense_log_weights(layer: SumLayer) -> np.ndarray:
    """
    :param layer: A sum layer.
    :return: The normalized logarithmic weights of the layer as a dense matrix, ``-inf``
        where the layer has no edge.
    """
    result = np.full((layer.number_of_nodes, layer.column_offsets[-1]), -np.inf)
    result[layer.log_weights.rows, layer.log_weights.columns] = (
        layer.normalized_edge_log_weights
    )
    return result


# %% differentiable circuits


def region_graph_circuit(variables: SortedSet) -> DifferentiableLayeredCircuit:
    """
    :param variables: At least four variables, so that every region of the region
        graph can be split.
    :return: A randomly initialized differentiable circuit over the variables, as a
        region graph builds it for training.
    """
    region_graph = RegionGraph(variables, partitions=2, depth=1, repetitions=2)
    return region_graph.create_random_region_graph().as_probabilistic_circuit(
        input_units=3, sum_units=3
    )


def mixture_of_products(
    first_layer: DifferentiableInputLayer,
    second_layer: DifferentiableInputLayer,
    variables: SortedSet,
) -> DifferentiableLayeredCircuit:
    """
    :param first_layer: The layer over the first variable.
    :param second_layer: The layer over the second variable.
    :param variables: The two variables.
    :return: A circuit whose root mixes two products of the nodes of the layers with
        unnormalized weights, as training leaves them.
    """
    product = DifferentiableProductLayer(
        [first_layer, second_layer],
        sparse_matrix([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)], (2, 2)),
    )
    root = DifferentiableSparseSumLayer(
        [product], [sparse_matrix([(0, 0, 0.3), (0, 1, -1.2)], (1, 2))]
    )
    return DifferentiableLayeredCircuit(variables, root)


def gaussian_layer(variable: int) -> DifferentiableGaussianLayer:
    """
    :param variable: The index of the variable of the layer.
    :return: A differentiable Gaussian layer with two nodes over the variable.
    """
    return DifferentiableGaussianLayer(
        variable,
        location=jnp.array([-1.0, 2.0]),
        log_scale=jnp.log(jnp.array([0.5, 1.5])),
        minimum_scale=jnp.array([0.1, 0.2]),
    )


# %% layered circuits


def layered_circuit_over(color: Symbolic, x: Continuous) -> LayeredProbabilisticCircuit:
    """
    :param color: A symbolic variable over :class:`Color`.
    :param x: A continuous variable.
    :return: A layered circuit over the variables that uses every input layer a
        differentiable circuit supports.
    """
    colors = SymbolicLayer(
        0,
        np.arange(len(Color)),
        DenseProbabilityTable(np.log(np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]]))),
        SymbolicEncoding(color).hashes,
    )
    gaussians = GaussianLayer(1, np.array([0.0, 3.0]), np.array([1.0, 0.5]))
    uniforms = UniformLayer(
        1,
        np.array([[-1.0, 1.0], [2.0, 5.0]]),
        np.full((2, 2), int(Bound.OPEN), dtype=np.int64),
    )
    dirac_deltas = DiracDeltaLayer(1, np.array([0.5]), np.array([2.0]))
    x_mixture = SumLayer(
        [gaussians, uniforms, dirac_deltas],
        RowGroupedSparseArray.from_entries(
            SparseEntries(
                np.log(np.array([0.4, 0.4, 0.2, 0.7, 0.3])),
                np.array([0, 0, 0, 1, 1]),
                np.array([0, 2, 4, 1, 3]),
            ),
            (2, 5),
        ),
    )
    products = ProductLayer(
        [colors, x_mixture],
        coo_array(
            (np.array([0, 1, 0, 1]), (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]))),
            shape=(2, 2),
        ),
    )
    root = SumLayer(
        [products],
        RowGroupedSparseArray.from_entries(
            SparseEntries(
                np.log(np.array([0.25, 0.75])), np.array([0, 0]), np.array([0, 1])
            ),
            (1, 2),
        ),
    )
    return LayeredProbabilisticCircuit(SortedSet([color, x]), root)
