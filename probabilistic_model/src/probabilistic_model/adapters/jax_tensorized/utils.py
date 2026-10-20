"""
Functions that the converters between the ``jax`` package and the ``tensorized``
package need. Some of them serve one direction of the conversion only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from random_events.variable import Symbolic
from typing_extensions import Tuple

from probabilistic_model.adapters.jax_tensorized.exceptions import (
    StatesAreNotColumnIndicesError,
)
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import (
    DifferentiableDiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    StateIndices,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    DiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)

# %% arrays


def to_numpy(array: jax.Array) -> np.ndarray:
    """
    :param array: An array of the ``jax`` package, usually in single precision.
    :return: The array in double precision, as the ``tensorized`` package computes.
    """
    return np.asarray(array, dtype=float)


def sparse_matrix(
    values: np.ndarray, rows: np.ndarray, columns: np.ndarray, shape: Tuple[int, int]
) -> BCOO:
    """
    :param values: The value of every stored entry.
    :param rows: The row of every stored entry.
    :param columns: The column of every stored entry.
    :param shape: The shape of the dense matrix.
    :return: The sparse matrix of the ``jax`` package that stores the entries.
    """
    indices = np.stack([rows, columns], axis=1).astype(np.int32).reshape(-1, 2)
    return BCOO(
        (jnp.asarray(values), jnp.asarray(indices)), shape=tuple(shape)
    ).sort_indices()


# %% discrete layers


def columns_of_domain_elements(variable: Symbolic) -> StateIndices:
    """
    :param variable: A symbolic variable.
    :return: The column of the probability table of a discrete layer of the ``jax``
        package that holds every domain element, at the position of the element.
    :raises StatesAreNotColumnIndicesError: If the domain elements of the variable do
        not hash to the numbers from zero to the size of the domain.
    """
    hashes = SymbolicEncoding(variable).hashes
    if not np.array_equal(np.sort(hashes), np.arange(len(hashes))):
        raise StatesAreNotColumnIndicesError(variable=variable, states=hashes)
    return hashes.astype(np.int64)


def discrete_layer_of(
    data: DiscreteLayer, columns: StateIndices, number_of_columns: int
) -> DifferentiableDiscreteLayer:
    """
    :param data: A discrete layer.
    :param columns: The column of the probability table that holds every state of the
        layer.
    :param number_of_columns: The number of columns of the probability table.
    :return: The differentiable discrete layer, impossible in every column
        that holds no state.
    """
    probabilities = np.zeros((data.number_of_nodes, number_of_columns))
    probabilities[:, columns] = data.table.dense_probabilities()
    with np.errstate(divide="ignore"):
        log_probabilities = np.log(probabilities)
    return DifferentiableDiscreteLayer(data.variable, jnp.asarray(log_probabilities))
