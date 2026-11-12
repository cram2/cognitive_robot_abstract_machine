"""
Fixtures shared by the tests of the layered circuits.
"""

from __future__ import annotations

from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.row_grouped_sparse_array import (
    RowGroupedSparseArray,
    RowGroupedSparsity,
    SparseEntries,
)

from ..circuit_building import mixture_of_products_of_mixtures, uniform_leaf

# %% circuits


@pytest.fixture
def shared_children_circuit(x, y) -> ProbabilisticCircuit:
    """
    A circuit whose leaves are shared by several sum units, so the layered circuit is a
    directed acyclic graph rather than a tree.
    """
    circuit = ProbabilisticCircuit()
    return mixture_of_products_of_mixtures(
        circuit,
        (uniform_leaf(x, 0, 1, circuit), uniform_leaf(x, 1, 2, circuit)),
        (uniform_leaf(y, 0, 1, circuit), uniform_leaf(y, 1, 2, circuit)),
        [((0.8, 0.2), (0.6, 0.4)), ((0.3, 0.7), (0.1, 0.9))],
    )


# %% sparsity


@pytest.fixture
def uneven_rows() -> RowGroupedSparsity:
    """
    Four rows holding two, one, three and no stored entries, stored out of row order.
    """
    return RowGroupedSparseArray.from_entries(
        SparseEntries(np.zeros(6), [2, 0, 2, 1, 2, 0], [0, 1, 2, 3, 4, 5]), (4, 6)
    ).sparsity


# %% array backends


@pytest.fixture(params=[np, jnp], ids=["numpy", "jax"])
def array_backend(request) -> ModuleType:
    """
    The array namespace of every supported backend, with jax computing in double
    precision.
    """
    with jax.enable_x64(True):
        yield request.param
