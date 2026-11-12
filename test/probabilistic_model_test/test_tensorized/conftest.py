"""
Fixtures shared by the tests of the layered circuits.
"""

from __future__ import annotations

from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from array_api_compat import array_namespace

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    DenseEdgeMatrix,
    EdgeMatrix,
    SparseEdgeMatrix,
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
def uneven_rows() -> SparseEdgeMatrix:
    """
    Four rows holding two, one, three and no stored entries, stored out of row order.
    """
    return SparseEdgeMatrix(
        np.array([2, 0, 2, 1, 2, 0]), np.array([0, 1, 2, 3, 4, 5]), (4, 6)
    )


@pytest.fixture(params=[SparseEdgeMatrix, DenseEdgeMatrix], ids=["sparse", "dense"])
def edge_matrix_type(request) -> type[EdgeMatrix]:
    """
    Every type of edge matrix.
    """
    return request.param


# %% array backends


@pytest.fixture(params=["numpy", "jax", "torch"])
def array_backend(request) -> ModuleType:
    """
    The array namespace of every supported backend, with jax computing in double
    precision and torch defaulting to double precision.

    Torch is an optional dependency, so its tests are skipped without it.
    """
    match request.param:
        case "numpy":
            yield np
        case "jax":
            with jax.enable_x64(True):
                yield jnp
        case "torch":
            torch = pytest.importorskip("torch")
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            yield array_namespace(torch.zeros(0))
            torch.set_default_dtype(default_dtype)
