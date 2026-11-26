"""
Fixtures shared by the tests of the layered circuits.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from enum import StrEnum
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from array_api_compat import array_namespace
from random_events.variable import Continuous

from probabilistic_model.probabilistic_circuit.tensorized.matrix_shape import (
    MatrixShape2D,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
    DenseEdgeMatrix,
    EdgeMatrix,
    SparseEdgeMatrix,
)

from ..circuit_building import mixture_of_products_of_mixtures, uniform_leaf
from .test_inner_layers_on_array_backends import MixtureOfProductsCase

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


@pytest.fixture
def mixture_of_products_case() -> MixtureOfProductsCase:
    """
    The circuit whose queries are compared across backends.
    """
    return MixtureOfProductsCase(Continuous("x"), Continuous("y"))


# %% sparsity


@pytest.fixture
def uneven_rows() -> SparseEdgeMatrix:
    """
    Four rows holding two, one, three and no stored entries, stored out of row order.
    """
    return SparseEdgeMatrix(
        np.array([2, 0, 2, 1, 2, 0]),
        np.array([0, 1, 2, 3, 4, 5]),
        MatrixShape2D(number_of_rows=4, number_of_columns=6),
    )


@pytest.fixture(
    params=[SparseEdgeMatrix, DenseEdgeMatrix, BlockEdgeMatrix],
    ids=["sparse", "dense", "block"],
)
def edge_matrix_type(request) -> type[EdgeMatrix]:
    """
    Every type of edge matrix.
    """
    return request.param


# %% array backends


class ArrayBackend(StrEnum):
    """
    The array libraries the numeric queries are tested on.
    """

    NUMPY = "numpy"
    """
    NumPy, the default backend.
    """

    JAX = "jax"
    """
    JAX, in double precision.
    """

    TORCH = "torch"
    """
    PyTorch, an optional dependency, in double precision.
    """


@contextlib.contextmanager
def namespace_in_double_precision(backend: ArrayBackend) -> Iterator[ModuleType]:
    """
    :param backend: The array library.
    :return: Its array namespace while jax computes in double precision and torch
        defaults to it; torch tests are skipped without torch.
    """
    match backend:
        case ArrayBackend.NUMPY:
            yield np
        case ArrayBackend.JAX:
            with jax.enable_x64(True):
                yield jnp
        case ArrayBackend.TORCH:
            torch = pytest.importorskip("torch")
            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)
            yield array_namespace(torch.zeros(0))
            torch.set_default_dtype(default_dtype)


@pytest.fixture(params=list(ArrayBackend))
def array_backend(request) -> ModuleType:
    """
    The array namespace of every supported backend, in double precision.
    """
    with namespace_in_double_precision(request.param) as namespace:
        yield namespace


@pytest.fixture(params=[ArrayBackend.JAX, ArrayBackend.TORCH])
def differentiable_backend(request) -> ModuleType:
    """
    The array namespace of every backend that differentiates automatically, in double
    precision.
    """
    with namespace_in_double_precision(request.param) as namespace:
        yield namespace
