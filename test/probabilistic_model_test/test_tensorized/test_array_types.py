"""
Tests of the names for the arrays the layers of a layered circuit pass around.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# %% arrays of any backend


def test_numpy_and_jax_arrays_are_backend_arrays():
    """
    Arrays of the numpy and jax backends are backend arrays without deriving from it.
    """
    assert isinstance(np.zeros(2), BackendArray)
    assert isinstance(jnp.zeros(2), BackendArray)


def test_torch_tensors_are_backend_arrays():
    """
    Tensors of the torch backend are backend arrays without deriving from it.
    """
    torch = pytest.importorskip("torch")

    assert isinstance(torch.zeros(2), BackendArray)


def test_a_list_is_no_backend_array():
    """
    A list of numbers implements no array backend, so it is no backend array.
    """
    assert not isinstance([0.0, 1.0], BackendArray)
