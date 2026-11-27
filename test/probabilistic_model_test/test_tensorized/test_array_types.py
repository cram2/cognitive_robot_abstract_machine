"""
Tests of the names for the arrays the layers of a layered circuit pass around.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from krrood.adapters.json_serializer import from_json, to_json

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


def test_the_array_class_jax_annotates_with_is_a_backend_array():
    """
    :class:`jax.Array`, which annotations name but no array is created as, is a backend
    array, though only the class of its arrays exchanges them through DLPack.
    """
    assert issubclass(jax.Array, BackendArray)


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


# %% JSON of arrays of any backend


def test_a_jax_array_is_read_back_from_json_as_a_numpy_array_of_its_values():
    """
    The JSON of an array of another backend holds its values, which are read back as a
    numpy array.
    """
    array = jnp.arange(3.0)

    restored = from_json(to_json(array))

    assert type(restored) is np.ndarray
    np.testing.assert_array_equal(restored, np.asarray(array))
