"""
Tests that numpy arrays of a structure are copied to the device of the torch tensors
they are used with once, handed to numpy and jax as they are, and left out of the deep
copies and the JSON of an input layer.
"""

from __future__ import annotations

# %% imports
import copy

import jax.numpy as jnp
import numpy as np
import pytest
from krrood.adapters.json_serializer import from_json, to_json

from probabilistic_model.probabilistic_circuit.tensorized.structure_array_copies import (
    StructureArrayCopies,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)

# %% placed arrays


@pytest.fixture
def indices() -> np.ndarray:
    """
    Positions of two columns, out of order.
    """
    return np.array([2, 0])


def test_numpy_and_jax_arrays_are_indexed_with_the_numpy_indices(indices):
    """
    Numpy and jax arrays are used with the numpy indices themselves.
    """
    copies = StructureArrayCopies()
    assert copies.placed(array=indices, values=np.zeros(3)) is indices
    assert copies.placed(array=indices, values=jnp.zeros(3)) is indices


def test_torch_tensors_are_indexed_with_one_copy_on_their_device(indices):
    """
    A torch tensor is used with a copy of the indices on its device, the same copy every
    time.
    """
    torch = pytest.importorskip("torch")
    copies = StructureArrayCopies()
    values = torch.zeros(3)

    first = copies.placed(array=indices, values=values)

    assert first is copies.placed(array=indices, values=values)
    assert first.device == values.device
    assert first.tolist() == indices.tolist()


def test_copies_of_an_input_layer_share_its_structure_array_copies():
    """
    A copy of an input layer with other parameters keeps the structure array copies of
    the layer it was copied from.
    """
    torch = pytest.importorskip("torch")
    layer = GaussianLayer(
        np.array([0, 0, 1]),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
    )

    copy = layer.copy_with_parameters(parameters=[torch.ones(3), torch.zeros(3)])

    assert copy.structure_array_copies is layer.structure_array_copies


# %% deep copies and JSON


@pytest.fixture
def layer_queried_with_torch() -> GaussianLayer:
    """
    A Gaussian layer over two variables whose node variables were copied to the device
    of torch events.
    """
    torch = pytest.importorskip("torch")
    layer = GaussianLayer(
        np.array([0, 0, 1]),
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
    )
    layer.column_of(events=torch.zeros((4, 2), dtype=torch.float64))
    return layer


def test_a_deep_copy_of_an_input_layer_starts_without_structure_array_copies(
    layer_queried_with_torch,
):
    """
    A deep copy of an input layer has structure array copies of its own, with no copies
    made yet.
    """
    deep_copy = copy.deepcopy(layer_queried_with_torch)

    assert (
        deep_copy.structure_array_copies
        is not layer_queried_with_torch.structure_array_copies
    )
    assert deep_copy.structure_array_copies.copies == {}


def test_a_layer_copied_back_from_torch_round_trips_through_json(
    layer_queried_with_torch,
):
    """
    A numpy copy of an input layer that was queried with torch serializes, although it
    shares the torch copies of its structure arrays, and comes back with none of them.
    """
    host = layer_queried_with_torch.copy_with_parameters(
        parameters=[np.arange(3.0), np.zeros(3)]
    )

    restored = from_json(to_json(host))

    np.testing.assert_array_equal(restored.node_variables, host.node_variables)
    np.testing.assert_array_equal(restored.location, host.location)
    assert restored.structure_array_copies.copies == {}
