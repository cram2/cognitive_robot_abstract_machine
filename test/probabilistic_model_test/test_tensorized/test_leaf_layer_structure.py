"""
Tests of the structure of the leaf layers of a layered circuit.
"""

from __future__ import annotations

from krrood.adapters.json_serializer import from_json, to_json

from ..circuit_building import uniform_layer_of

# %% child layers


def test_a_leaf_layer_has_no_child_layers():
    """
    A leaf layer points to no other layer.
    """
    assert uniform_layer_of(0, [(0.0, 1.0)]).child_layers == []


def test_the_json_of_a_leaf_layer_leaves_out_its_child_layers():
    """
    The JSON of a leaf layer holds no child layers, and reading it back gives a leaf
    layer without child layers.
    """
    layer = uniform_layer_of(0, [(0.0, 1.0)])

    serialized = to_json(layer)

    assert "child_layers" not in serialized
    assert from_json(serialized).child_layers == []
