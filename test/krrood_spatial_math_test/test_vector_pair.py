"""
Tests of :class:`krrood_spatial_math.spatial_types.VectorPair`, through each of its
subclasses.
"""

from __future__ import annotations

import pytest

from krrood.adapters.json_serializer import to_json
from krrood_spatial_math.spatial_types import (
    Displacement,
    ScrewAxis,
    Twist,
    Vector3,
    VectorPair,
)

VECTOR_PAIR_TYPES = [Displacement, Twist, ScrewAxis]

# %% the vectors of a pair


@pytest.mark.parametrize("vector_pair_type", VECTOR_PAIR_TYPES)
def test_a_vector_pair_stores_its_vectors_under_their_names(
    vector_pair_type: type[VectorPair],
):
    """
    Each part is given to the constructor by its name and read back in the json under
    that name.
    """
    rotational_name = vector_pair_type._rotational_part_name
    translational_name = vector_pair_type._translational_part_name
    vector_pair = vector_pair_type(
        **{
            rotational_name: Vector3(1.0, 2.0, 3.0),
            translational_name: Vector3(4.0, 5.0, 6.0),
        }
    )

    json_data = to_json(vector_pair)

    assert json_data[rotational_name] == [1.0, 2.0, 3.0]
    assert json_data[translational_name] == [4.0, 5.0, 6.0]


@pytest.mark.parametrize("vector_pair_type", VECTOR_PAIR_TYPES)
def test_a_vector_pair_takes_its_vectors_only_by_name(
    vector_pair_type: type[VectorPair],
):
    """
    Passing the vectors positionally is refused, so they cannot be swapped unnoticed.
    """
    with pytest.raises(TypeError):
        vector_pair_type(Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0))
