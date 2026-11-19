"""
Tests of :class:`krrood_spatial_math.spatial_types.Displacement`.
"""

from __future__ import annotations

import json

import numpy as np

from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.spatial_types import (
    SymbolicSpatialValue3Dimensional,
    Displacement,
    Vector3Dimensional,
)

from .dataset.frames import KnownFrames

# %% the parts of a displacement


def test_a_displacement_has_six_entries():
    """
    Three entries for the translation and three for the rotation vector.
    """
    displacement = Displacement()

    assert displacement.shape == (6, 1)


def test_a_displacement_without_parts_neither_moves_nor_turns():
    """
    Left out, both parts default to zero.
    """
    displacement = Displacement()

    np.testing.assert_allclose(displacement.translation.to_np()[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        displacement.rotation_vector.to_np()[:3], [0.0, 0.0, 0.0]
    )


def test_a_displacement_returns_the_parts_it_was_given():
    """
    The translation and rotation vector read back as they were given.
    """
    displacement = Displacement(
        translation=Vector3Dimensional(0.1, -0.2, 0.3),
        rotation_vector=Vector3Dimensional(0.0, 0.7, 0.0),
    )

    np.testing.assert_allclose(displacement.translation.to_np()[:3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(
        displacement.rotation_vector.to_np()[:3], [0.0, 0.7, 0.0]
    )


# %% the frame a displacement is expressed in


def test_a_displacement_is_a_value_given_in_a_cartesian_frame():
    """
    A displacement is expressed in a reference frame.
    """
    assert issubclass(Displacement, SymbolicSpatialValue3Dimensional)


def test_the_parts_of_a_displacement_are_expressed_in_its_frame(table_frame):
    """
    Both parts carry the displacement's reference frame.
    """
    displacement = Displacement(
        translation=Vector3Dimensional(0.1, 0.0, 0.0),
        rotation_vector=Vector3Dimensional(0.0, 0.0, 0.2),
        reference_frame=table_frame,
    )

    assert displacement.translation.reference_frame is table_frame
    assert displacement.rotation_vector.reference_frame is table_frame


def test_a_deserialized_displacement_refers_to_the_readers_frame(table_frame):
    """
    A JSON round trip keeps the values and resolves the frame on the reading side.
    """
    displacement = Displacement(
        translation=Vector3Dimensional(0.1, 0.2, 0.3),
        rotation_vector=Vector3Dimensional(0.4, 0.0, 0.0),
        reference_frame=table_frame,
    )
    known_frames = KnownFrames()
    known_frames.add(key=table_frame.id, tracked_object=table_frame)

    displacement_copy = from_json(
        json.loads(json.dumps(to_json(displacement))), **known_frames.create_kwargs()
    )

    assert displacement_copy.reference_frame is table_frame
    np.testing.assert_allclose(displacement_copy.to_np(), displacement.to_np())
