"""
Tests of :class:`krrood_spatial_math.spatial_types.Twist`.
"""

from __future__ import annotations

import json

import numpy as np

from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.cartesian_frame_3d import ValueInCartesianFrame3D
from krrood_spatial_math.spatial_types import Twist, Vector3

from .dataset.frames import KnownFrames

# %% the parts of a twist


def test_a_twist_has_six_entries():
    """
    Three entries for the angular and three for the linear velocity.
    """
    twist = Twist()

    assert twist.shape == (6, 1)


def test_a_twist_without_parts_stands_still():
    """
    Left out, both velocities default to zero.
    """
    twist = Twist()

    np.testing.assert_allclose(twist.angular_velocity.to_np()[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(twist.linear_velocity.to_np()[:3], [0.0, 0.0, 0.0])


def test_a_twist_returns_the_velocities_it_was_given():
    """
    The angular and linear velocity read back as they were given.
    """
    twist = Twist(
        angular_velocity=Vector3(0.0, 0.0, 1.0), linear_velocity=Vector3(0.0, -1.0, 0.0)
    )

    np.testing.assert_allclose(twist.angular_velocity.to_np()[:3], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(twist.linear_velocity.to_np()[:3], [0.0, -1.0, 0.0])


# %% the frame a twist is expressed in


def test_a_twist_is_a_value_given_in_a_cartesian_frame():
    """
    A twist is expressed in a reference frame.
    """
    assert issubclass(Twist, ValueInCartesianFrame3D)


def test_the_velocities_of_a_twist_are_expressed_in_its_frame(table_frame):
    """
    Both velocities carry the twist's reference frame.
    """
    twist = Twist(
        angular_velocity=Vector3(0.0, 0.0, 0.2),
        linear_velocity=Vector3(0.1, 0.0, 0.0),
        reference_frame=table_frame,
    )

    assert twist.angular_velocity.reference_frame is table_frame
    assert twist.linear_velocity.reference_frame is table_frame


def test_a_deserialized_twist_refers_to_the_readers_frame(table_frame):
    """
    A JSON round trip keeps the values and resolves the frame on the reading side.
    """
    twist = Twist(
        angular_velocity=Vector3(0.4, 0.0, 0.0),
        linear_velocity=Vector3(0.1, 0.2, 0.3),
        reference_frame=table_frame,
    )
    known_frames = KnownFrames()
    known_frames.add(key=table_frame.id, tracked_object=table_frame)

    twist_copy = from_json(
        json.loads(json.dumps(to_json(twist))), **known_frames.create_kwargs()
    )

    assert twist_copy.reference_frame is table_frame
    np.testing.assert_allclose(twist_copy.to_np(), twist.to_np())
