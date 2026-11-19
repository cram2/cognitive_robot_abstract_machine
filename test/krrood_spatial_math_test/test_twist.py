"""
Tests of :class:`krrood_spatial_math.spatial_types.Twist`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.cartesian_frame_3d import ValueInCartesianFrame3D
from krrood_spatial_math.spatial_types import RigidMotionRate, Twist, Vector3

from .dataset.frames import KnownFrames
from .reference_implementations import matrix_exponential, rigid_motion_rate_matrix

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


# %% the displacement after moving at a twist


def test_a_twist_is_a_rate_of_rigid_motion():
    """
    A twist shares the exponential of every rate of rigid motion.
    """
    assert issubclass(Twist, RigidMotionRate)


@pytest.mark.parametrize("duration", [0.0, 1e-6, 0.5, 2.0])
def test_the_displacement_of_a_twist_agrees_with_the_matrix_exponential(
    duration: float,
):
    """
    Moving at a constant twist for a duration displaces the body by the exponential of
    the twist's matrix times that duration.
    """
    twist = Twist(
        angular_velocity=Vector3(0.2, -0.4, 0.9),
        linear_velocity=Vector3(0.5, 0.1, -0.3),
    )
    expected = matrix_exponential(
        rigid_motion_rate_matrix(
            rotation=twist.angular_velocity.to_np()[:3],
            translation=twist.linear_velocity.to_np()[:3],
        )
        * duration
    )

    displacement = twist.to_displacement(duration=duration)

    np.testing.assert_allclose(
        displacement.translation.to_np()[:3], expected[:3, 3], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        displacement.rotation_vector.to_np()[:3],
        twist.angular_velocity.to_np()[:3] * duration,
    )
