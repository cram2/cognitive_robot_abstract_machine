from __future__ import annotations

import json

import numpy as np

from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.cartesian_frame_3d import ValueInCartesianFrame3D
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Twist,
    Vector3,
)

from .dataset.frames import KnownFrames

# %% reading a twist off a transformation


def test_a_twist_moves_as_far_as_the_transformation():
    start_T_end = HomogeneousTransformationMatrix.from_xyz_rpy(0.1, -0.2, 0.3)

    twist = Twist.from_transformation(start_T_end)

    np.testing.assert_allclose(twist.translation.to_np()[:3], [0.1, -0.2, 0.3])


def test_a_twist_has_six_entries():
    twist = Twist.from_transformation(HomogeneousTransformationMatrix())

    assert twist.shape == (6, 1)


def test_a_twist_turns_by_the_axis_scaled_by_the_angle():
    axis, angle = np.array([0.0, 1.0, 0.0]), 0.7
    start_T_end = HomogeneousTransformationMatrix.from_xyz_axis_angle(
        axis=Vector3(*axis), angle=angle
    )

    twist = Twist.from_transformation(start_T_end)

    np.testing.assert_allclose(
        twist.rotation_vector.to_np()[:3], axis * angle, atol=1e-9
    )


def test_no_rotation_is_the_zero_rotation_vector():
    twist = Twist.from_transformation(HomogeneousTransformationMatrix())

    np.testing.assert_allclose(twist.rotation_vector.to_np()[:3], [0.0, 0.0, 0.0])


def test_a_rotation_is_stated_by_an_angle_of_at_most_half_a_turn():
    three_quarter_turn = HomogeneousTransformationMatrix.from_xyz_axis_angle(
        axis=Vector3.X(), angle=1.5 * np.pi
    )

    twist = Twist.from_transformation(three_quarter_turn)

    np.testing.assert_allclose(
        twist.rotation_vector.to_np()[:3], [-np.pi / 2, 0.0, 0.0], atol=1e-9
    )


def test_a_twist_is_built_from_constants():
    """
    A twist is data to record and learn from, so it holds numbers, not expressions.
    """
    twist = Twist.from_transformation(
        HomogeneousTransformationMatrix.from_xyz_rpy(0.1, 0.2, 0.3, roll=0.4)
    )

    assert twist.translation.casadi_sx.is_constant()
    assert twist.rotation_vector.casadi_sx.is_constant()


# %% the frame a twist is expressed in


def test_a_twist_is_a_value_given_in_a_cartesian_frame():
    assert issubclass(Twist, ValueInCartesianFrame3D)


def test_a_twist_is_expressed_in_the_frame_of_the_transformation(table_frame):
    start_T_end = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=0.1, yaw=0.2, reference_frame=table_frame
    )

    twist = Twist.from_transformation(start_T_end)

    assert twist.reference_frame is table_frame
    assert twist.translation.reference_frame is table_frame
    assert twist.rotation_vector.reference_frame is table_frame


def test_a_deserialized_twist_refers_to_the_readers_frame(table_frame):
    twist = Twist.from_transformation(
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.1, 0.2, 0.3, roll=0.4, reference_frame=table_frame
        )
    )
    known_frames = KnownFrames()
    known_frames.add(table_frame.id, table_frame)

    twist_copy = from_json(
        json.loads(json.dumps(to_json(twist))), **known_frames.create_kwargs()
    )

    assert twist_copy.reference_frame is table_frame
    np.testing.assert_allclose(twist_copy.to_np(), twist.to_np())
