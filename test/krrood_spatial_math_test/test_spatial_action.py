"""
How transformations and rotations act on the values they are applied to, and the frames
they require of them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from krrood_spatial_math.exceptions import ReferenceFrameMismatchError
from krrood_spatial_math.spatial_types import (
    AxisAngle,
    Point3Dimensional,
    HomogeneousTransformationMatrix3Dimensional,
    RotationMatrix,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    Point2Dimensional,
    HomogeneousTransformationMatrix2Dimensional,
)


# %% a rotation acting on a point


def test_a_rotation_turns_a_point_about_the_origin(table_frame, camera_frame):
    """
    A rotation applied to a point turns it about the origin of the point's frame and
    gives it in the rotation's frame.
    """
    quarter_turn_about_z = RotationMatrix.from_rpy(
        yaw=math.pi / 2, reference_frame=table_frame
    )

    turned = quarter_turn_about_z @ Point3Dimensional(
        1, 0, 0, reference_frame=camera_frame
    )

    assert isinstance(turned, Point3Dimensional)
    assert turned.reference_frame is table_frame
    np.testing.assert_allclose(turned.to_np()[:3], [0, 1, 0], atol=1e-12)


# %% a transformation acting on a value given in its child frame


def test_a_transformation_turns_and_shifts_a_point(table_frame, camera_frame):
    """
    A transformation applied to a point turns it and shifts it by the translation.
    """
    table_T_camera = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1, yaw=math.pi / 2, reference_frame=table_frame, child_frame=camera_frame
    )

    moved = table_T_camera @ Point3Dimensional(1, 0, 0, reference_frame=camera_frame)

    assert moved.reference_frame is table_frame
    np.testing.assert_allclose(moved.to_np()[:3], [1, 1, 0], atol=1e-12)


def test_a_transformation_turns_an_axis_angle(table_frame, camera_frame):
    """
    A transformation applied to an axis angle composes its rotation with the axis
    angle's and drops its translation.
    """
    table_T_camera = HomogeneousTransformationMatrix3Dimensional.from_xyz_rpy(
        x=1, yaw=math.pi / 2, reference_frame=table_frame, child_frame=camera_frame
    )
    turn_about_x = AxisAngle(Vector3Dimensional.X(), 0.3, reference_frame=camera_frame)

    turned = table_T_camera @ turn_about_x

    assert isinstance(turned, AxisAngle)
    assert turned.reference_frame is table_frame
    np.testing.assert_allclose(
        turned.rotation_matrix.to_np(),
        (table_T_camera.rotation_matrix @ turn_about_x.rotation_matrix).to_np(),
        atol=1e-12,
    )


# %% frames a transformation requires


def test_a_transformation_refuses_a_value_outside_its_child_frame(
    table_frame, camera_frame, gripper_frame
):
    """
    A transformation only applies to a value given in its child frame.
    """
    table_T_camera = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=table_frame, child_frame=camera_frame
    )

    with pytest.raises(ReferenceFrameMismatchError):
        table_T_camera @ Point3Dimensional(reference_frame=gripper_frame)


def test_a_transformation_composes_only_with_a_change_into_its_child_frame(
    table_frame, camera_frame, gripper_frame
):
    """
    A transformation composes only with a change whose reference frame is its child
    frame.
    """
    table_T_camera = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=table_frame, child_frame=camera_frame
    )
    table_T_gripper = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=table_frame, child_frame=gripper_frame
    )

    with pytest.raises(ReferenceFrameMismatchError):
        table_T_camera @ table_T_gripper


def test_a_transformation_accepts_a_value_without_a_frame(table_frame, camera_frame):
    """
    A value given in no frame is taken to be given in the transformation's child frame.
    """
    table_T_camera = HomogeneousTransformationMatrix3Dimensional(
        reference_frame=table_frame, child_frame=camera_frame
    )

    assert (table_T_camera @ Vector3Dimensional.X()).reference_frame is table_frame


def test_a_planar_transformation_refuses_a_point_on_another_plane(
    table_frame, camera_frame, gripper_frame
):
    """
    A planar transformation only applies to a point on the plane of its child frame.
    """
    table_T_camera = HomogeneousTransformationMatrix2Dimensional(
        reference_frame=table_frame, child_frame=camera_frame
    )

    with pytest.raises(ReferenceFrameMismatchError):
        table_T_camera @ Point2Dimensional(reference_frame=gripper_frame)


# %% arithmetic between values of two frames


@pytest.mark.parametrize(
    "left_type, right_type",
    [
        (Point3Dimensional, Point3Dimensional),
        (Point3Dimensional, Vector3Dimensional),
        (Vector3Dimensional, Vector3Dimensional),
    ],
)
def test_subtracting_a_value_of_another_frame_is_refused(
    table_frame, camera_frame, left_type, right_type
):
    """
    Coordinates of two frames are not subtracted from each other.
    """
    with pytest.raises(ReferenceFrameMismatchError):
        left_type(reference_frame=table_frame) - right_type(
            reference_frame=camera_frame
        )


@pytest.mark.parametrize("left_type", [Point3Dimensional, Vector3Dimensional])
def test_adding_a_vector_of_another_frame_is_refused(
    table_frame, camera_frame, left_type
):
    """
    Coordinates of two frames are not added to each other.
    """
    with pytest.raises(ReferenceFrameMismatchError):
        left_type(reference_frame=table_frame) + Vector3Dimensional(
            reference_frame=camera_frame
        )


def test_a_value_without_a_frame_combines_with_one_in_a_frame(table_frame):
    """
    A value given in no frame, such as a constant, combines with a value in any frame.
    """
    point = Point3Dimensional(1, 2, 3, reference_frame=table_frame)

    shifted = point + Vector3Dimensional.X()

    assert shifted.reference_frame is table_frame
    np.testing.assert_allclose(shifted.to_np()[:3], [2, 2, 3])
