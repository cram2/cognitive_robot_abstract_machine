import numpy as np
import pytest

from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Twist,
    Vector3,
)

# %% reading a twist off a transformation


def test_a_twist_moves_as_far_as_the_transformation():
    start_T_end = HomogeneousTransformationMatrix.from_xyz_rpy(0.1, -0.2, 0.3)

    twist = Twist.from_transformation(start_T_end)

    np.testing.assert_allclose(twist.translation.to_np()[:3], [0.1, -0.2, 0.3])


def test_a_twist_turns_as_the_transformation_turns():
    axis, angle = np.array([0.0, 1.0, 0.0]), 0.7
    start_T_end = HomogeneousTransformationMatrix.from_xyz_axis_angle(
        axis=Vector3(*axis), angle=angle
    )

    twist = Twist.from_transformation(start_T_end)

    np.testing.assert_allclose(twist.rotation.axis.to_np()[:3], axis, atol=1e-9)
    assert float(twist.rotation.angle) == pytest.approx(angle)


def test_no_rotation_is_the_angle_zero_about_the_z_axis():
    twist = Twist.from_transformation(HomogeneousTransformationMatrix())

    assert float(twist.rotation.angle) == 0.0
    np.testing.assert_allclose(twist.rotation.axis.to_np()[:3], [0.0, 0.0, 1.0])


def test_a_rotation_is_stated_by_an_angle_of_at_most_half_a_turn():
    three_quarter_turn = HomogeneousTransformationMatrix.from_xyz_axis_angle(
        axis=Vector3.X(), angle=1.5 * np.pi
    )

    twist = Twist.from_transformation(three_quarter_turn)

    assert float(twist.rotation.angle) == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(
        twist.rotation.axis.to_np()[:3], [-1.0, 0.0, 0.0], atol=1e-9
    )


def test_a_twist_is_built_from_constants():
    """
    A twist is data to record and learn from, so it holds numbers, not expressions.
    """
    twist = Twist.from_transformation(
        HomogeneousTransformationMatrix.from_xyz_rpy(0.1, 0.2, 0.3, roll=0.4)
    )

    assert twist.translation.casadi_sx.is_constant()
    assert twist.rotation.casadi_sx.is_constant()
