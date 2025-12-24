import math
import pytest
from copy import deepcopy

import numpy as np

from pycram.datastructures.pose import (
    PoseStamped,
    TransformStamped,
    PyCramQuaternion,
    PyCramVector3,
    AxisIdentifier,
)


def test_pose_creation(immutable_model_world):
    world, robot_view, context = immutable_model_world
    p = PoseStamped.from_list([1, 2, 3], [0, 0, 0, 1], world.root)

    assert p.position.to_list() == [1, 2, 3]
    assert p.orientation.to_list() == [0, 0, 0, 1]
    assert p.frame_id == world.root
    assert p.pose.to_list() == [[1, 2, 3], [0, 0, 0, 1]]


def test_pose_to_transform(immutable_model_world):
    world, robot_view, context = immutable_model_world
    p = PoseStamped.from_list([3, 2, 1], [0, 0, 1, 0], world.root)

    transform = p.to_transform_stamped(world.get_body_by_name("r_gripper_tool_frame"))

    assert transform == TransformStamped.from_list(
        [3, 2, 1], [0, 0, 1, 0], world.root, world.get_body_by_name("r_gripper_tool_frame")
    )


def test_pose_edit(immutable_model_world):
    world, robot_view, context = immutable_model_world
    p = PoseStamped.from_list([3, 4, 5], [0, 1, 0, 0], world.root)

    p.position = PyCramVector3(1, 1, 1)
    assert p.position.to_list() == [1, 1, 1]
    p.position.x = 2
    assert p.position.to_list() == [2, 1, 1]
    p.position = PyCramVector3(3, 3, 3)
    assert p.position.to_list() == [3, 3, 3]

    p.orientation = PyCramQuaternion(0, 0, 0, 1)
    assert p.orientation.to_list() == [0, 0, 0, 1]
    p.orientation.x = 1
    assert p.orientation.to_list() == [1, 0, 0, 1]
    p.orientation = PyCramQuaternion(0, 0, 1, 0)
    assert p.orientation.to_list() == [0, 0, 1, 0]


def test_pose_copy(immutable_model_world):
    world, robot_view, context = immutable_model_world
    p1 = PoseStamped.from_list([1, 2, 3], [0, 0, 0, 1], world.root)
    p2 = deepcopy(p1)

    assert p1 == p2
    assert p1 is not p2


def test_transform_creation(immutable_model_world):
    world, robot_view, context = immutable_model_world
    test_body = world.get_body_by_name("milk.stl")
    t = TransformStamped.from_list([1, 2, 3], [0, 0, 0, 1], world.root, test_body)

    assert t.translation.to_list() == [1, 2, 3]
    assert t.rotation.to_list() == [0, 0, 0, 1]
    assert t.frame_id == world.root
    assert t.child_frame_id == test_body


def test_transform_edit(immutable_model_world):
    world, robot_view, context = immutable_model_world
    test_body = world.get_body_by_name("milk.stl")
    t = TransformStamped.from_list([3, 2, 1], [0, 1, 0, 0], world.root, test_body)

    t.translation = PyCramVector3(2, 2, 2)
    assert t.translation.to_list() == [2, 2, 2]
    t.translation.x = 3
    assert t.translation.to_list() == [3, 2, 2]
    t.translation = PyCramVector3(1, 1, 1)
    assert t.translation.to_list() == [1, 1, 1]

    t.rotation = PyCramQuaternion(1, 0, 0, 0)
    assert t.rotation.to_list() == [1, 0, 0, 0]
    t.rotation.y = 1
    assert t.rotation.to_list() == [1, 1, 0, 0]
    t.rotation = PyCramQuaternion(0, 0, 0, 1)
    assert t.rotation.to_list() == [0, 0, 0, 1]


def test_transform_copy(immutable_model_world):
    world, robot_view, context = immutable_model_world
    test_body = world.get_body_by_name("milk.stl")
    t = TransformStamped.from_list([1, 1, 1], [0, 0, 0, 1], world.root, test_body)

    t_copy = deepcopy(t)
    assert t == t_copy
    assert t is not t_copy


def test_transform_multiplication(immutable_model_world):
    world, robot_view, context = immutable_model_world
    test_body = world.get_body_by_name("milk.stl")
    t = TransformStamped.from_list([1, 2, 3], [0, 0, 0, 1], world.root, test_body)
    t2 = TransformStamped.from_list(
        [3, 2, 1], [0, 0, 0, 1], test_body, world.get_body_by_name("r_gripper_tool_frame")
    )

    mul_t = t * t2

    assert mul_t.translation.to_list() == [4, 4, 4]


def test_is_facing_2d_axis(immutable_model_world):
    world, robot_view, context = immutable_model_world
    a = PoseStamped.from_list([0, 0, 0], [0, 0, 0, 1], world.root)  # facing +x
    b = PoseStamped.from_list([1, 0, 0], [0, 0, 0, 1], world.root)

    facing, angle = a.is_facing_2d_axis(b, axis=AxisIdentifier.X)
    assert facing
    assert angle == pytest.approx(0, abs=1e-6)

    # now krrood_test Y alignment (should be 90 deg difference)
    facing_y, angle_y = a.is_facing_2d_axis(b, axis=AxisIdentifier.Y)
    assert not facing_y
    assert abs(angle_y) == pytest.approx(math.pi / 2, abs=1e-6)


def test_is_facing_x_or_y(immutable_model_world):
    world, robot_view, context = immutable_model_world
    a = PoseStamped.from_list([0, 0, 0], [0, 0, 0, 1], world.root)
    b = PoseStamped.from_list([1, 0, 0], [0, 0, 0, 1], world.root)

    assert a.is_facing_x_or_y(b)

    # reverse direction
    b.position.x = -1
    assert not a.is_facing_x_or_y(b)


def test_transform_stamped_multiplication(immutable_model_world):
    world, robot_view, context = immutable_model_world
    t1 = TransformStamped.from_list([1, 2, 3], [0, 0, 0, 1], world.root, robot_view.root)
    t2 = TransformStamped.from_list(
        [4, 5, 6], [0, 0, 0, 1], robot_view.root, world.get_body_by_name("r_gripper_tool_frame")
    )

    result = t1 * t2

    assert result.translation.to_list() == [5, 7, 9]
    assert result.frame_id == world.root
    assert result.child_frame_id == world.get_body_by_name("r_gripper_tool_frame")


def test_transform_multiplication_inverse(immutable_model_world):
    world, robot_view, context = immutable_model_world
    t1 = TransformStamped.from_list([1, 2, 3], [0, 0, 0, 1], world.root, robot_view.root)
    t2 = TransformStamped.from_list(
        [4, 5, 6], [0, 0, 0, 1], robot_view.root, world.get_body_by_name("r_gripper_tool_frame")
    )

    result = t1 * t2
    inverse_result = ~result

    assert inverse_result.translation.to_list() == [-5, -7, -9]
    assert inverse_result.frame_id == world.get_body_by_name("r_gripper_tool_frame")
    assert inverse_result.child_frame_id == world.root


def test_transform_multiplication_rotation(immutable_model_world):
    world, robot_view, context = immutable_model_world
    t1 = TransformStamped.from_list([1, 1, 1], [0, 0, 1, 1], world.root, robot_view.root)
    t2 = TransformStamped.from_list(
        [1, 0, 0], [0, 0, 0, 1], robot_view.root, world.get_body_by_name("r_gripper_tool_frame")
    )

    result = t1 * t2

    assert [1, 2, 1] == result.translation.to_list()
    assert result.frame_id == world.root
    np.testing.assert_almost_equal(result.rotation.to_list(), [0, 0, 0.707, 0.707], decimal=3)


def test_transform_multiplication_translation_inverse(immutable_model_world):
    world, robot_view, context = immutable_model_world
    t1 = TransformStamped.from_list([1, 1, 1], [0, 0, 0, 1], world.root, robot_view.root)
    t2 = TransformStamped.from_list(
        [1, 0, 0], [0, 0, 0, 1], world.root, world.get_body_by_name("r_gripper_tool_frame")
    )

    result = ~t1 * t2

    assert result.frame_id == robot_view.root
    np.testing.assert_almost_equal(result.rotation.to_list(), [0, 0, 0, 1], decimal=3)
    assert [0, -1, -1] == result.translation.to_list()


def test_transform_multiplication_with_inverse(immutable_model_world):
    world, robot_view, context = immutable_model_world
    t1 = TransformStamped.from_list([1, 1, 1], [0, 0, 0, 1], world.root, robot_view.root)
    t2 = TransformStamped.from_list(
        [2, 2, 1], [0, 0, -1, 1], world.root, world.get_body_by_name("r_gripper_tool_frame")
    )
    result = ~t1 * t2

    assert result.frame_id == robot_view.root
    assert result.child_frame_id == world.get_body_by_name("r_gripper_tool_frame")
    np.testing.assert_almost_equal(result.translation.to_list(), [1, 1, 0], decimal=3)
    np.testing.assert_almost_equal(result.rotation.to_list(), [0, 0, -0.707, 0.707], decimal=3)


def test_rotation_multiplication(immutable_model_world):
    world, robot_view, context = immutable_model_world
    t1 = TransformStamped.from_list([0, 0, 0], [0, 0, 1, 1], world.root, robot_view.root)
    t2 = TransformStamped.from_list(
        [0, 0, 0], [0, 0, 1, 0], robot_view.root, world.get_body_by_name("r_gripper_tool_frame")
    )

    result = t1 * t2
    assert result.frame_id == world.root
    np.testing.assert_almost_equal(result.rotation.to_list(), [0, 0, -0.707, 0.707], decimal=3)
    assert result.translation.to_list() == [0, 0, 0]
