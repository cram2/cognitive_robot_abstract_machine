from pycram.datastructures.pose import PoseStamped
from pycram.pose_validator import (
    reachability_validator,
    pose_sequence_reachability_validator,
)


def test_pose_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = PoseStamped.from_list([1.7, 1.4, 1], frame=world.root)

    assert reachability_validator(
        pose, world.get_body_by_name("r_gripper_tool_frame"), robot_view, world
    )


def test_pose_reachable_full_body(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = PoseStamped.from_list([2.7, 1.4, 1], frame=world.root)

    assert reachability_validator(
        pose, world.get_body_by_name("r_gripper_tool_frame"), robot_view, world, True
    )
    assert not reachability_validator(
        pose,
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
    )


def test_pose_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = PoseStamped.from_list([2.3, 2, 1], frame=world.root)

    assert not reachability_validator(
        pose, world.get_body_by_name("r_gripper_tool_frame"), robot_view, world
    )


def test_pose_sequence_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = PoseStamped.from_list([1.6, 1.4, 1], frame=world.root)
    pose2 = PoseStamped.from_list([1.7, 1.4, 1], frame=world.root)
    pose3 = PoseStamped.from_list([1.7, 1.4, 1.1], frame=world.root)

    assert pose_sequence_reachability_validator(
        [pose1, pose2, pose3],
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
    )


def test_pose_sequence_reachable_full_body(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = PoseStamped.from_list([2.6, 1.4, 1], frame=world.root)
    pose2 = PoseStamped.from_list([2.7, 1.4, 1], frame=world.root)
    pose3 = PoseStamped.from_list([2.7, 1.4, 1.1], frame=world.root)

    assert pose_sequence_reachability_validator(
        [pose1, pose2, pose3],
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
        True,
    )

    assert not pose_sequence_reachability_validator(
        [pose1, pose2, pose3],
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
    )


def test_pose_sequence_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = PoseStamped.from_list([2.6, 1.4, 1], frame=world.root)
    pose2 = PoseStamped.from_list([2.7, 1.4, 1], frame=world.root)
    pose3 = PoseStamped.from_list([2.7, 1.4, 1.1], frame=world.root)

    assert not pose_sequence_reachability_validator(
        [pose1, pose2, pose3],
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
    )


def test_pose_sequence_one_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = PoseStamped.from_list([1.6, 1.4, 1], frame=world.root)
    pose2 = PoseStamped.from_list([1.7, 1.4, 1], frame=world.root)
    pose3 = PoseStamped.from_list([2.7, 2.4, 1.5], frame=world.root)

    assert not reachability_validator(
        pose3,
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
    )

    assert not pose_sequence_reachability_validator(
        [pose1, pose2, pose3],
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
    )
