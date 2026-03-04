from pycram.pose_validator import (
    reachability_validator,
    pose_sequence_reachability_validator,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3


def test_pose_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)

    assert reachability_validator(
        pose, world.get_body_by_name("r_gripper_tool_frame"), robot_view, world
    )


def test_pose_reachable_full_body(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = Pose(Point3.from_iterable([2.7, 1.4, 1]), reference_frame=world.root)

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

    pose = Pose(Point3.from_iterable([2.3, 2, 1]), reference_frame=world.root)

    assert not reachability_validator(
        pose, world.get_body_by_name("r_gripper_tool_frame"), robot_view, world
    )


def test_pose_sequence_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([1.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([1.7, 1.4, 1.1]), reference_frame=world.root)

    assert pose_sequence_reachability_validator(
        [pose1, pose2, pose3],
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
    )


def test_pose_sequence_reachable_full_body(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([2.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([2.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([2.7, 1.4, 1.1]), reference_frame=world.root)

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

    pose1 = Pose(Point3.from_iterable([2.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([2.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([2.7, 1.4, 1.1]), reference_frame=world.root)

    assert not pose_sequence_reachability_validator(
        [pose1, pose2, pose3],
        world.get_body_by_name("r_gripper_tool_frame"),
        robot_view,
        world,
    )


def test_pose_sequence_one_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([1.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([2.7, 2.4, 1.5]), reference_frame=world.root)

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
