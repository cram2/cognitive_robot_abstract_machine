import pytest

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.exceptions import TipLinkDoesNotMatchAnyArm
from coraplex.locations.pose_validator import (
    IsReachableBy,
    AreReachableBy,
)
from coraplex.motion_executor import simulated_robot
from coraplex.robot_plans import MoveToolCenterPointMotion
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3


def test_pose_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)

    assert IsReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose=pose,
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


def test_pose_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = Pose(Point3.from_iterable([2.3, 2, 1]), reference_frame=world.root)

    assert not IsReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose=pose,
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


def test_pose_sequence_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([1.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([1.7, 1.4, 1.1]), reference_frame=world.root)

    assert AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose_sequence=[pose1, pose2, pose3],
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


def test_pose_sequence_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([2.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([2.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([2.7, 1.4, 1.1]), reference_frame=world.root)

    assert not AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose_sequence=[pose1, pose2, pose3],
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


class _MoveTcpAlternativeForPr2(MoveToolCenterPointMotion, AlternativeMotion[PR2]):
    """Minimal alternative used to exercise the unmatched-tip-link guard."""

    execution_type = ExecutionType.SIMULATED


def test_unmatched_tip_link_raises(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)

    validator = AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=[_MoveTcpAlternativeForPr2],
        ),
        pose_sequence=[pose],
        tip_link=robot_view.root,
    )

    with simulated_robot, pytest.raises(TipLinkDoesNotMatchAnyArm):
        validator.create_msc()


def test_pose_sequence_one_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([1.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([2.7, 2.4, 1.5]), reference_frame=world.root)

    assert not IsReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose=pose3,
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )

    assert not AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose_sequence=[pose1, pose2, pose3],
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )
