import numpy as np
import pytest

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from coraplex.datastructures.enums import ExecutionType
from coraplex.datastructures.grasp import GraspDescription
from coraplex.exceptions import TipLinkDoesNotMatchAnyArm
from coraplex.execution_environment import simulated_robot
from coraplex.locations.pose_validator import (
    IsReachableBy,
    AreReachableBy,
    IsObjectReachableBy,
)
from coraplex.robot_plans import MoveToolCenterPointMotion
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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


def _right_front_grasp(view):
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        view.right_arm.end_effector,
    )


def test_is_object_reachable_by_copies_current_world_lazily(
    immutable_model_world, monkeypatch
):
    """
    The world copy and pose sequence must be produced when the predicate is
    *evaluated*, not when it is constructed, so the check reflects the current
    world state. We capture what the predicate hands to ``AreReachableBy``.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")

    captured = {}

    def fake_call(self, *args, **kwargs):
        captured["world"] = self.world
        captured["pose_sequence"] = self.pose_sequence
        captured["tip_link"] = self.tip_link
        return True

    monkeypatch.setattr(AreReachableBy, "__call__", fake_call)

    predicate = IsObjectReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        object_designator=milk,
        grasp_description=_right_front_grasp(view),
    )

    # Move the object *after* the predicate has been constructed.
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0
    )

    assert predicate()

    # The validator runs on a throwaway copy, never the live world.
    assert captured["world"] is not world
    # The copy reflects the *current* (moved) object pose, not the parse-time one.
    copied_milk = captured["world"].get_body_by_name("milk.stl")
    assert np.allclose(
        copied_milk.global_pose.to_position().to_np()[:3],
        milk.global_pose.to_position().to_np()[:3],
    )
    # A full grasp sequence (pre-pose, grasp, lift) is generated.
    assert len(captured["pose_sequence"]) == 3
    # The tip link is resolved in the copy, not the live world.
    assert captured["tip_link"]._world is captured["world"]


def test_is_object_reachable_by_uses_target_pose_sequence(
    immutable_model_world, monkeypatch
):
    """With a target pose set, the reach pose sequence is checked."""
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    target = Pose(Point3.from_iterable([2, 1.5, 0.7]), reference_frame=world.root)

    captured = {}
    monkeypatch.setattr(
        AreReachableBy,
        "__call__",
        lambda self, *a, **k: captured.setdefault("seq", self.pose_sequence) or True,
    )

    assert IsObjectReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        object_designator=milk,
        grasp_description=_right_front_grasp(view),
        target_pose=target,
    )()

    assert len(captured["seq"]) == 3


def test_is_object_reachable_by_single_grasp_delegates_to_is_reachable_by(
    immutable_model_world, monkeypatch
):
    """``as_single_grasp`` checks a single grasp pose at the object's pose."""
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")

    seq_calls = []
    single_calls = []
    monkeypatch.setattr(
        AreReachableBy, "__call__", lambda self, *a, **k: seq_calls.append(self) or True
    )
    monkeypatch.setattr(
        IsReachableBy,
        "__call__",
        lambda self, *a, **k: single_calls.append(self.pose) or True,
    )

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0
    )

    assert IsObjectReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        object_designator=milk,
        as_single_grasp=True,
    )()

    # Only the single-pose validator is used, against the current object pose.
    assert not seq_calls
    assert len(single_calls) == 1
    assert np.allclose(
        single_calls[0].to_position().to_np()[:3],
        milk.global_pose.to_position().to_np()[:3],
    )


def test_is_object_reachable_by_reachable(immutable_model_world):
    """End-to-end: a graspable object in front of the robot is reachable."""
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0
    )

    assert IsObjectReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        object_designator=milk,
        grasp_description=_right_front_grasp(view),
    )


def test_is_object_reachable_by_not_reachable(immutable_model_world):
    """End-to-end: an object far away from the robot is not reachable."""
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        5, 5, 0.7, 0, 0, 0
    )

    assert not IsObjectReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        object_designator=milk,
        grasp_description=_right_front_grasp(view),
    )
