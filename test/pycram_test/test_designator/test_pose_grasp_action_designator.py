from copy import deepcopy
from dataclasses import dataclass
from typing import Generator

import numpy
import pytest

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.plans.failures import PlanFailure
from pycram.robot_plans.actions.core.pose_grasp import (
    PoseGraspAction,
    PoseGraspAndLiftAction,
)
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction
from pycram.view_manager import ViewManager
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasGraspPose
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False)
class GraspableBox(HasGraspPose):
    """Concrete HasGraspPose with one grasp pose per side of the box (4 yaw orientations).

    If grasp_pose is explicitly set to a falsy non-None value the object yields no
    grasp poses (used by precondition tests). If it is set to a Pose instance that
    single pose is yielded instead.
    """

    def grasp_poses(self) -> Generator[Pose, None, None]:
        if self.grasp_pose is not None:
            yield from super().grasp_poses()
            return
        for yaw in [0, numpy.pi / 2, numpy.pi, 3 * numpy.pi / 2]:
            yield Pose.from_xyz_rpy(
                roll=numpy.pi / 2,
                pitch=numpy.pi,
                yaw=yaw,
                reference_frame=self.root,
            )


@pytest.fixture
def pose_grasp_world(tracy_world):
    """Function-scoped world with a graspable box placed in reach of Tracy's left arm."""
    copy_world = deepcopy(tracy_world)
    copy_view = Tracy.from_world(copy_world)

    box = Body(
        name=PrefixedName("grasp_box"),
        collision=ShapeCollection([Box(scale=Scale(0.07, 0.07, 0.1))]),
        visual=ShapeCollection([Box(scale=Scale(0.07, 0.07, 0.1))]),
    )
    with copy_world.modify_world():
        connection = Connection6DoF.create_with_dofs(
            copy_world,
            copy_world.root,
            box,
            PrefixedName("grasp_box_connection"),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, 0.5, 0.93, reference_frame=tracy_world.root
            ),
        )
        copy_world.add_connection(connection)

    return copy_world, copy_view, Context(copy_world, copy_view)


@pytest.fixture
def pose_grasp_world_with_obstacle(pose_grasp_world):
    """Extends pose_grasp_world with an obstacle box that blocks the first grasp pose (yaw=0)."""
    copy_world, copy_view, context = pose_grasp_world

    obstacle = Body(
        name=PrefixedName("obstacle_box"),
        collision=ShapeCollection([Box(scale=Scale(0.07, 0.07, 0.1))]),
        visual=ShapeCollection([Box(scale=Scale(0.07, 0.07, 0.1))]),
    )
    with copy_world.modify_world():
        connection = Connection6DoF.create_with_dofs(
            copy_world,
            copy_world.root,
            obstacle,
            PrefixedName("obstacle_box_connection"),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                0.79568, 0.68311, 0.87959, reference_frame=copy_world.root
            ),
        )
        copy_world.add_connection(connection)

    return copy_world, copy_view, context


def test_pose_grasp_precondition_fails_without_grasp_pose(pose_grasp_world):
    world, view, context = pose_grasp_world
    box_body = world.get_body_by_name("grasp_box")

    obj = GraspableBox(root=box_body, _world=world, grasp_pose="")
    action = PoseGraspAction(target=obj, arm=Arms.LEFT)

    with pytest.raises(PlanFailure):
        action.validate_precondition()


def test_pose_grasp_and_lift_precondition_fails_without_grasp_pose(pose_grasp_world):
    world, view, context = pose_grasp_world
    box_body = world.get_body_by_name("grasp_box")

    obj = GraspableBox(root=box_body, _world=world, grasp_pose="")
    action = PoseGraspAndLiftAction(target=obj, arm=Arms.LEFT)

    with pytest.raises(PlanFailure):
        action.validate_precondition()


def test_pose_grasp_action(pose_grasp_world, rclpy_node):
    VizMarkerPublisher(_world=pose_grasp_world[0], node=rclpy_node).with_tf_publisher()
    world, view, context = pose_grasp_world
    box_body = world.get_body_by_name("grasp_box")

    obj = GraspableBox(root=box_body, _world=world)
    with world.modify_world():
        world.add_semantic_annotation(obj)

    with simulated_robot:
        sequential(
            [ParkArmsAction(Arms.BOTH), PoseGraspAction(target=obj, arm=Arms.LEFT)],
            context,
        ).perform()

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    tool_frame_position = (
        left_arm.manipulator.tool_frame.global_pose.to_position().to_np()
    )
    box_world_position = box_body.global_pose.to_position().to_np()
    assert tool_frame_position[:3] == pytest.approx(box_world_position[:3], abs=0.02)


def test_pose_grasp_and_lift_action(pose_grasp_world, rclpy_node):
    VizMarkerPublisher(_world=pose_grasp_world[0], node=rclpy_node).with_tf_publisher()
    world, view, context = pose_grasp_world
    box_body = world.get_body_by_name("grasp_box")

    obj = GraspableBox(root=box_body, _world=world)
    with world.modify_world():
        world.add_semantic_annotation(obj)

    with simulated_robot:
        sequential(
            [
                ParkArmsAction(Arms.BOTH),
                PoseGraspAndLiftAction(target=obj, arm=Arms.LEFT),
            ],
            context,
        ).perform()

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    assert world.get_connection(left_arm.manipulator.tool_frame, box_body) is not None


def test_pose_grasp_action_skips_blocked_pose(
    pose_grasp_world_with_obstacle, rclpy_node
):
    """The obstacle blocks the first grasp pose (yaw=0); the action must fall back to a free pose."""
    VizMarkerPublisher(
        _world=pose_grasp_world_with_obstacle[0], node=rclpy_node
    ).with_tf_publisher()
    world, view, context = pose_grasp_world_with_obstacle
    box_body = world.get_body_by_name("grasp_box")

    obj = GraspableBox(root=box_body, _world=world)
    with world.modify_world():
        world.add_semantic_annotation(obj)

    action = PoseGraspAction(target=obj, arm=Arms.LEFT)

    with simulated_robot:
        sequential(
            [ParkArmsAction(Arms.BOTH), action],
            context,
        ).perform()

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    tool_frame_position = (
        left_arm.manipulator.tool_frame.global_pose.to_position().to_np()
    )
    box_world_position = box_body.global_pose.to_position().to_np()
    assert tool_frame_position[:3] == pytest.approx(box_world_position[:3], abs=0.02)
    assert action._resolved_grasp_pose is not None
