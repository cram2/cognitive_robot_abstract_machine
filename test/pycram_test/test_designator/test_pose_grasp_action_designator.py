from copy import deepcopy
from dataclasses import dataclass

import numpy
import pytest

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.failures import PlanFailure
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.plans.failures import PlanFailure
from pycram.robot_plans.actions.core.pose_grasp import (
    PoseGraspActionDescription,
    PoseGraspAndLiftActionDescription,
)
from pycram.robot_plans.actions.core.robot_body import ParkArmsActionDescription
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
    """Minimal concrete HasGraspPose implementation for testing."""

    def __post_init__(self):
        super().__post_init__()
        if self.grasp_pose is None:
            self.grasp_pose = Pose.from_xyz_rpy(
                roll=numpy.pi / 2,
                pitch=numpy.pi,
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
    hindrance_box = Body(
        name=PrefixedName("hindrance_box"),
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
        connection2 = Connection6DoF.create_with_dofs(
            copy_world,
            copy_world.root,
            hindrance_box,
            PrefixedName("hindrance_box_connection"),
            HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, 0.6, 0.93, reference_frame=tracy_world.root
            ),
        )
        copy_world.add_connection(connection2)

    return copy_world, copy_view, Context(copy_world, copy_view)


def _compute_grasp_pose(world, view, body_name: str, arm: Arms) -> Pose:
    """Derive a valid grasp pose for a body using GraspDescription."""
    arm_view = ViewManager.get_arm_view(arm, view)
    grasp_desc = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.TOP, arm_view.manipulator
    )
    return grasp_desc.grasp_pose(world.get_body_by_name(body_name))


def test_pose_grasp_precondition_fails_without_grasp_pose(pose_grasp_world):
    world, view, context = pose_grasp_world
    box_body = world.get_body_by_name("grasp_box")

    obj = GraspableBox(root=box_body, _world=world, grasp_pose="")
    action = PoseGraspActionDescription(target=obj, arm=Arms.LEFT).resolve()

    with pytest.raises(PlanFailure):
        action.validate_precondition()


def test_pose_grasp_and_lift_precondition_fails_without_grasp_pose(pose_grasp_world):
    world, view, context = pose_grasp_world
    box_body = world.get_body_by_name("grasp_box")

    obj = GraspableBox(root=box_body, _world=world, grasp_pose="")
    action = PoseGraspAndLiftActionDescription(target=obj, arm=Arms.LEFT).resolve()

    with pytest.raises(PlanFailure):
        action.validate_precondition()


def test_pose_grasp_action(pose_grasp_world, rclpy_node):
    VizMarkerPublisher(_world=pose_grasp_world[0], node=rclpy_node).with_tf_publisher()
    world, view, context = pose_grasp_world
    box_body = world.get_body_by_name("grasp_box")
    grasp_pose = _compute_grasp_pose(world, view, "grasp_box", Arms.LEFT)

    obj = GraspableBox(root=box_body, _world=world, grasp_pose=grasp_pose)
    with world.modify_world():
        world.add_semantic_annotation(obj)

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        PoseGraspActionDescription(target=obj, arm=Arms.LEFT),
    )
    with simulated_robot:
        plan.perform()

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

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        PoseGraspAndLiftActionDescription(target=obj, arm=Arms.LEFT),
    )
    with simulated_robot:
        plan.perform()

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    assert world.get_connection(left_arm.manipulator.tool_frame, box_body) is not None
