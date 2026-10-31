"""
The real DAiSy driving its grippers through their griplink action servers, skipped where
``griplink_interfaces`` is not installed.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("griplink_interfaces")

from coraplex.griplink_grippers import RealDAiSyPlanExecutor
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveFingersToWidthAction,
    SetGripperAction,
)
from cramph.context import ContextExtension, StatechartContext
from giskardpy.motion_statechart.goals.gripper import MoveFingers
from giskardpy.motion_statechart.ros2_nodes.griplink import (
    GriplinkActionServerTask,
    GriplinkFlexActionServerTask,
    GriplinkPresetActionServerTask,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyLeftGripper
from semantic_digital_twin.robots.griplink_gripper import GriplinkFlexConfiguration
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from .test_robot_specific_plan_transformations import prepared
from ...plan_running import robot_executor, robot_extensions, world_of

# %% helpers


@pytest.fixture
def daisy_extensions(daisy_world) -> list[ContextExtension]:
    world = deepcopy(daisy_world)
    return robot_extensions(world.get_semantic_annotations_by_type(DAiSy)[0])


def real_daisy_executor(extensions: list[ContextExtension]) -> RealDAiSyPlanExecutor:
    """
    :param extensions: The context extensions of the plan, holding a
        :class:`~coraplex.plans.context_extensions.RobotAccess` of DAiSy.
    :return: An executor sending plans to the real DAiSy.
    """
    return RealDAiSyPlanExecutor(
        StatechartContext.with_extensions(world_of(extensions), extensions)
    )


def left_gripper_of(extensions: list[ContextExtension]) -> DAiSyLeftGripper:
    """
    :return: The left gripper of the DAiSy in `extensions`.
    """
    return (
        world_of(extensions)
        .get_semantic_annotations_by_type(DAiSy)[0]
        .left_arm.end_effector
    )


def top_down_grasp_of_a_box(extensions: list[ContextExtension]) -> GraspCandidate:
    """
    :return: A grasp reaching down onto a new box on the table in front of the left arm
        of the DAiSy in `extensions`.
    """
    world = world_of(extensions)
    box = Body(
        name=PrefixedName("box"),
        collision=ShapeCollection([Box(scale=Scale(0.05, 0.05, 0.1))]),
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.6, 1.0, 0.9
                ),
            )
        )
        graspable = HasGraspCandidates(root=box)
        world.add_semantic_annotations([graspable])
    return GraspCandidate(
        graspable, Pose.from_xyz_rpy(pitch=np.pi / 2, reference_frame=box)
    )


# %% the gripper states


def test_the_real_daisy_moves_its_gripper_to_a_preset_state_through_the_griplink_server(
    daisy_extensions,
) -> None:
    gripper = left_gripper_of(daisy_extensions)
    configuration = gripper.default_configuration(GripperState.CLOSE)

    action = prepared(
        SetGripperAction(configuration=configuration),
        real_daisy_executor(daisy_extensions),
    )

    assert type(action.goal) is GriplinkPresetActionServerTask
    assert (
        action.goal.action_topic
        == gripper.griplink_endpoint(GripperState.CLOSE).action_topic
    )


def test_the_real_daisy_moves_its_gripper_to_a_width_through_the_griplink_server(
    daisy_extensions,
) -> None:
    gripper = left_gripper_of(daisy_extensions)
    configuration = GriplinkFlexConfiguration.from_state_type(
        gripper, GripperState.FLEXCLOSE, grip_position=0.04
    )

    action = prepared(
        SetGripperAction(configuration=configuration),
        real_daisy_executor(daisy_extensions),
    )

    assert type(action.goal) is GriplinkFlexActionServerTask
    assert action.goal.grip_position == configuration.grip_position


def test_a_daisy_driven_by_a_standalone_giskard_moves_its_fingers_with_a_giskard_goal(
    daisy_extensions,
) -> None:
    configuration = left_gripper_of(daisy_extensions).default_configuration(
        GripperState.CLOSE
    )

    action = prepared(
        SetGripperAction(configuration=configuration),
        robot_executor(daisy_extensions),
    )

    assert type(action.goal) is type(MoveFingers.moving_gripper_to(configuration))


# %% grasping


def test_the_real_daisy_opens_and_shuts_its_gripper_on_a_grasp_through_the_griplink_server(
    daisy_extensions,
) -> None:
    grasp = top_down_grasp_of_a_box(daisy_extensions)
    arm = world_of(daisy_extensions).get_semantic_annotations_by_type(DAiSy)[0].left_arm

    action = prepared(
        GraspingAction(grasp=grasp, arm=arm), real_daisy_executor(daisy_extensions)
    )

    assert action.gripper_goal_type is GriplinkActionServerTask
    assert not any(
        isinstance(node, MoveFingersToWidthAction) for node in action.descendants
    )
    assert [
        node.action_topic
        for node in action.descendants
        if isinstance(node, GriplinkPresetActionServerTask)
    ] == [
        arm.end_effector.griplink_endpoint(state).action_topic
        for state in (GripperState.OPEN, GripperState.CLOSE)
    ]
