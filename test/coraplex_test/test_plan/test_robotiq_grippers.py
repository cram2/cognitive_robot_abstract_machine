"""
Real robots driving their grippers through parallel gripper command action servers,
skipped where ``control_msgs`` is not installed.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("control_msgs")

from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveFingersToWidthAction,
    SetGripperAction,
)
from cramph.context import ContextExtension
from giskardpy.motion_statechart.goals.gripper import MoveFingers
from giskardpy.motion_statechart.ros2_nodes.robotiq import (
    RobotiqGripperActionServerTask,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.robots.tracy import Tracy

from .test_robot_specific_plan_transformations import prepared, top_down_grasp_of_a_box
from ...plan_running import (
    robot_executor,
    robot_extensions,
    robot_of,
    simulated_executor,
)

# %% helpers


@pytest.fixture(
    params=[(Tracy, "tracy_world"), (Tiago, "_tiago_world_setup")],
    ids=["Tracy", "Tiago"],
)
def extensions(request) -> list[ContextExtension]:
    """
    The context extensions of the plans of a robot whose grippers have a parallel
    gripper command server.
    """
    robot_type: type[AbstractRobot]
    robot_type, world_fixture = request.param
    world = deepcopy(request.getfixturevalue(world_fixture))
    return robot_extensions(world.get_semantic_annotations_by_type(robot_type)[0])


# %% the gripper states


@pytest.mark.parametrize("state", [GripperState.OPEN, GripperState.CLOSE])
def test_a_real_robot_moves_its_gripper_through_the_parallel_gripper_command_server(
    extensions, state
) -> None:
    gripper = robot_of(extensions).all_arms[0].end_effector
    configuration = gripper.default_configuration(state)

    action = prepared(
        SetGripperAction(configuration=configuration), robot_executor(extensions)
    )

    assert type(action.goal) is RobotiqGripperActionServerTask
    assert action.goal.action_topic == gripper.parallel_gripper_command_topic
    assert action.goal.target_position == gripper.separation_at(
        configuration.joint_state
    )


def test_a_simulated_robot_moves_its_fingers_with_a_giskard_goal(extensions) -> None:
    configuration = (
        robot_of(extensions)
        .all_arms[0]
        .end_effector.default_configuration(GripperState.CLOSE)
    )

    action = prepared(
        SetGripperAction(configuration=configuration), simulated_executor(extensions)
    )

    assert type(action.goal) is type(MoveFingers.moving_gripper_to(configuration))


def test_a_real_robot_whose_grippers_take_other_action_messages_keeps_its_finger_goals(
    daisy_world,
) -> None:
    extensions = robot_extensions(
        deepcopy(daisy_world).get_semantic_annotations_by_type(DAiSy)[0]
    )
    configuration = (
        robot_of(extensions)
        .all_arms[0]
        .end_effector.default_configuration(GripperState.CLOSE)
    )

    action = prepared(
        SetGripperAction(configuration=configuration), robot_executor(extensions)
    )

    assert action.gripper_goal_type is MoveFingers
    assert type(action.goal) is type(MoveFingers.moving_gripper_to(configuration))


# %% grasping


def test_a_real_robot_opens_and_shuts_its_gripper_on_a_grasp_through_the_parallel_gripper_command_server(
    extensions,
) -> None:
    grasp = top_down_grasp_of_a_box(extensions)
    arm = robot_of(extensions).all_arms[0]

    action = prepared(GraspingAction(grasp=grasp, arm=arm), robot_executor(extensions))

    assert action.gripper_goal_type is RobotiqGripperActionServerTask
    assert not any(
        isinstance(node, MoveFingersToWidthAction) for node in action.descendants
    )
    assert [
        node.target_position
        for node in action.descendants
        if isinstance(node, RobotiqGripperActionServerTask)
    ] == [
        arm.end_effector.separation_at(
            arm.end_effector.default_configuration(state).joint_state
        )
        for state in (GripperState.OPEN, GripperState.CLOSE)
    ]
