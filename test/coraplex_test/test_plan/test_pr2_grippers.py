"""
The real PR2 driving its grippers through their gripper action servers, skipped where
``pr2_controllers_msgs`` is not installed.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("pr2_controllers_msgs")

from coraplex.pr2_grippers import RealPR2PlanExecutor
from coraplex.robot_plans.actions.core.pick_up import GraspingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveFingersToWidthAction,
    SetGripperAction,
)
from cramph.context import ContextExtension, StatechartContext
from giskardpy.motion_statechart.ros2_nodes.pr2_gripper import (
    PR2GripperActionServerTask,
)
from giskardpy.motion_statechart.tasks.gripper import MoveFingers
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.pr2 import PR2

from .test_robot_specific_plan_transformations import prepared, top_down_grasp_of_a_box
from ...plan_running import (
    robot_extensions,
    robot_of,
    simulated_executor,
    world_of,
)

# %% helpers


@pytest.fixture()
def extensions(pr2_world_state_reset) -> list[ContextExtension]:
    """
    The context extensions of the plans of the PR2.
    """
    world = deepcopy(pr2_world_state_reset)
    return robot_extensions(world.get_semantic_annotations_by_type(PR2)[0])


def real_pr2_executor(extensions: list[ContextExtension]) -> RealPR2PlanExecutor:
    """
    :return: An executor sending plans to the real PR2 in `extensions`.
    """
    return RealPR2PlanExecutor(
        context=StatechartContext.with_extensions(
            world=world_of(extensions), extensions=extensions
        )
    )


# %% the gripper states


@pytest.mark.parametrize("state", [GripperState.OPEN, GripperState.CLOSE])
def test_the_real_pr2_moves_its_gripper_through_the_gripper_action(
    extensions, state
) -> None:
    """
    The real PR2 moves its gripper through its gripper action, to the gap of the
    configuration asked for.
    """
    for arm in robot_of(extensions).all_arms:
        gripper = arm.end_effector
        configuration = gripper.default_configuration(state)

        action = prepared(
            SetGripperAction(gripper_configuration=configuration),
            real_pr2_executor(extensions),
        )

        assert type(action.goal) is PR2GripperActionServerTask
        assert (action.goal.action_topic, action.goal.target_gap) == (
            gripper.gripper_action_name,
            gripper.target_gap(joint_state=configuration.joint_state),
        )


def test_a_simulated_pr2_moves_its_fingers_with_a_giskard_goal(extensions) -> None:
    configuration = (
        robot_of(extensions)
        .all_arms[0]
        .end_effector.default_configuration(GripperState.CLOSE)
    )

    action = prepared(
        SetGripperAction(gripper_configuration=configuration),
        simulated_executor(extensions),
    )

    assert type(action.goal) is type(
        MoveFingers.create_gripper_goal(configuration=configuration)
    )


# %% grasping


def test_the_real_pr2_opens_and_shuts_its_gripper_on_a_grasp_through_the_gripper_action(
    extensions,
) -> None:
    """
    On a grasp the real PR2 opens and then shuts its gripper through its gripper action,
    instead of closing its fingers onto the object's width with a Giskard goal.
    """
    grasp = top_down_grasp_of_a_box(extensions)
    arm = robot_of(extensions).all_arms[0]

    action = prepared(
        GraspingAction(grasp_candidate=grasp, arm=arm),
        real_pr2_executor(extensions),
    )

    assert action.gripper_goal_type is PR2GripperActionServerTask
    assert not any(
        isinstance(node, MoveFingersToWidthAction) for node in action.descendants
    )
    assert [
        node.target_gap
        for node in action.descendants
        if isinstance(node, PR2GripperActionServerTask)
    ] == [
        arm.end_effector.target_gap(
            joint_state=arm.end_effector.get_joint_state_by_type(state)
        )
        for state in (GripperState.OPEN, GripperState.CLOSE)
    ]
