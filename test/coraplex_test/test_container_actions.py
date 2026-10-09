import numpy as np
import pytest

from krrood.entity_query_language.factories import evaluate_condition

from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from .conftest import left_or_only_arm

# %% fixtures

DRAWER_HANDLE = "handle_cab10_m"
"""
The handle of the apartment's middle cabinet drawer.
"""


@pytest.fixture
def pr2_at_drawer(pr2_apartment_context):
    """
    A PR2 in the apartment, with the drawer handle it works on annotated as such.
    """
    world, robot, context = pr2_apartment_context
    body = world.get_body_by_name(DRAWER_HANDLE)
    annotated = [
        handle
        for handle in world.get_semantic_annotations_by_type(Handle)
        if handle.root is body
    ]
    if not annotated:
        with world.modify_world():
            world.add_semantic_annotation_recursively(handle := Handle(root=body))
        annotated = [handle]
    return world, robot, context, annotated[0]


def motions_of(action) -> list:
    """
    :param action: The action whose plan to read.
    :return: The designators of the plan the action builds, in execution order.
    """
    return [node.designator for node in action._action_plan.children]


# %% backing off the handle


@pytest.mark.parametrize("action_type", [OpenAction, CloseAction])
def test_gripper_backs_off_the_handle_after_letting_go(pr2_at_drawer, action_type):
    """
    The gripper still straddles the handle once it opens, so the action ends by
    clearing it along the direction it approached from.
    """
    world, robot, context, handle = pr2_at_drawer
    action = action_type(handle, left_or_only_arm(robot))
    sequential([action], context)

    motions = motions_of(action)
    gripper_openings = [
        index
        for index, motion in enumerate(motions)
        if isinstance(motion, MoveGripperMotion) and motion.motion is GripperState.OPEN
    ]
    back_off = motions[-1]

    assert isinstance(back_off, MoveToolCenterPointMotion)
    assert gripper_openings[-1] == len(motions) - 2
    np.testing.assert_allclose(
        back_off.target.to_np(),
        action.back_off_pose(
            Pose(reference_frame=handle.root),
            left_or_only_arm(robot).end_effector,
        ).to_np(),
        atol=1e-9,
    )


def test_back_off_pose_clears_the_grasp_by_the_release_clearance(pr2_at_drawer):
    """
    The gripper backs straight out along the way it came in, by the clearance it is
    given and no further.
    """
    world, robot, context, handle = pr2_at_drawer
    action = OpenAction(handle, left_or_only_arm(robot))
    sequential([action], context)

    end_effector = left_or_only_arm(robot).end_effector
    grasp_pose = Pose(reference_frame=handle.root)
    tool_goal = end_effector.tool_frame_goal(grasp_pose)
    back_off = action.back_off_pose(grasp_pose, end_effector)

    np.testing.assert_allclose(
        back_off.to_np()[:3, :3], tool_goal.to_np()[:3, :3], atol=1e-9
    )
    np.testing.assert_allclose(
        back_off.to_np()[:3, 3] - tool_goal.to_np()[:3, 3],
        -action.release_clearance * np.asarray(grasp_pose.to_np()[:3, 0]),
        atol=1e-9,
    )


# %% what the actions promise


def test_open_promises_the_container_is_open_rather_than_still_held(pr2_at_drawer):
    """
    The action ends with the gripper clear of the handle, so what it leaves behind is
    an open container, not a handle still between the fingers.
    """
    world, robot, context, handle = pr2_at_drawer
    action = OpenAction(handle, left_or_only_arm(robot))
    sequential([action], context)
    drawer = handle.root.get_first_parent_connection_of_type(ActiveConnection1DOF)
    drawer.position = 0.45
    world.notify_state_change()

    assert evaluate_condition(
        action.post_condition(
            action.bound_variables, context, action.designator_parameter
        )
    )
