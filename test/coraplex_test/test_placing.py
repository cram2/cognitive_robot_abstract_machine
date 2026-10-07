import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.exceptions import ObjectIsNotHeld
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose

from ..conftest import SAMPLING_SEED

# %% fixtures

HELD_AT = (0.03, -0.02, 0.05)
"""
Where the held object's origin sits relative to the tool frame, in meters; off-centre,
as a rim grasp would hold it.
"""

HELD_YAW = np.pi / 3
"""
How far the held object is turned about the tool frame's z-axis.
"""


@pytest.fixture
def pr2_holding_milk(simple_pr2_context):
    """
    A PR2 whose left tool frame holds the milk off-centre, at :data:`HELD_AT`.
    """
    world, robot, _ = simple_pr2_context
    milk_body = world.get_body_by_name("milk.stl")
    milk = Milk(root=milk_body)
    tool_frame = robot.left_arm.end_effector.tool_frame
    with world.modify_world():
        world.move_branch(milk_body, tool_frame)
        world.add_semantic_annotation(milk)
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        *HELD_AT, yaw=HELD_YAW, reference_frame=tool_frame
    )
    world.notify_state_change()
    return world, robot, milk


# %% releasing what is actually held


def test_place_derives_the_grasp_from_the_live_tool_frame_transform(pr2_holding_milk):
    """
    Sending the tool frame to the target pose itself would place the object wherever
    the grasp happens to hold it, which is beside the target unless the object is held
    at its own origin. The release has to account for the transform the gripper
    actually holds it at.
    """
    world, robot, milk = pr2_holding_milk
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, yaw=np.pi / 4, reference_frame=world.root)
    place = PlaceAction(milk, target)
    sequential([place], context=Context(world, robot, sampling_seed=SAMPLING_SEED))

    end_effector = robot.left_arm.end_effector
    tool_goal = end_effector.tool_frame_goal(
        place._grasp_on_the_held_object().moved_to(target)
    )

    tool_T_milk = world.transform(milk.root.global_transform, end_effector.tool_frame)
    placed_milk = tool_goal.homogeneous_matrix @ tool_T_milk

    np.testing.assert_allclose(
        placed_milk.to_np(), target.homogeneous_matrix.to_np(), atol=1e-9
    )


def test_a_place_runs_the_grasp_backwards(pr2_holding_milk):
    """
    A release comes down onto the target from where a pick-up would lift the object to,
    and leaves the way a pick-up would approach it.
    """
    world, robot, milk = pr2_holding_milk
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)
    place = PlaceAction(milk, target)
    sequential([place], context=Context(world, robot, sampling_seed=SAMPLING_SEED))
    grasp = place._grasp_on_the_held_object()
    poses = place.grasp_pose_sequence(
        grasp.moved_to(target), robot.left_arm.end_effector, grasp
    )

    root = place._action_plan
    tool_goals = [
        node.designator.target
        for node in root.descendants
        if isinstance(node, MotionNode)
        and isinstance(node.designator, MoveToolCenterPointMotion)
    ]

    assert len(tool_goals) == 3
    for expected, actual in zip(
        [poses.retreat, poses.grasp, poses.pre_grasp], tool_goals
    ):
        np.testing.assert_allclose(expected.to_np(), actual.to_np(), atol=1e-9)


# %% releasing what has not been picked up yet


def test_place_uses_the_grasp_its_pick_up_will_take(pr2_apartment_context):
    """
    A plan is built before it runs, so a place that follows a pick-up in the same plan
    is expanded while the object is still on its shelf, nowhere near the gripper. The
    grasp then has to come from the pick-up that is going to take it, not from where
    the object happens to lie.
    """
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)

    pick_up = PickUpAction(milk.grasp_candidates()[0], context.robot.left_arm)
    place = PlaceAction(milk, target)
    sequential([pick_up, place], context=context)

    np.testing.assert_allclose(
        place._grasp_on_the_held_object().grasp_pose.to_np(),
        pick_up.grasp.grasp_pose.to_np(),
        atol=1e-9,
    )


def test_a_place_of_an_object_nothing_holds_is_refused(pr2_apartment_context):
    """
    Nothing in the gripper and no pick-up before it leaves no arm to place with.
    """
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)

    place = PlaceAction(milk, target)
    sequential([place], context=context)

    with pytest.raises(ObjectIsNotHeld):
        place._action_plan


# %% the arm that places


def _arms_moved_by(place: PlaceAction) -> set[Arm]:
    """
    :return: The arms whose tool frames the plan of `place` moves.
    """
    root = place._action_plan
    return {
        node.designator.arm
        for node in [root] + root.descendants
        if isinstance(node, MotionNode)
        and isinstance(node.designator, MoveToolCenterPointMotion)
    }


def test_place_takes_the_arm_that_holds_the_object(pr2_holding_milk):
    world, robot, milk = pr2_holding_milk
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)
    place = PlaceAction(milk, target)
    sequential([place], context=Context(world, robot, sampling_seed=SAMPLING_SEED))

    assert _arms_moved_by(place) == {robot.left_arm}


def test_place_takes_the_arm_its_pick_up_will_use(pr2_apartment_context):
    """
    A plan is built before it runs, so a place that follows a pick-up in the same plan
    places with the arm that pick-up is going to hold the object in.
    """
    world, robot, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)
    pick_up = PickUpAction(milk.grasp_candidates()[0], robot.right_arm)
    place = PlaceAction(milk, target)
    sequential([pick_up, place], context=context)

    assert _arms_moved_by(place) == {robot.right_arm}
