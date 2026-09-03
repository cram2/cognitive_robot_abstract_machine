import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% fixtures

HELD_AT = (0.03, -0.02, 0.05)
"""
Where the held object's origin sits relative to the tool frame, in meters.

Deliberately off-centre: an object is rarely gripped exactly at its own origin, and a
grasp on the rim of a bowl never is.
"""

HELD_YAW = np.pi / 3
"""
How far the held object is turned about the tool frame's z-axis.
"""


@pytest.fixture
def pr2_holding_milk(mutable_simple_pr2_world):
    """
    A PR2 whose left tool frame holds the milk off-centre, at :data:`HELD_AT`.
    """
    world, robot, _ = mutable_simple_pr2_world
    milk = world.get_body_by_name("milk.stl")
    tool_frame = ViewManager.get_end_effector_view(Arms.LEFT, robot).tool_frame
    with world.modify_world():
        world.move_branch(milk, tool_frame)
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
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
    place = PlaceAction(milk, target, Arms.LEFT)
    sequential([place], context=Context(world, robot))

    end_effector = ViewManager.get_end_effector_view(Arms.LEFT, robot)
    tool_goal = end_effector.tool_frame_goal(place._grasp_pose_at(target))

    tool_T_milk = world.transform(milk.global_transform, end_effector.tool_frame)
    placed_milk = tool_goal.to_homogeneous_matrix() @ tool_T_milk

    np.testing.assert_allclose(
        placed_milk.to_np(), target.to_homogeneous_matrix().to_np(), atol=1e-9
    )


# %% releasing what has not been picked up yet


def test_place_uses_the_grasp_its_pick_up_will_take(mutable_model_world):
    """
    A plan is built before it runs, so a place that follows a pick-up in the same plan
    is expanded while the object is still on its shelf, nowhere near the gripper. The
    grasp then has to come from the pick-up that is going to take it, not from where
    the object happens to lie.
    """
    world, robot, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)

    pick_up = PickUpAction(milk, Arms.LEFT)
    place = PlaceAction(milk.root, target, Arms.LEFT)
    sequential([pick_up, place], context=context)

    np.testing.assert_allclose(
        place._grasp_on_the_held_object().to_np(), pick_up.grasp_pose.to_np(), atol=1e-9
    )


def test_place_without_a_preceding_pick_up_grasps_at_the_objects_origin(
    mutable_model_world,
):
    """
    Nothing in the plan and nothing in the gripper leaves only the object's own frame to
    go on, since a place is handed a body rather than an annotation to ask.
    """
    world, robot, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, reference_frame=world.root)

    place = PlaceAction(milk.root, target, Arms.LEFT)
    sequential([place], context=context)

    np.testing.assert_allclose(
        place._grasp_on_the_held_object().to_np(), np.eye(4), atol=1e-9
    )
