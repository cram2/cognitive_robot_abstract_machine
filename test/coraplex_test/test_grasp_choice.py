"""
How a pick-up settles on the grasp it takes.
"""

import numpy as np

from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from semantic_digital_twin.semantic_annotations.mixins import GraspPose
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose


def _reach_of(pick_up: PickUpAction) -> ReachAction:
    """
    :return: The reach the pick-up's plan performs.

    A pick-up reaches through the grasp it is built from, so the reach only appears
    once the plan below it has been expanded.
    """
    pick_up.plan_node.notify()
    [reach_node] = pick_up.plan_node.plan.get_nodes_by_designator_type(ReachAction)
    return reach_node.designator


def test_pick_up_takes_the_grasp_it_is_given(immutable_model_world):
    """
    A caller that settled on a grasp -- together with the pose the robot stands at, say
    -- has the pick-up take that one instead of ranking the object's grasps again.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    given = GraspPose(milk, Pose.from_xyz_rpy(yaw=np.pi / 3, reference_frame=milk.root))

    pick_up = PickUpAction(given, context.robot.left_arm)
    sequential([pick_up], context=context)

    assert pick_up.grasp is given


def test_pick_up_reaches_for_the_grasp_it_settled_on(immutable_model_world):
    """
    The grasp the pick-up chose is the one its plan reaches for, so a caller's choice
    reaches the motions rather than stopping at the action.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    given = GraspPose(milk, Pose.from_xyz_rpy(yaw=np.pi / 3, reference_frame=milk.root))

    pick_up = PickUpAction(given, context.robot.left_arm)
    sequential([pick_up], context=context)

    assert _reach_of(pick_up).grasp is given


def test_pick_up_keeps_its_grasp_even_when_it_cannot_be_reached(immutable_model_world):
    """
    The action takes the grasp it was given and no other.

    Quietly swapping in one that works would perform a different action than the one
    described.
    """
    world, view, context = immutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    pick_up = PickUpAction(milk.grasp_poses()[0], context.robot.left_arm)
    sequential([pick_up], context=context)

    np.testing.assert_allclose(
        _reach_of(pick_up).grasp.root_T_grasp.to_homogeneous_matrix().to_np(),
        milk.grasp_poses()[0].root_T_grasp.to_homogeneous_matrix().to_np(),
    )
