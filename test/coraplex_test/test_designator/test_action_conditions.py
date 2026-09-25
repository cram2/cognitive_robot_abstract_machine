import pytest

from krrood.entity_query_language.factories import (
    evaluate_condition,
    ConditionType,
)
from coraplex.exceptions import ConditionNotSatisfied
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.querying.predicates import GripperIsFree
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.semantic_annotations.mixins import GraspPose
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk


def _construct_and_evaluate_condition(action, action_condition):

    condition = action_condition(
        action.bound_variables,
        action.context,
        action.designator_parameter,
    )
    evaluation = evaluate_condition(condition)
    if evaluation:
        return True
    raise ConditionNotSatisfied(
        pre_condition=True, action=action.__class__, condition=condition
    )


def test_get_bound_variables(immutable_model_world):
    world, view, context = immutable_model_world

    milk = world.get_semantic_annotations_by_type(Milk)[0]
    grasp = milk.grasp_poses()[0]
    pick_action = PickUpAction(grasp, context.robot.left_arm)

    bound_variables = pick_action._create_variables()

    assert len(bound_variables) == 15
    assert list(bound_variables.keys()) == [
        "position_threshold",
        "orientation_threshold",
        "grasp_detection_threshold",
        "pre_approach_linear_velocity",
        "final_approach_linear_velocity",
        "grasp_closing_velocity",
        "lift_linear_velocity",
        "grasp_stall_minimum_time",
        "object_friction",
        "approach_clearance",
        "retreat_distance",
        "grasp",
        "arm",
        "tolerate_grasp_stall",
        "perceive_before_grasp",
    ]
    assert list(bound_variables["arm"]._domain_) == [context.robot.left_arm]
    assert bound_variables["arm"]._type_ == type(context.robot.left_arm)
    assert list(bound_variables["grasp"]._domain_) == [grasp]
    assert bound_variables["grasp"]._type_ == GraspPose


def test_pick_up_pre_condition_leaves_reaching_to_the_attempt(mutable_model_world):
    """
    A precondition only checks the current state cheaply, so a grasp out of reach from
    where the robot starts does not refuse the pick-up while the gripper is free.
    """
    world, view, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_action = PickUpAction(milk.grasp_poses()[0], context.robot.left_arm)
    sequential([pick_action], context)

    assert _construct_and_evaluate_condition(pick_action, pick_action.pre_condition)


def test_pick_up_pre_condition_needs_a_free_gripper(mutable_model_world):
    world, view, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_action = PickUpAction(milk.grasp_poses()[0], context.robot.left_arm)
    # The standing pose from which the left arm reaches the milk.
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )
    plan = sequential([pick_action], context)

    with simulated_robot:
        plan.perform()

    pre_condition = pick_action.pre_condition(
        pick_action.bound_variables, context, pick_action.designator_parameter
    )
    assert pre_condition._name_ == GripperIsFree.__name__
    assert not evaluate_condition(pre_condition)


def test_pick_up_post_condition(mutable_model_world):
    world, view, context = mutable_model_world
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_action = PickUpAction(milk.grasp_poses()[0], context.robot.left_arm)
    # The standing pose test_pick_up_pre_condition establishes as reaching the milk.
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    plan = sequential([pick_action], context)

    assert _construct_and_evaluate_condition(pick_action, pick_action.pre_condition)

    with simulated_robot:
        plan.perform()

    assert world.get_body_by_name(
        "milk.stl"
    ) in world.get_kinematic_structure_entities_of_branch(
        view.left_arm.end_effector.tool_frame
    )

    assert _construct_and_evaluate_condition(pick_action, pick_action.post_condition)
