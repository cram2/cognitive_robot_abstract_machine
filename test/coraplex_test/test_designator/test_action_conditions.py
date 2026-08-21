import pytest

from krrood.entity_query_language.factories import (
    get_false_statements,
    evaluate_condition,
    ConditionType,
)
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.exceptions import ConditionNotSatisfied, MotionDidNotFinish
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.container import CloseAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.world_entity import Body


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

    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )

    bound_variables = pick_action._create_variables()

    assert len(bound_variables) == 11
    assert list(bound_variables.keys()) == [
        "grasp_detection_threshold",
        "pre_approach_linear_velocity",
        "final_approach_linear_velocity",
        "grasp_closing_velocity",
        "lift_linear_velocity",
        "grasp_stall_minimum_time",
        "object_friction",
        "object_designator",
        "arm",
        "grasp_description",
        "tolerate_grasp_stall",
    ]
    assert list(bound_variables["arm"]._domain_) == [Arms.LEFT]
    assert bound_variables["arm"]._type_ == Arms
    assert list(bound_variables["object_designator"]._domain_) == [
        world.get_body_by_name("milk.stl")
    ]
    assert bound_variables["object_designator"]._type_ == Body


def test_pick_up_pre_conditions(mutable_model_world):
    """
    The pre condition reports whether the object can be reached, and says which of its
    statements is the one that fails.

    The robot drives while it reaches, so an object it cannot get to is one out of the
    height its arm rises to rather than one merely standing far away.
    """
    world, view, context = mutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk_origin_within_reach = milk.parent_connection.origin

    pick_action = PickUpAction(
        milk,
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )

    plan = sequential([pick_action], context)

    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 3, reference_frame=milk.parent_connection.parent
    )

    with pytest.raises(ConditionNotSatisfied):
        _construct_and_evaluate_condition(
            pick_action,
            pick_action.pre_condition,
        )

    pre_condition = pick_action.pre_condition(
        pick_action.bound_variables, context, pick_action.designator_parameter
    )

    false_statements = get_false_statements(pre_condition)

    assert len(false_statements) == 1
    assert false_statements[0]._name_ == "IsObjectReachableBy"

    with pytest.raises(ConditionNotSatisfied):
        _construct_and_evaluate_condition(pick_action, pick_action.pre_condition)

    milk.parent_connection.origin = milk_origin_within_reach
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )

    pre_condition = pick_action.pre_condition(
        pick_action.bound_variables, context, pick_action.designator_parameter
    )

    assert evaluate_condition(pre_condition) == True

    with simulated_robot:
        plan.perform()

    assert evaluate_condition(pre_condition) == False
    _construct_and_evaluate_condition(pick_action, pick_action.post_condition)
    assert _construct_and_evaluate_condition(pick_action, pick_action.post_condition)


def test_pick_up_post_condition(mutable_model_world):
    world, view, context = mutable_model_world
    pick_action = PickUpAction(
        world.get_body_by_name("milk.stl"),
        Arms.LEFT,
        GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
    )
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.8, 2, 0
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


# %% how far short of shut still counts as closed


def test_close_action_judges_a_container_by_the_tolerance_it_was_given(
    mutable_model_world,
):
    """
    A mechanism is only driven onto its own limit asymptotically, so how far short of
    the goal still counts as closed is the caller's to say.
    """
    world, view, context = mutable_model_world
    handle = world.get_body_by_name("handle_cab10_m")
    close_action = CloseAction(
        handle, Arms.LEFT, goal_joint_state=0.0, goal_joint_state_tolerance=0.3
    )
    connection = handle.get_first_parent_connection_of_type(ActiveConnection1DOF)
    post_condition = close_action.post_condition(
        close_action.bound_variables, context, close_action.designator_parameter
    )
    tolerance = close_action.goal_joint_state_tolerance

    connection.position = close_action.goal_joint_state + tolerance / 2
    stopped_within_tolerance = evaluate_condition(post_condition)
    connection.position = close_action.goal_joint_state + tolerance * 2
    stopped_beyond_tolerance = evaluate_condition(post_condition)

    assert stopped_within_tolerance
    assert not stopped_beyond_tolerance
