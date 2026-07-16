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
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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

    assert len(bound_variables) == 3
    assert list(bound_variables.keys()) == [
        "object_designator",
        "arm",
        "grasp_description",
    ]
    assert list(bound_variables["arm"]._domain_) == [Arms.LEFT]
    assert bound_variables["arm"]._type_ == Arms
    assert list(bound_variables["object_designator"]._domain_) == [
        world.get_body_by_name("milk.stl")
    ]
    assert bound_variables["object_designator"]._type_ == Body


def test_pick_up_pre_conditions(mutable_model_world):
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

    plan = sequential([pick_action], context)

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


def test_context_evaluate_condition(mutable_model_world):
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
    # Make action impossible
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.0, 2, 0
    )

    plan = sequential([pick_action], context)
    with pytest.raises(ConditionNotSatisfied):
        with simulated_robot:
            plan.perform()

    context.evaluate_conditions = False

    with pytest.raises(MotionDidNotFinish):
        with simulated_robot:
            plan.perform()
