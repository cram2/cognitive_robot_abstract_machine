import pytest
from krrood.entity_query_language.factories import (
    evaluate_condition,
)
from coraplex.exceptions import ConditionNotSatisfied
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.querying.predicates import GripperIsFree, ToolFrameIsAtGrasp
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

LEFT_ARM_REACHES_THE_MILK_FROM = HomogeneousTransformationMatrix.from_xyz_rpy(
    1.9, 1.4, 0
)
"""
Where the PR2 stands so that its left arm reaches the milk in the apartment.
"""


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


def test_get_bound_variables(pr2_apartment_context):
    world, view, context = pr2_apartment_context

    milk = world.get_semantic_annotations_by_type(Milk)[0]
    grasp = milk.grasp_candidates()[0]
    pick_action = PickUpAction(grasp=grasp, arm=context.robot.left_arm)

    bound_variables = pick_action._create_variables()

    assert len(bound_variables) == 14
    assert list(bound_variables.keys()) == [
        "grasp_detection_threshold",
        "pre_approach_linear_velocity",
        "final_approach_linear_velocity",
        "grasp_closing_velocity",
        "lift_linear_velocity",
        "grasp_stall_minimum_time",
        "object_friction",
        "approach_clearance",
        "retreat_distance",
        "position_threshold",
        "orientation_threshold",
        "arm",
        "grasp",
        "tolerate_grasp_stall",
    ]
    assert list(bound_variables["arm"]._domain_) == [context.robot.left_arm]
    assert bound_variables["arm"]._type_ == type(context.robot.left_arm)
    assert list(bound_variables["grasp"]._domain_) == [grasp]
    assert bound_variables["grasp"]._type_ == GraspCandidate


def test_pick_up_pre_condition_leaves_reaching_to_the_attempt(pr2_apartment_context):
    """
    A pick-up is not refused for an unreachable grasp while the gripper is free.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_action = PickUpAction(
        grasp=milk.grasp_candidates()[0], arm=context.robot.left_arm
    )
    sequential([pick_action], context)

    assert _construct_and_evaluate_condition(pick_action, pick_action.pre_condition)


def test_pick_up_pre_condition_needs_a_free_gripper(pr2_apartment_context):
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_action = PickUpAction(
        grasp=milk.grasp_candidates()[0], arm=context.robot.left_arm
    )
    view.root.parent_connection.origin = LEFT_ARM_REACHES_THE_MILK_FROM
    plan = sequential([pick_action], context)
    pre_condition = pick_action.pre_condition(
        pick_action.bound_variables, context, pick_action.designator_parameter
    )
    assert pre_condition._name_ == GripperIsFree.__name__
    assert evaluate_condition(pre_condition)

    with simulated_robot:
        plan.perform()

    assert not evaluate_condition(pre_condition)


def test_pick_up_post_condition_needs_the_object_itself_in_the_gripper(
    pr2_apartment_context,
):
    """
    A gripper holding another body has not picked up the object.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_action = PickUpAction(
        grasp=milk.grasp_candidates()[0], arm=context.robot.left_arm
    )
    sequential([pick_action], context)
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=view.left_arm.end_effector.tool_frame,
                child=Body(name=PrefixedName("something_else")),
            )
        )

    post_condition = pick_action.post_condition(
        pick_action.bound_variables, context, pick_action.designator_parameter
    )

    assert not evaluate_condition(post_condition)


def test_pick_up_post_condition(pr2_apartment_context):
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_action = PickUpAction(
        grasp=milk.grasp_candidates()[0], arm=context.robot.left_arm
    )
    view.root.parent_connection.origin = LEFT_ARM_REACHES_THE_MILK_FROM

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


def _grasp_offset_from_the_tool_frame(
    world, view, milk, offset: float
) -> GraspCandidate:
    """
    :return: A grasp on the milk standing `offset` meters along the world's x-axis from
        where the left gripper's tool frame is now.
    """
    world_T_tool = view.left_arm.end_effector.tool_frame.global_transform
    world_P_grasp = world_T_tool.position.to_np()[:3] + [offset, 0.0, 0.0]
    return GraspCandidate(
        milk,
        world.transform(
            Pose.from_xyz_rpy(*world_P_grasp, reference_frame=world.root), milk.root
        ),
    )


@pytest.mark.parametrize("offset, reached", [(0.0, True), (0.1, False)])
def test_a_tool_frame_is_at_a_grasp_wherever_on_the_object_it_lies(
    pr2_apartment_context, offset, reached
):
    """
    The tool frame is compared with the grasp's position, not the object's origin.
    """
    world, view, context = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    grasp = _grasp_offset_from_the_tool_frame(world, view, milk, offset)

    assert ToolFrameIsAtGrasp(view.left_arm.end_effector, grasp)() is reached
