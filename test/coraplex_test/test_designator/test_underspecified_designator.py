from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
    ProbabilisticBackend,
)
from krrood.entity_query_language.factories import a, an, variable_from
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    TaskStatus,
)
from coraplex.datastructures.grasp import GraspDescription

from coraplex.language import SequentialNode
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential, execute_single
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Pose


def test_underspecified_action(apartment_world_pr2_copy_with_context):
    """
    Test that an underspecified action resolves to a concrete candidate and parses into
    an executable.

    Execution is deferred to parse().execute(), so performing the node only expands it;
    the resolved candidate is not performed here.
    """
    world, robot, context = apartment_world_pr2_copy_with_context
    action = a(NavigateAction)(
        target_location=variable_from(
            [
                Pose.from_xyz_quaternion(1, -1, 0, reference_frame=world.root),
                Pose.from_xyz_quaternion(2, -1, 0, reference_frame=world.root),
            ]
        ),
        keep_joint_states=True,
    )

    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

    assert plan.root.status == TaskStatus.SUCCEEDED
    candidate = plan.root.children[0]
    assert isinstance(candidate.designator, NavigateAction)
    assert plan.root.parse() is not None
    assert plan.root._action_iterator is None, (
        "the action iterator must be released once grounding succeeds, so any resources a "
        "candidate generator only holds to validate against (for example a location's "
        "deep-copied test world) are not retained for the node's whole lifetime"
    )


def test_underspecified_action_with_ellipsis(apartment_world_pr2_copy_with_context):
    """
    Test that an underspecified action resolves and parses when a factory for a spatial
    type is used with ellipsis.

    Execution is deferred to parse().execute(), so performing the node only expands it;
    the resolved candidate is not performed here.
    """
    world, robot, context = apartment_world_pr2_copy_with_context
    context.query_backend = ProbabilisticBackend()
    action = a(NavigateAction)(
        target_location=a(Pose.from_xyz_rpy)(
            x=...,
            y=...,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            reference_frame=context.robot.root,
        ),
        keep_joint_states=...,
    )

    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

    assert plan.root.status == TaskStatus.SUCCEEDED
    candidate = plan.root.children[-1]
    assert isinstance(candidate.designator, NavigateAction)
    assert plan.root.parse() is not None


def test_underspecified_language(apartment_world_pr2_copy_with_context):
    """
    Test that entire plans can be underspecified.
    """
    world, robot, context = apartment_world_pr2_copy_with_context
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        robot.left_arm.end_effector,
    )
    plan_generator = an(sequential, target_type=SequentialNode)(
        children=[
            a(NavigateAction)(
                target_location=(
                    target_locations := variable_from(
                        [
                            Pose.from_xyz_quaternion(
                                1, 0, 0, reference_frame=world.root
                            ),
                            Pose.from_xyz_quaternion(
                                2, 0, 0, reference_frame=world.root
                            ),
                        ]
                    )
                ),
                keep_joint_states=True,
            ),
            a(PickUpAction)(
                arm=...,
                grasp_description=grasp_description,
                object_designator=world.get_body_by_name("milk.stl"),
            ),
        ],
        context=context,
    )
    plan_generator.resolve()
    plans = list(EntityQueryLanguageGenerativeBackend().evaluate(plan_generator))
    assert len(plans) == len(list(target_locations._domain_)) * len(list(Arms))
