from krrood.entity_query_language.backends import (
    EntityQueryLanguageBackend,
    ProbabilisticBackend,
)
from krrood.entity_query_language.factories import underspecified, variable_from
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    TaskStatus,
)
from pycram.datastructures.grasp import GraspDescription

from pycram.language import SequentialNode
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential, execute_single
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Pose


def test_underspecified_action(mutable_model_world):
    """
    Test that an underspecified action can be executed
    """
    world, robot, context = mutable_model_world
    action = underspecified(NavigateAction)(
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
    assert len(plan.nodes) == 3
    assert plan.root.status == TaskStatus.SUCCEEDED
    assert plan.root.children[0].status == TaskStatus.SUCCEEDED


def test_underspecified_action_with_ellipsis(mutable_model_world):
    """
    Test that an underspecified action can be executed when a factory for a spatial type is used with ellipsis
    """
    world, robot, context = mutable_model_world
    context.query_backend = ProbabilisticBackend()
    action = underspecified(NavigateAction)(
        target_location=underspecified(Pose.from_xyz_rpy)(
            x=...,
            y=...,
            z=0,
            roll=0,
            pitch=0,
            yaw=0,
            reference_frame=context.robot.root,
        ),
        keep_joint_states=...,
    )

    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()
    assert len(plan.nodes) >= 3
    assert plan.root.status == TaskStatus.SUCCEEDED
    assert plan.root.children[-1].status == TaskStatus.SUCCEEDED


def test_underspecified_language(mutable_model_world):
    """
    Test that entire plans can be underspecified
    """
    world, robot, context = mutable_model_world
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        robot.left_arm.manipulator,
    )
    plan_generator = underspecified(sequential, target_type=SequentialNode)(
        children=[
            underspecified(NavigateAction)(
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
            underspecified(PickUpAction)(
                arm=...,
                grasp_description=grasp_description,
                object_designator=world.get_body_by_name("milk.stl"),
            ),
        ],
        context=context,
    )
    plan_generator.resolve()
    plans = list(EntityQueryLanguageBackend().evaluate(plan_generator))
    assert len(plans) == len(list(target_locations._domain_)) * len(list(Arms))
