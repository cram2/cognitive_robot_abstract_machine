"""
The real GARMI driving its grippers through their Franka gripper driver, skipped where
``franka_msgs`` is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("franka_msgs")

from cramph.composites import Parallel, Sequence
from cramph.context import ContextExtension, StatechartContext
from coraplex.robot_plans.actions.core.robot_body import (
    MoveFingersToWidthAction,
    SetGripperAction,
)
from coraplex.garmi_franka_grippers import (
    CloseGarmiFingersToWidthWithTheirFrankaDriver,
    DriveGarmiGripperStatesWithTheirFrankaDriver,
    FrankaGripperActionName,
    RealGarmiPlanExecutor,
)
from giskardpy.middleware.ros2.franka_gripper_tasks import (
    GripperGraspTask,
    GripperMoveTask,
)
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.garmi import GarmiLeftGripper, GarmiRightGripper

from .test_robot_specific_plan_transformations import garmi_extensions, prepared
from ...plan_running import robot_executor, world_of

# %% helpers


def real_garmi_executor(extensions: list[ContextExtension]) -> RealGarmiPlanExecutor:
    """
    :param extensions: The context extensions of the plan, holding a
        :class:`~coraplex.plans.context_extensions.RobotAccess` of GARMI.
    :return: An executor sending plans to the real GARMI.
    """
    return RealGarmiPlanExecutor(
        StatechartContext.with_extensions(world_of(extensions), extensions)
    )


def grasp_parameters_of(task: GripperGraspTask) -> tuple[str, float, ...]:
    """
    :return: What `task` sends to the gripper driver, and where to.
    """
    return (
        task.action_topic,
        task.width,
        task.speed,
        task.force,
        task.inner_tolerance,
        task.outer_tolerance,
    )


def test_garmi_opens_its_gripper_through_the_franka_driver(garmi_extensions) -> None:
    gripper = world_of(garmi_extensions).get_semantic_annotations_by_type(
        GarmiLeftGripper
    )[0]
    transformation = DriveGarmiGripperStatesWithTheirFrankaDriver()

    plan = prepared(
        Sequence(
            nodes=[
                SetGripperAction(
                    configuration=gripper.default_configuration(GripperState.OPEN)
                )
            ]
        ),
        real_garmi_executor(garmi_extensions),
    )

    [task] = plan.nodes
    assert type(task) is GripperMoveTask
    assert (task.action_topic, task.width, task.speed) == (
        transformation.action_name(gripper, FrankaGripperActionName.MOVE),
        gripper.separation_when_open,
        transformation.finger_speed,
    )


def test_garmi_shuts_its_gripper_with_a_grasp_of_the_franka_driver_while_doing_more(
    garmi_extensions,
) -> None:
    gripper = world_of(garmi_extensions).get_semantic_annotations_by_type(
        GarmiLeftGripper
    )[0]
    expected = DriveGarmiGripperStatesWithTheirFrankaDriver().grasp(gripper)

    plan = prepared(
        Sequence(
            nodes=[
                Parallel(
                    nodes=[
                        MoveGripper(
                            gripper=gripper,
                            width=gripper.separation_in(GripperState.CLOSE),
                        )
                    ]
                )
            ]
        ),
        real_garmi_executor(garmi_extensions),
    )

    [task] = plan.nodes[0].nodes
    assert type(task) is GripperGraspTask
    assert grasp_parameters_of(task) == grasp_parameters_of(expected)


def test_the_right_gripper_of_garmi_is_driven_in_its_own_namespace(
    garmi_extensions,
) -> None:
    gripper = world_of(garmi_extensions).get_semantic_annotations_by_type(
        GarmiRightGripper
    )[0]
    transformation = DriveGarmiGripperStatesWithTheirFrankaDriver()

    assert transformation.action_name(gripper, FrankaGripperActionName.MOVE) == (
        f"{transformation.right_gripper_namespace}/{FrankaGripperActionName.MOVE}"
    )


def test_garmi_closes_its_fingers_to_a_width_with_a_grasp_of_the_franka_driver(
    garmi_extensions,
) -> None:
    gripper = world_of(garmi_extensions).get_semantic_annotations_by_type(
        GarmiLeftGripper
    )[0]
    expected = CloseGarmiFingersToWidthWithTheirFrankaDriver().grasp(gripper)

    plan = prepared(
        Sequence(
            nodes=[
                Parallel(nodes=[MoveFingersToWidthAction(gripper=gripper, width=0.03)])
            ]
        ),
        real_garmi_executor(garmi_extensions),
    )

    [task] = plan.nodes[0].nodes
    assert type(task) is GripperGraspTask
    assert grasp_parameters_of(task) == grasp_parameters_of(expected)


# %% the executor of the real GARMI


def test_the_real_garmi_adapts_plans_like_every_real_robot_and_drives_its_grippers(
    garmi_extensions,
) -> None:
    every_real_robot = robot_executor(garmi_extensions).plan_adaptations

    adaptations = real_garmi_executor(garmi_extensions).plan_adaptations

    assert [type(adaptation) for adaptation in adaptations] == [
        *(type(adaptation) for adaptation in every_real_robot),
        DriveGarmiGripperStatesWithTheirFrankaDriver,
        CloseGarmiFingersToWidthWithTheirFrankaDriver,
    ]
