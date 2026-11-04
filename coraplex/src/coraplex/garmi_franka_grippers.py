"""
Driving the grippers of the real GARMI through their Franka gripper driver.

.. note:: Importing this module needs ``franka_msgs``.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Generic

from coraplex.plans.executors import RealRobotPlanExecutor
from coraplex.plans.plan_transformation import MatchedType, PlanTransformation
from coraplex.robot_plans.actions.core.robot_body import MoveFingersToWidthAction
from cramph.composites import ControlStructureNode
from giskardpy.middleware.ros2.franka_gripper_tasks import (
    GripperGraspTask,
    GripperMoveTask,
)
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.garmi import GarmiLeftGripper, GarmiRightGripper

# %% moving GARMI's fingers through the Franka gripper driver


class FrankaGripperActionName(StrEnum):
    """
    The names the Franka gripper driver serves its actions under, inside its namespace.
    """

    MOVE = "move"
    GRASP = "grasp"


@dataclass
class DriveGarmiGrippersWithTheirFrankaDriver(
    PlanTransformation[MatchedType], Generic[MatchedType], SubClassSafeGeneric, ABC
):
    """
    Moves the fingers of GARMI's real grippers through the actions of their Franka
    gripper driver instead of with a Giskard goal.
    """

    left_gripper_namespace: str = "/garmi/arms/left/franka_gripper"
    """
    The ROS namespace the driver of the left gripper serves its actions in.
    """

    right_gripper_namespace: str = "/garmi/arms/right/franka_gripper"
    """
    The ROS namespace the driver of the right gripper serves its actions in.
    """

    finger_speed: float = 0.1
    """
    The speed the fingers move at, in metres per second.
    """

    grasp_force: float = 0.0
    """
    The force the fingers hold an object with, in newtons.
    """

    grasp_width_tolerance: float = 0.1
    """
    How far the fingers may stop from closed for a grasp to succeed, in metres.

    The driver aborts a grasp whose fingers stop beyond it.
    """

    def action_name(
        self,
        gripper: GarmiLeftGripper | GarmiRightGripper,
        action: FrankaGripperActionName,
    ) -> str:
        """
        :return: The name the driver of `gripper` serves `action` under.
        """
        if isinstance(gripper, GarmiLeftGripper):
            return f"{self.left_gripper_namespace}/{action}"
        return f"{self.right_gripper_namespace}/{action}"

    def grasp(self, gripper: GarmiLeftGripper | GarmiRightGripper) -> GripperGraspTask:
        """
        :return: A new task shutting `gripper` on whatever lies between its fingers,
            which succeeds where the fingers stop within :attr:`grasp_width_tolerance`.
        """
        return GripperGraspTask(
            action_topic=self.action_name(gripper, FrankaGripperActionName.GRASP),
            width=0.0,
            speed=self.finger_speed,
            force=self.grasp_force,
            inner_tolerance=self.grasp_width_tolerance,
            outer_tolerance=self.grasp_width_tolerance,
        )


@dataclass
class DriveGarmiGripperStatesWithTheirFrankaDriver(
    DriveGarmiGrippersWithTheirFrankaDriver[MoveGripper]
):
    """
    Moves the fingers of a gripper of GARMI to a width, or shuts them on whatever lies
    between them where they are sent to stand closed, through its Franka gripper driver.
    """

    def is_applicable(self, plan_node: MoveGripper) -> bool:
        return isinstance(plan_node.gripper, (GarmiLeftGripper, GarmiRightGripper))

    def apply(self, plan_node: MoveGripper) -> None:
        """
        Puts the task in place of the step that drives the fingers: `plan_node`, or the
        action whose goal it is.
        """
        gripper = plan_node.gripper
        if plan_node.width <= gripper.separation_in(GripperState.CLOSE):
            task = self.grasp(gripper)
        else:
            task = GripperMoveTask(
                action_topic=self.action_name(gripper, FrankaGripperActionName.MOVE),
                width=plan_node.width,
                speed=self.finger_speed,
            )
        plan_node.statechart.replace(ControlStructureNode.step_holding(plan_node), task)


@dataclass
class CloseGarmiFingersToWidthWithTheirFrankaDriver(
    DriveGarmiGrippersWithTheirFrankaDriver[MoveFingersToWidthAction]
):
    """
    Shuts a gripper of GARMI on the object it is to meet at its width through its Franka
    gripper driver, whose grasp stops the fingers where the object pushes back.
    """

    def is_applicable(self, plan_node: MoveFingersToWidthAction) -> bool:
        return isinstance(plan_node.gripper, (GarmiLeftGripper, GarmiRightGripper))

    def apply(self, plan_node: MoveFingersToWidthAction) -> None:
        plan_node.statechart.replace(plan_node, self.grasp(plan_node.gripper))


# %% running plans on the real GARMI


@dataclass
class RealGarmiPlanExecutor(RealRobotPlanExecutor):
    """
    Runs plans on the real GARMI, whose grippers move through their Franka gripper
    driver: :attr:`plan_adaptations` is extended by
    :class:`DriveGarmiGripperStatesWithTheirFrankaDriver` and
    :class:`CloseGarmiFingersToWidthWithTheirFrankaDriver`.
    """

    def __post_init__(self) -> None:
        self.plan_adaptations = [
            *self.plan_adaptations,
            DriveGarmiGripperStatesWithTheirFrankaDriver(),
            CloseGarmiFingersToWidthWithTheirFrankaDriver(),
        ]
        super().__post_init__()
