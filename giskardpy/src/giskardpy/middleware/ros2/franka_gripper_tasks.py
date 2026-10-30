"""
Motion statechart nodes calling the action servers of a Franka gripper driver.

.. note:: Importing this module needs ``franka_msgs``.
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field

from franka_msgs.action import Grasp, Homing, Move
from franka_msgs.msg import GraspEpsilon
from rclpy.task import Future

from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    Action,
    ActionFeedback,
    ActionGoal,
    ActionResult,
    ActionServerTask,
)

logger = logging.getLogger(__name__)

# %% calling the gripper driver


@dataclass(eq=False, repr=False, kw_only=True)
class FrankaGripperTask(
    ActionServerTask[Action, ActionGoal, ActionResult, ActionFeedback], ABC
):
    """
    Calls an action server of a Franka gripper, whose results report ``success`` and
    ``error``.
    """

    def result_callback(self, future: Future) -> None:
        super().result_callback(future)
        if not self._result.result.success:
            logger.error(
                f"Gripper action {self.action_topic} failed: {self._result.result.error}"
            )

    def on_tick(self) -> ObservationStateValues:
        """
        :return: Whether the gripper reported success, unknown until it reports.
        """
        if self._result is None:
            return ObservationStateValues.UNKNOWN
        if self._result.result.success:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE


@dataclass(eq=False, repr=False, kw_only=True)
class GripperMoveTask(FrankaGripperTask[Move, Move.Goal, Move.Result, Move.Feedback]):
    """
    Moves the fingers of a Franka gripper to a width without checking for a grasped
    object.
    """

    message_type: type[Move] = field(default=Move, init=False)

    width: float
    """
    Distance between the fingers to move to, in meters.
    """

    speed: float
    """
    Speed of the fingers, in meters per second.
    """

    def build_action_goal(self) -> None:
        self._action_goal = Move.Goal(width=self.width, speed=self.speed)


@dataclass(eq=False, repr=False, kw_only=True)
class GripperGraspTask(
    FrankaGripperTask[Grasp, Grasp.Goal, Grasp.Result, Grasp.Feedback]
):
    """
    Closes a Franka gripper on an object and succeeds if the grasped width lies within
    the tolerance of :attr:`width`.
    """

    message_type: type[Grasp] = field(default=Grasp, init=False)

    width: float
    """
    Expected width of the grasped object, in meters.
    """

    speed: float
    """
    Speed of the fingers, in meters per second.
    """

    force: float
    """
    Grasping force, in newtons.
    """

    inner_tolerance: float
    """
    Largest accepted amount by which the grasped width falls below :attr:`width`, in
    meters.
    """

    outer_tolerance: float
    """
    Largest accepted amount by which the grasped width exceeds :attr:`width`, in meters.
    """

    def build_action_goal(self) -> None:
        self._action_goal = Grasp.Goal(
            width=self.width,
            epsilon=GraspEpsilon(
                inner=self.inner_tolerance, outer=self.outer_tolerance
            ),
            speed=self.speed,
            force=self.force,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class GripperHomingTask(
    FrankaGripperTask[Homing, Homing.Goal, Homing.Result, Homing.Feedback]
):
    """
    Homes a Franka gripper, which calibrates its maximum width.
    """

    message_type: type[Homing] = field(default=Homing, init=False)

    def build_action_goal(self) -> None:
        self._action_goal = Homing.Goal()
