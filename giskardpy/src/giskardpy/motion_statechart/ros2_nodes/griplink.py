from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic

from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import (
    Action,
    ActionFeedback,
    ActionGoal,
    ActionResult,
    ActionServerTask,
)
from semantic_digital_twin.datastructures.robots.gripper_configurations import (
    GriplinkGripPreset,
)

logger = logging.getLogger(__name__)


class GriplinkStatus(IntEnum):
    """
    The status values a griplink action server reports in its result.
    """

    SUCCESS = 0
    OVERRUN = 1
    RANGE_ERROR = 2
    NOT_AVAILABLE = 3
    NOT_INITIALIZED = 4
    TIMEOUT = 5
    INSUFFICIENT_RESOURCES = 6
    CHECKSUM_ERROR = 7
    ACCESS_DENIED = 8
    INVALID_HANDLE = 9
    INVALID_PARAMETER = 10
    INDEX_OUT_OF_BOUNDS = 11
    IO_ERROR = 12
    READ_ERROR = 13
    WRITE_ERROR = 14
    NOT_FOUND = 15
    NOT_OPEN = 16
    EXISTS = 17
    NO_COMM = 18
    STATE_CONFLICT = 19
    NOT_SUPPORTED = 20
    INCONSISTENT_DATA = 21
    CMD_SYNTAX = 22
    CMD_UNKNOWN = 23
    CMD_ABORTED = 24
    CMD_FAILED = 25
    AXIS_BLOCKED = 26
    PENDING = 27


# %% griplink action server tasks


@dataclass(eq=False, repr=False)
class GriplinkActionServerTask(
    ActionServerTask[Action, ActionGoal, ActionResult, ActionFeedback],
    Generic[Action, ActionGoal, ActionResult, ActionFeedback],
):
    """
    Base class for tasks calling a griplink action server.

    Observes the gripper status the server reports in its result; subclasses build the
    goal for their griplink actions.
    """

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        """
        Observes the gripper status the server reports once its result arrived.

        :param context: The motion statechart context the task runs in.
        :return:``TRUE`` when the server reported success, ``FALSE`` otherwise, and
            ``UNKNOWN`` while no result has arrived yet.
        """
        if self._result:
            gripper_status = self._result.result.status
            logger.info(f"Gripper status: {GriplinkStatus(gripper_status)}")
            return (
                ObservationStateValues.TRUE
                if gripper_status == GriplinkStatus.SUCCESS
                else ObservationStateValues.FALSE
            )
        return ObservationStateValues.UNKNOWN


@dataclass(eq=False, repr=False)
class GriplinkPresetActionServerTask(
    GriplinkActionServerTask[
        Grip | Release,
        Grip.Goal | Release.Goal,
        Grip.Result | Release.Result,
        Grip.Feedback | Release.Feedback,
    ]
):
    """
    Node for calling the griplink action server of a griplink gripper to execute a
    stored grip preset (``Grip``) or open the gripper (``Release``).
    """

    grip_preset: GriplinkGripPreset = GriplinkGripPreset.PRESET_0
    """
    Grip preset the server executes.
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Builds the ``Grip`` or ``Release`` goal selecting the configured preset.

        :param context: The motion statechart context the task runs in.
        :return: None; the goal is stored for :meth:`on_start` to send.
        :raises ValueError: If the task was built for neither ``Grip`` nor ``Release``.
        """
        if self.message_type is Grip:
            goal_type = Grip.Goal
        elif self.message_type is Release:
            goal_type = Release.Goal
        else:
            raise ValueError(f"Unknown message type: {self.message_type}")
        self._msg = goal_type(
            port=0,
            index=self.grip_preset.value,
        )


@dataclass(eq=False, repr=False)
class GriplinkFlexActionServerTask(
    GriplinkActionServerTask[
        Flexgrip | Flexrelease,
        Flexgrip.Goal | Flexrelease.Goal,
        Flexgrip.Result | Flexrelease.Result,
        Flexgrip.Feedback | Flexrelease.Feedback,
    ]
):
    """
    Node for calling the griplink action server of a griplink gripper to flex grip to a
    commanded opening width (``Flexgrip``) or flex release from it (``Flexrelease``).
    """

    grip_position: int | None = None
    """
    Opening width of the gripper in mm [-5..120].

    Converted to µm when building the goal message.
    """

    grip_force: int | None = None
    """
    Force the gripper applies to the object in N [30..300].

    Converted to mN when building the Flexgrip goal message; ``Flexrelease`` has no
    force goal.
    """

    grip_speed: int | None = None
    """
    Motion speed of the gripper in mm/s [5..350].

    Converted to µm/s when building the goal message.
    """

    grip_acceleration: int | None = None
    """
    Motion acceleration of the gripper in mm/s² [100..4000].

    Converted to µm/s² when building the goal message.
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Builds the ``Flexgrip`` or ``Flexrelease`` goal, filling unset parameters with
        the defaults of the commanded action.

        :param context: The motion statechart context the task runs in.
        :return: None; the goal is stored for :meth:`on_start` to send.
        :raises ValueError: If the task was built for neither ``Flexgrip`` nor
            ``Flexrelease``.
        """
        if self.message_type is Flexgrip:
            position = 0 if self.grip_position is None else self.grip_position
            force = 90 if self.grip_force is None else self.grip_force
            speed = 150 if self.grip_speed is None else self.grip_speed
            acceleration = (
                600 if self.grip_acceleration is None else self.grip_acceleration
            )
            self._msg = Flexgrip.Goal(
                port=0,
                position=position * 1000,
                force=force * 1000,
                speed=speed * 1000,
                acceleration=acceleration * 1000,
            )
        elif self.message_type is Flexrelease:
            position = 120 if self.grip_position is None else self.grip_position
            speed = 250 if self.grip_speed is None else self.grip_speed
            acceleration = (
                2000 if self.grip_acceleration is None else self.grip_acceleration
            )
            self._msg = Flexrelease.Goal(
                port=0,
                position=position * 1000,
                speed=speed * 1000,
                acceleration=acceleration * 1000,
            )
        else:
            raise ValueError(f"Unknown message type: {self.message_type}")
