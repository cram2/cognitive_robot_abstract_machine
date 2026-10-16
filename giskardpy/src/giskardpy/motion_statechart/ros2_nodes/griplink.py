from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release
from semantic_digital_twin.robots.griplink_gripper import GriplinkGripPreset

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import UnknownGriplinkActionError
from giskardpy.motion_statechart.ros2_nodes.ros_tasks import ActionServerTask

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


# %% griplink message protocols


class GriplinkGoal(Protocol):
    """
    A griplink action goal; every griplink action addresses one port of the controller.
    """

    port: int


class GriplinkResult(Protocol):
    """
    The result every griplink action reports: a status code, a human-readable message
    and the controller's device state.
    """

    status: int
    message: str
    device_state: int


class GriplinkFeedback(Protocol):
    """
    The feedback every griplink action streams: the controller's device state.
    """

    device_state: int


class GriplinkResultResponse(Protocol):
    """
    The response the action client resolves a result future with, wrapping the
    :class:`GriplinkResult` the server reported and the goal state it reached.
    """

    status: int
    result: GriplinkResult


class GriplinkAction(Protocol):
    """
    A griplink action, carrying the goal, result and feedback message classes the action
    client sends and receives.
    """

    Goal: type[GriplinkGoal]
    Result: type[GriplinkResult]
    Feedback: type[GriplinkFeedback]


# %% griplink action server tasks


@dataclass(eq=False, repr=False)
class GriplinkActionServerTask(
    ActionServerTask[
        GriplinkAction,
        GriplinkGoal,
        GriplinkResultResponse,
        GriplinkFeedback,
    ]
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
class GriplinkPresetActionServerTask(GriplinkActionServerTask):
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
        :raises UnknownGriplinkActionError: If the task was built for neither ``Grip``
            nor ``Release``.
        """
        if self.message_type is Grip:
            goal_type = Grip.Goal
        elif self.message_type is Release:
            goal_type = Release.Goal
        else:
            raise UnknownGriplinkActionError(message_type=self.message_type)
        self._msg = goal_type(
            port=0,
            index=self.grip_preset.value,
        )


@dataclass(eq=False, repr=False)
class GriplinkFlexActionServerTask(GriplinkActionServerTask):
    """
    Node for calling the griplink action server of a griplink gripper to flex grip to a
    commanded opening width (``Flexgrip``) or flex release from it (``Flexrelease``).
    """

    grip_position: float | None = None
    """
    Opening width of the gripper in mm [-5..120], rounded to one decimal.

    Converted to µm when building the goal message.
    """

    grip_force: float | None = None
    """
    Force the gripper applies to the object in N [30..300], rounded to one decimal.

    Converted to mN when building the Flexgrip goal message; ``Flexrelease`` has no
    force goal.
    """

    grip_velocity: float | None = None
    """
    Motion velocity of the gripper in mm/s [5..350], rounded to one decimal.

    Converted to µm/s when building the goal message.
    """

    grip_acceleration: float | None = None
    """
    Motion acceleration of the gripper in mm/s² [100..4000], rounded to one decimal.

    Converted to µm/s² when building the goal message.
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Builds the ``Flexgrip`` or ``Flexrelease`` goal, filling unset parameters with
        the defaults of the commanded action.

        :param context: The motion statechart context the task runs in.
        :return: None; the goal is stored for :meth:`on_start` to send.
        :raises UnknownGriplinkActionError: If the task was built for neither
            ``Flexgrip`` nor ``Flexrelease``.
        """
        if self.message_type is Flexgrip:
            position = 0 if self.grip_position is None else self.grip_position
            force = 90 if self.grip_force is None else self.grip_force
            velocity = 150 if self.grip_velocity is None else self.grip_velocity
            acceleration = (
                600 if self.grip_acceleration is None else self.grip_acceleration
            )
            self._msg = Flexgrip.Goal(
                port=0,
                position=self._to_micrometres(position),
                force=self._to_micrometres(force),
                speed=self._to_micrometres(velocity),
                acceleration=self._to_micrometres(acceleration),
            )
        elif self.message_type is Flexrelease:
            position = 120 if self.grip_position is None else self.grip_position
            velocity = 250 if self.grip_velocity is None else self.grip_velocity
            acceleration = (
                2000 if self.grip_acceleration is None else self.grip_acceleration
            )
            self._msg = Flexrelease.Goal(
                port=0,
                position=self._to_micrometres(position),
                speed=self._to_micrometres(velocity),
                acceleration=self._to_micrometres(acceleration),
            )
        else:
            raise UnknownGriplinkActionError(message_type=self.message_type)

    @staticmethod
    def _to_micrometres(value: float) -> int:
        """
        :param value: A gripper parameter in millimetres, newtons or the per-second
            derivations of them, rounded to one decimal.
        :return: The value the griplink controller expects, in the matching µm-scaled
            integer unit.
        """
        return round(value * 1000)
