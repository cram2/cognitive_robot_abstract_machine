from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from enum import IntEnum

from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release
from semantic_digital_twin.robots.griplink_gripper import GriplinkGripPreset

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.exceptions import (
    GriplinkParameterOutOfRangeError,
    UnknownGriplinkActionError,
)
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


# %% griplink action server tasks


# %% griplink message types


@dataclass
class GriplinkGoal:
    """
    A griplink action goal; every griplink action addresses one port of the controller.
    """

    port: int
    """
    The port of the controller the goal addresses.
    """


@dataclass
class GriplinkResult:
    """
    The result every griplink action reports: a status code, a human-readable message
    and the controller's device state.
    """

    status: int
    """
    The status code the action ended with, a :class:`GriplinkStatus` value.
    """

    message: str
    """
    Human-readable description of the status.
    """

    device_state: int
    """
    The state of the controller's device.
    """


@dataclass
class GriplinkFeedback:
    """
    The feedback every griplink action streams: the controller's device state.
    """

    device_state: int
    """
    The state of the controller's device.
    """


@dataclass
class GriplinkResultResponse:
    """
    The response the action client resolves a result future with, wrapping the
    :class:`GriplinkResult` the server reported and the goal state it reached.
    """

    status: int
    """
    The goal state the action reached.
    """

    result: GriplinkResult
    """
    The result the server reported.
    """


@dataclass
class GriplinkAction:
    """
    A griplink action, carrying the goal, result and feedback message classes the action
    client sends and receives.
    """

    Goal: type[GriplinkGoal]
    """
    The goal message class of the action.
    """

    Result: type[GriplinkResult]
    """
    The result message class of the action.
    """

    Feedback: type[GriplinkFeedback]
    """
    The feedback message class of the action.
    """


# %% griplink action server tasks


@dataclass(eq=False, repr=False)
class GriplinkActionServerTask(
    ActionServerTask[
        GriplinkAction,
        GriplinkGoal,
        GriplinkResultResponse,
        GriplinkFeedback,
    ],
    ABC,
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
    Opening width of the gripper in m [-0.005..0.12].

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
    Motion velocity of the gripper in m/s [0.005..0.35].

    Converted to µm/s when building the goal message.
    """

    grip_acceleration: float | None = None
    """
    Motion acceleration of the gripper in m/s² [0.1..4].

    Converted to µm/s² when building the goal message.
    """

    def __post_init__(self):
        """
        Fills the parameters left unset with the defaults of the commanded action.

        :raises UnknownGriplinkActionError: If the task was built for neither
            ``Flexgrip`` nor ``Flexrelease``.
        :raises GriplinkParameterOutOfRangeError: If a parameter lies outside its
            documented interval.
        """
        super().__post_init__()
        if self.message_type is Flexgrip:
            default_position, default_velocity, default_acceleration = 0.0, 0.15, 0.6
        elif self.message_type is Flexrelease:
            default_position, default_velocity, default_acceleration = 0.12, 0.25, 2.0
        else:
            raise UnknownGriplinkActionError(message_type=self.message_type)
        if self.grip_position is None:
            self.grip_position = default_position
        if self.grip_force is None:
            self.grip_force = 90.0
        if self.grip_velocity is None:
            self.grip_velocity = default_velocity
        if self.grip_acceleration is None:
            self.grip_acceleration = default_acceleration
        self._check_within(self.grip_position, "grip_position", -0.005, 0.12)
        self._check_within(self.grip_force, "grip_force", 30.0, 300.0)
        self._check_within(self.grip_velocity, "grip_velocity", 0.005, 0.35)
        self._check_within(self.grip_acceleration, "grip_acceleration", 0.1, 4.0)

    @staticmethod
    def _check_within(
        value: float, parameter_name: str, lower_bound: float, upper_bound: float
    ) -> None:
        """
        :param value: The parameter value to check.
        :param parameter_name: The name of the parameter, for the error.
        :param lower_bound: The smallest accepted value.
        :param upper_bound: The largest accepted value.
        :raises GriplinkParameterOutOfRangeError: If the value is outside the interval.
        """
        if not lower_bound <= value <= upper_bound:
            raise GriplinkParameterOutOfRangeError(
                parameter_name=parameter_name,
                value=value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

    def build_msg(self, context: MotionStatechartContext):
        """
        Builds the ``Flexgrip`` or ``Flexrelease`` goal from the task's parameters.

        :param context: The motion statechart context the task runs in.
        :return: None; the goal is stored for :meth:`on_start` to send.
        :raises UnknownGriplinkActionError: If the task was built for neither
            ``Flexgrip`` nor ``Flexrelease``.
        """
        if self.message_type is Flexgrip:
            self._msg = Flexgrip.Goal(
                port=0,
                position=self._to_micrometres(self.grip_position),
                force=self._to_millinewtons(self.grip_force),
                speed=self._to_micrometres(self.grip_velocity),
                acceleration=self._to_micrometres(self.grip_acceleration),
            )
        elif self.message_type is Flexrelease:
            self._msg = Flexrelease.Goal(
                port=0,
                position=self._to_micrometres(self.grip_position),
                speed=self._to_micrometres(self.grip_velocity),
                acceleration=self._to_micrometres(self.grip_acceleration),
            )
        else:
            raise UnknownGriplinkActionError(message_type=self.message_type)

    @staticmethod
    def _to_micrometres(value: float) -> int:
        """
        :param value: A length in metres or its per-second derivations.
        :return: The value the griplink controller expects, in the matching µm-scaled
            integer unit.
        """
        return round(value * 1_000_000)

    @staticmethod
    def _to_millinewtons(value: float) -> int:
        """
        :param value: A force in newtons.
        :return: The force the griplink controller expects, in integer millinewtons.
        """
        return round(value * 1000)
