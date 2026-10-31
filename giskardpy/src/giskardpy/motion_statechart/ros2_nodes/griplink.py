from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

from griplink_interfaces.action import Flexgrip, Flexrelease, Grip, Release
from typing_extensions import Self

from cramph.data_types import ObservationStateValues
from cramph.node import StatechartNode
from giskardpy.motion_statechart.exceptions import (
    GriplinkParameterOutOfRangeError,
    NoGriplinkEndpoint,
    UnknownGriplinkActionError,
)
from giskardpy.motion_statechart.goals.gripper import (
    MoveFingers,
    MovesGripperToConfiguration,
)
from krrood.ormatic.utils import classproperty
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.robots.griplink_gripper import (
    GriplinkEndpoint,
    GriplinkFlexConfiguration,
    GriplinkGripperConfiguration,
    GriplinkGripPreset,
    GriplinkPresetConfiguration,
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


@dataclass(eq=False, repr=False, kw_only=True)
class GriplinkActionServerTask(
    ActionServerTask[
        GriplinkAction,
        GriplinkGoal,
        GriplinkResultResponse,
        GriplinkFeedback,
    ],
    MovesGripperToConfiguration,
    ABC,
):
    """
    Base class for tasks calling a griplink action server.

    Observes the gripper status the server reports in its result; subclasses build the
    goal for their griplink actions.
    """

    @classproperty
    @abstractmethod
    def commanded_actions(cls) -> frozenset[type]:
        """
        The griplink actions this task builds goals for.
        """

    @classmethod
    def moving_gripper_to(
        cls,
        configuration: GripperConfiguration,
        stall_minimum_time: float | None = None,
        tolerate_stall: bool = False,
    ) -> StatechartNode:
        """
        Differs from :meth:`MovesGripperToConfiguration.moving_gripper_to` in driving a
        griplink gripper through the griplink action server serving the state of
        `configuration`; any other configuration is reached as
        :meth:`MoveFingers.moving_gripper_to` reaches it, which alone reads
        `stall_minimum_time` and `tolerate_stall`.

        :raises NoGriplinkEndpoint: If the gripper declares no endpoint for the state of
            a griplink configuration that executes it.
        """
        match configuration:
            case GriplinkPresetConfiguration():
                return GriplinkPresetActionServerTask.from_configuration(configuration)
            case GriplinkFlexConfiguration():
                return GriplinkFlexActionServerTask.from_configuration(configuration)
        return MoveFingers.moving_gripper_to(
            configuration, stall_minimum_time, tolerate_stall
        )

    @classmethod
    @abstractmethod
    def from_configuration(cls, configuration: GriplinkGripperConfiguration) -> Self:
        """
        :param configuration: The configuration to bring the gripper into.
        :return: A new task bringing the gripper into `configuration` through the
            endpoint serving its state.
        :raises NoGriplinkEndpoint: If the gripper declares no endpoint for that state
            that executes one of :attr:`commanded_actions`.
        """

    @classmethod
    def endpoint_serving(
        cls, configuration: GriplinkGripperConfiguration
    ) -> GriplinkEndpoint:
        """
        :param configuration: The configuration to bring the gripper into.
        :return: The endpoint the gripper of `configuration` declares for its state.
        :raises NoGriplinkEndpoint: If the gripper declares none, or one executing an
            action other than :attr:`commanded_actions`.
        """
        state_type = configuration.joint_state.state_type
        endpoint = configuration.end_effector.griplink_endpoint(state_type)
        if endpoint is None or endpoint.message_type not in cls.commanded_actions:
            raise NoGriplinkEndpoint(
                end_effector=configuration.end_effector, state_type=state_type
            )
        return endpoint

    def on_tick(self) -> ObservationStateValues:
        """
        Observes the gripper status the server reports once its result arrived.

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


@dataclass(eq=False, repr=False, kw_only=True)
class GriplinkPresetActionServerTask(GriplinkActionServerTask):
    """
    Node for calling the griplink action server of a griplink gripper to execute a
    stored grip preset (``Grip``) or open the gripper (``Release``).
    """

    grip_preset: GriplinkGripPreset = GriplinkGripPreset.PRESET_0
    """
    Grip preset the server executes.
    """

    @classproperty
    def commanded_actions(cls) -> frozenset[type]:
        return frozenset((Grip, Release))

    @classmethod
    def from_configuration(cls, configuration: GriplinkPresetConfiguration) -> Self:
        endpoint = cls.endpoint_serving(configuration)
        return cls(
            action_topic=endpoint.action_topic,
            message_type=endpoint.message_type,
            grip_preset=configuration.grip_preset,
        )

    def build_action_goal(self) -> None:
        """
        Builds the ``Grip`` or ``Release`` goal selecting the configured preset.

        :raises UnknownGriplinkActionError: If the task was built for neither ``Grip``
            nor ``Release``.
        """
        if self.message_type is Grip:
            goal_type = Grip.Goal
        elif self.message_type is Release:
            goal_type = Release.Goal
        else:
            raise UnknownGriplinkActionError(message_type=self.message_type)
        self._action_goal = goal_type(
            port=0,
            index=self.grip_preset.value,
        )


@dataclass(eq=False, repr=False, kw_only=True)
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

    @classproperty
    def commanded_actions(cls) -> frozenset[type]:
        return frozenset((Flexgrip, Flexrelease))

    @classmethod
    def from_configuration(cls, configuration: GriplinkFlexConfiguration) -> Self:
        endpoint = cls.endpoint_serving(configuration)
        return cls(
            action_topic=endpoint.action_topic,
            message_type=endpoint.message_type,
            grip_position=configuration.grip_position,
            grip_force=configuration.grip_force,
            grip_velocity=configuration.grip_velocity,
            grip_acceleration=configuration.grip_acceleration,
        )

    def __post_init__(self) -> None:
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

    def build_action_goal(self) -> None:
        """
        Builds the ``Flexgrip`` or ``Flexrelease`` goal from the task's parameters.

        :raises UnknownGriplinkActionError: If the task was built for neither
            ``Flexgrip`` nor ``Flexrelease``.
        """
        if self.message_type is Flexgrip:
            self._action_goal = Flexgrip.Goal(
                port=0,
                position=self._to_micrometres(self.grip_position),
                force=self._to_millinewtons(self.grip_force),
                speed=self._to_micrometres(self.grip_velocity),
                acceleration=self._to_micrometres(self.grip_acceleration),
            )
        elif self.message_type is Flexrelease:
            self._action_goal = Flexrelease.Goal(
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
