from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Self

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    GripperConfiguration,
)
from semantic_digital_twin.robots.robot_parts import EndEffector

# %% griplink action server endpoints


@dataclass(frozen=True)
class GriplinkEndpoint:
    """
    A griplink action server endpoint a single gripper is reached on.
    """

    action_topic: str
    """
    ROS action topic the griplink server for one gripper listens on.
    """

    message_type: type
    """
    The griplink ROS action this endpoint's server executes (``Grip``/``Release``/
    ``Flexgrip``/``Flexrelease``).
    """


# %% grip presets


class GriplinkGripPreset(Enum):
    """
    Grip preset selectable on a WEISS WPG gripper controller.
    """

    PRESET_0 = 0
    PRESET_1 = 1
    PRESET_2 = 2
    PRESET_3 = 3
    PRESET_4 = 4
    PRESET_5 = 5
    PRESET_6 = 6
    PRESET_7 = 7


# %% griplink configurations


@dataclass(eq=False)
class GriplinkGripperConfiguration(GripperConfiguration, ABC):
    """
    A configuration for a griplink gripper, carrying the hardware parameters the
    griplink action server executes the motion with.
    """


@dataclass(eq=False)
class GriplinkPresetConfiguration(GriplinkGripperConfiguration):
    """
    A griplink gripper motion driven by a stored grip preset, used for ``Grip``/
    ``Release`` actions.
    """

    grip_preset: GriplinkGripPreset = GriplinkGripPreset.PRESET_0
    """
    Stored grip preset selected on the controller, used by ``Grip``/``Release``.
    """

    @classmethod
    def from_state_type(
        cls,
        end_effector: EndEffector,
        state_type: GripperState,
        grip_preset: GriplinkGripPreset = GriplinkGripPreset.PRESET_0,
    ) -> Self:
        """
        :param end_effector: The griplink gripper to build the configuration for.
        :param state_type: The state type the configuration is labelled with.
        :param grip_preset: Stored grip preset selected on the controller.
        :return: The configuration for the given state type, commanding the joint state
            the robot description declares for it.
        """
        return cls(
            end_effector=end_effector,
            joint_state=end_effector.get_joint_state_by_type(state_type),
            grip_preset=grip_preset,
        )


@dataclass(eq=False)
class GriplinkFlexConfiguration(GriplinkGripperConfiguration):
    """
    A griplink gripper motion driven by a commanded opening width, used for
    ``Flexgrip``/``Flexrelease`` actions.
    """

    maximum_opening_width_mm: ClassVar[float] = 120.0
    """
    Maximum opening width of the gripper in millimetres, the scale the flex
    configuration interpolates ``grip_position`` on.
    """

    fully_closed_opening_width_mm: ClassVar[float] = 0.0
    """
    Opening width of the gripper in millimetres that commands the fully closed state.
    """

    grip_position: float | None = None
    """
    Opening width of the gripper in millimetres [-5..120], used by ``Flexgrip``/
    ``Flexrelease``.

    ``None`` commands the declared open or close state, depending on the state type.
    """

    grip_force: float | None = None
    """
    Force the gripper applies to the object in newtons [30..300], used by ``Flexgrip``
    only (ignored for ``Flexrelease``, which has no force goal).

    ``None`` defers to the default the griplink action server picks (90).
    """

    grip_velocity: float | None = None
    """
    Motion velocity of the gripper in millimetres per second [5..350], used by
    ``Flexgrip``/``Flexrelease``.

    ``None`` defers to the per-motion default the griplink action server picks (150 for
    ``Flexgrip``, 250 for ``Flexrelease``).
    """

    grip_acceleration: float | None = None
    """
    Motion acceleration of the gripper in millimetres per second squared [100..4000],
    used by ``Flexgrip``/``Flexrelease``.

    ``None`` defers to the per-motion default the griplink action server picks (600 for
    ``Flexgrip``, 2000 for ``Flexrelease``).
    """

    @classmethod
    def from_state_type(
        cls,
        end_effector: EndEffector,
        state_type: GripperState,
        grip_position: float | None = None,
        grip_force: float | None = None,
        grip_velocity: float | None = None,
        grip_acceleration: float | None = None,
    ) -> Self:
        """
        The hardware parameters are rounded to one decimal, the precision the griplink
        controller works with.

        :param end_effector: The griplink gripper to build the configuration for.
        :param state_type: The state type the configuration is labelled with.
        :param grip_position: Opening width in millimetres; ``None`` commands the
            declared open or close state.
        :param grip_force: Force in newtons; ``None`` defers to the action server
            default.
        :param grip_velocity: Velocity in millimetres per second; ``None`` defers to the
            action server default.
        :param grip_acceleration: Acceleration in millimetres per second squared;
            ``None`` defers to the action server default.
        :return: The configuration for the given state type, commanding the joint state
            the parameters describe.
        """
        grip_position = cls._rounded_to_one_decimal(grip_position)
        grip_force = cls._rounded_to_one_decimal(grip_force)
        grip_velocity = cls._rounded_to_one_decimal(grip_velocity)
        grip_acceleration = cls._rounded_to_one_decimal(grip_acceleration)
        return cls(
            end_effector=end_effector,
            joint_state=cls._build_joint_state(end_effector, state_type, grip_position),
            grip_position=grip_position,
            grip_force=grip_force,
            grip_velocity=grip_velocity,
            grip_acceleration=grip_acceleration,
        )

    @staticmethod
    def _rounded_to_one_decimal(value: float | None) -> float | None:
        """
        :param value: A gripper hardware parameter in its own unit.
        :return: The value rounded to one decimal, or ``None`` if it was ``None``.
        """
        if value is None:
            return None
        return round(value, 1)

    @classmethod
    def _build_joint_state(
        cls,
        end_effector: EndEffector,
        state_type: GripperState,
        grip_position: float | None,
    ) -> JointState:
        """
        Build the joint state a griplink flex motion commands, interpolating between the
        declared open and close states by the configured opening width.

        Without a configured opening width, a ``FLEXCLOSE`` motion commands the declared
        close state and a ``FLEXOPEN`` motion the declared open state.

        :param end_effector: The griplink gripper whose connections are commanded.
        :param state_type: The flex state type the joint state is labelled with.
        :param grip_position: The opening width in millimetres driving the joint state.
        :return: The joint state driving the gripper's connections to the interpolated
            position.
        """
        if grip_position is None:
            grip_position = (
                cls.fully_closed_opening_width_mm
                if state_type is GripperState.FLEXCLOSE
                else cls.maximum_opening_width_mm
            )
        open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)
        close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)
        close_targets = dict(close_state.items())
        fraction = 1 - grip_position / cls.maximum_opening_width_mm
        target_values = [
            open_target + fraction * (close_targets[connection] - open_target)
            for connection, open_target in open_state.items()
        ]
        return JointState(
            connections=open_state.connections,
            target_values=target_values,
            state_type=state_type,
            name=PrefixedName("flexgrip", prefix=end_effector.name.name),
        )


# %% griplink end effector


@dataclass(eq=False)
class GriplinkGripper(EndEffector, ABC):
    """
    An `WEISS WPG gripper <https://weiss-robotics.com/servo-electric/wpg-series/>`_ driven by the Griplink interface.

    Builds griplink-specific configurations so generic actions and demos produce robot-
    appropriate configurations without naming the robot.
    """

    griplink_endpoints: ClassVar[Mapping[GripperState, GriplinkEndpoint]]
    """
    Griplink action server endpoints of this gripper, by the state each endpoint's
    action commands; a gripper declares the table for the states it serves.
    """

    def default_configuration(
        self,
        state_type: GripperState,
        finger_velocity: float | None = None,
    ) -> GripperConfiguration:
        """
        Build the griplink configuration for a state this gripper declares.

        Preset states (``OPEN``/``CLOSE``) build a
        :class:`~semantic_digital_twin.robots.griplink_gripper.GriplinkPresetConfiguration`,
        flex states (``FLEXOPEN``/``FLEXCLOSE``) a
        :class:`~semantic_digital_twin.robots.griplink_gripper.GriplinkFlexConfiguration`.

        :param state_type: The state type to build the configuration for.
        :param finger_velocity: Optional maximum finger joint velocity (in m/s) to
            enforce during the motion.
        :return: The griplink configuration for that state type, or the configuration
            the base implementation builds for state types a griplink controller does
            not command.
        """
        if state_type in (GripperState.OPEN, GripperState.CLOSE):
            configuration = GriplinkPresetConfiguration.from_state_type(
                self, state_type
            )
        elif state_type in (GripperState.FLEXOPEN, GripperState.FLEXCLOSE):
            configuration = GriplinkFlexConfiguration.from_state_type(self, state_type)
        else:
            return super().default_configuration(state_type, finger_velocity)
        if finger_velocity is not None:
            configuration.finger_velocity = finger_velocity
        return configuration

    def griplink_endpoint(self, state_type: GripperState) -> GriplinkEndpoint | None:
        """
        :param state_type: The gripper state to command.
        :return: The endpoint the griplink server of this gripper listens on for the
            given state, or ``None`` if this gripper serves no endpoint for it.
        """
        return self.griplink_endpoints.get(state_type)
