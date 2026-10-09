from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Generic, Self

from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ConnectionsOutsideEndEffector
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import EndEffector

TGripperConfiguration = TypeVar("TGripperConfiguration", bound="GripperConfiguration")
"""
The configuration type a motion or mapping is bound to; concrete consumers bind it to
one configuration subclass.
"""

# %% Grip presets


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


# %% Base configuration


@dataclass(eq=False)
class GripperConfiguration(Generic[TGripperConfiguration], SubClassSafeGeneric, ABC):
    """
    A configuration a gripper can be commanded into.

    Carries the end effector the configuration configures and the joint positions to
    reach, so both the action and the motion that consume it share one object instead of
    resolving the end effector and looking up the joint state independently.
    """

    end_effector: EndEffector
    """
    The end effector this configuration configures.
    """

    joint_state: JointState
    """
    The positions the end effector's connections are commanded to.
    """

    finger_velocity: float | None = None
    """
    Maximum finger joint velocity (in m/s) enforced during the motion.

    ``None`` leaves the speed unconstrained.
    """

    @classmethod
    def from_state_type(
        cls, end_effector: EndEffector, state_type: GripperState
    ) -> Self:
        """
        :param end_effector: The end effector whose declared state is used.
        :param state_type: The state type to build the configuration for.
        :return: The configuration for the declared state of the given type.
        :raises NoJointStateWithType: If the end effector declares no such state.
        """
        return cls(
            end_effector=end_effector,
            joint_state=end_effector.get_joint_state_by_type(state_type),
        )

    def __post_init__(self):
        """
        Validates that the joint state only commands connections inside the end
        effector.

        :return: None
        :raises ConnectionsOutsideEndEffector: If the joint state commands a connection
            outside the end effector.
        """
        if self.end_effector._world is None:
            # The end effector is being reconstructed detached from a world (e.g. from
            # the database), so its connections cannot be looked up yet.
            return
        connections_outside_end_effector = set(self.joint_state.connections) - set(
            self.end_effector.active_connections
        )
        if connections_outside_end_effector:
            raise ConnectionsOutsideEndEffector(
                end_effector=self.end_effector,
                foreign_connection_names=[
                    str(c.name) for c in connections_outside_end_effector
                ],
            )


# %% Default configuration


@dataclass(eq=False)
class GripperStateConfiguration(GripperConfiguration):
    """
    The configuration a robot description already declares for a gripper, selected by
    its :class:`~semantic_digital_twin.datastructures.definitions.GripperState`.
    """

    @classmethod
    def closed(cls, end_effector: EndEffector) -> Self:
        """
        :param end_effector: The end effector to build the configuration for.
        :return: The configuration for the end effector's closed state.
        """
        return cls.from_state_type(end_effector, GripperState.CLOSE)


# %% Griplink configurations

MAXIMUM_OPENING_WIDTH_MM = 120
"""
Maximum opening width of the griplink gripper in millimetres, the scale the flex
configuration interpolates ``grip_position`` on.
"""

FULLY_CLOSED_OPENING_WIDTH_MM = 0
"""
Opening width of the griplink gripper in millimetres that commands the fully closed
state.
"""


@dataclass(eq=False)
class GriplinkGripperConfiguration(GripperConfiguration, SubclassJSONSerializer, ABC):
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

    grip_position: int | None = None
    """
    Opening width of the gripper in millimetres [-5..120], used by ``Flexgrip``/
    ``Flexrelease``.

    ``None`` commands the declared open or close state, depending on the state type.
    """

    grip_force: int | None = None
    """
    Force the gripper applies to the object in newtons [30..300], used by ``Flexgrip``
    only (ignored for ``Flexrelease``, which has no force goal).

    ``None`` defers to the default the griplink action server picks (90).
    """

    grip_speed: int | None = None
    """
    Motion speed of the gripper in millimetres per second [5..350], used by
    ``Flexgrip``/``Flexrelease``.

    ``None`` defers to the per-motion default the griplink action server picks (150 for
    ``Flexgrip``, 250 for ``Flexrelease``).
    """

    grip_acceleration: int | None = None
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
        grip_position: int | None = None,
        grip_force: int | None = None,
        grip_speed: int | None = None,
        grip_acceleration: int | None = None,
    ) -> Self:
        """
        :param end_effector: The griplink gripper to build the configuration for.
        :param state_type: The state type the configuration is labelled with.
        :param grip_position: Opening width in millimetres; ``None`` commands the
            declared open or close state.
        :param grip_force: Force in newtons; ``None`` defers to the action server
            default.
        :param grip_speed: Speed in millimetres per second; ``None`` defers to the
            action server default.
        :param grip_acceleration: Acceleration in millimetres per second squared;
            ``None`` defers to the action server default.
        :return: The configuration for the given state type, commanding the joint state
            the parameters describe.
        """
        return cls(
            end_effector=end_effector,
            joint_state=cls._build_joint_state(end_effector, state_type, grip_position),
            grip_position=grip_position,
            grip_force=grip_force,
            grip_speed=grip_speed,
            grip_acceleration=grip_acceleration,
        )

    @staticmethod
    def _build_joint_state(
        end_effector: EndEffector,
        state_type: GripperState,
        grip_position: int | None,
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
        if grip_position is not None:
            commanded_position = grip_position
        elif state_type is GripperState.FLEXCLOSE:
            commanded_position = FULLY_CLOSED_OPENING_WIDTH_MM
        else:
            commanded_position = MAXIMUM_OPENING_WIDTH_MM
        open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)
        close_state = end_effector.get_joint_state_by_type(GripperState.CLOSE)
        fraction = (MAXIMUM_OPENING_WIDTH_MM - commanded_position) / (
            MAXIMUM_OPENING_WIDTH_MM
        )
        close_targets = dict(close_state.items())
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
