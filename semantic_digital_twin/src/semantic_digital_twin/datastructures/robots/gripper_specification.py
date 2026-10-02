from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, Optional

from typing_extensions import Self, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ConnectionsOutsideEndEffector
from semantic_digital_twin.datastructures.robots.gripper_configurations import (
    GriplinkGripperConfiguration,
)

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import EndEffector


TGripperSpecification = TypeVar("TGripperSpecification", bound="GripperSpecification")
"""
The specification type a motion or mapping is bound to; concrete consumers bind it to
one specification subclass.
"""

# %% Base specification


@dataclass(eq=False)
class GripperSpecification(Generic[TGripperSpecification], SubClassSafeGeneric, ABC):
    """
    A configuration a gripper can be commanded into.

    Carries the end effector the specification configures and the joint positions to
    reach, so both the action and the motion that consume it share one object instead of
    resolving the end effector and looking up the joint state independently.
    """

    end_effector: EndEffector
    """
    The end effector this specification configures.
    """

    joint_state: JointState
    """
    The positions the end effector's connections are commanded to.
    """

    finger_velocity: Optional[float] = field(default=None)
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
        :param state_type: The state type to build the specification for.
        :return: The specification for the declared state of the given type.
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


# %% Default specification


@dataclass(eq=False)
class GripperStateSpecification(GripperSpecification):
    """
    The configuration a robot description already declares for a gripper, selected by
    its :class:`~semantic_digital_twin.datastructures.definitions.GripperState`.
    """

    @classmethod
    def opened(cls, end_effector: EndEffector) -> Self:
        """
        :param end_effector: The end effector to build the specification for.
        :return: The specification for the end effector's open state.
        """
        return cls.from_state_type(end_effector, GripperState.OPEN)

    @classmethod
    def closed(cls, end_effector: EndEffector) -> Self:
        """
        :param end_effector: The end effector to build the specification for.
        :return: The specification for the end effector's closed state.
        """
        return cls.from_state_type(end_effector, GripperState.CLOSE)


# %% Griplink specifications


@dataclass(eq=False)
class GriplinkPresetSpecification(GripperSpecification):
    """
    A griplink gripper motion driven by a stored grip preset, used for ``Grip``/
    ``Release`` actions.
    """

    configuration: GriplinkGripperConfiguration = field(
        default_factory=GriplinkGripperConfiguration
    )
    """
    Hardware parameters forwarded to the griplink action server; the alternative reads
    :attr:`~GriplinkGripperConfiguration.grip_preset`.
    """

    @classmethod
    def from_state_type(
        cls,
        end_effector: EndEffector,
        state_type: GripperState,
        configuration: Optional[GriplinkGripperConfiguration] = None,
    ) -> Self:
        """
        :param end_effector: The griplink gripper to build the preset specification for.
        :param state_type: The declared open/close state the preset executes.
        :param configuration: Hardware parameters forwarded to the griplink action
            server; ``None`` uses the default configuration.
        :return: The preset specification for the declared open/close state.
        """
        return cls(
            end_effector=end_effector,
            joint_state=end_effector.get_joint_state_by_type(state_type),
            configuration=configuration or GriplinkGripperConfiguration(),
        )


@dataclass(eq=False)
class GriplinkFlexSpecification(GripperSpecification):
    """
    A griplink gripper motion driven by a commanded opening width, used for
    ``Flexgrip``/``Flexrelease`` actions.
    """

    configuration: GriplinkGripperConfiguration = field(
        default_factory=GriplinkGripperConfiguration
    )
    """
    Hardware parameters forwarded to the griplink action server; the alternative reads
    :attr:`~GriplinkGripperConfiguration.grip_position`, :attr:`~grip_force`,
    :attr:`~grip_speed` and :attr:`~grip_acceleration`.
    """

    @staticmethod
    def _flex_joint_state(
        end_effector: EndEffector,
        grip_position: Optional[int],
        state_type: GripperState,
    ) -> JointState:
        """
        Build the joint state a griplink flex motion commands, from the opening width
        the controller accepts.

        :param end_effector: The griplink gripper whose connections are commanded.
        :param grip_position: Opening width in millimetres [-5..120]; ``None`` defaults
            to fully open (120).
        :param state_type: The flex state type the joint state is labelled with.
        :return: The joint state driving the gripper's connections to the interpolated
            position.
        """
        position = grip_position if grip_position is not None else 120
        open_state = end_effector.get_joint_state_by_type(GripperState.OPEN)
        fraction = (120 - position) / 120
        target_values = []
        for connection in open_state.connections:
            lower = connection.dof.limits.lower.position or 0.0
            upper = connection.dof.limits.upper.position or 0.0
            target_values.append(lower + fraction * (upper - lower))
        return JointState(
            connections=open_state.connections,
            target_values=target_values,
            state_type=state_type,
            name=PrefixedName("flexgrip", prefix=end_effector.name.name),
        )

    @classmethod
    def from_state_type(
        cls,
        end_effector: EndEffector,
        state_type: GripperState,
        configuration: Optional[GriplinkGripperConfiguration] = None,
    ) -> Self:
        """
        :param end_effector: The griplink gripper to build the flex specification for.
        :param state_type: The flex state type the specification is labelled with.
        :param configuration: Hardware parameters forwarded to the griplink action
            server; its opening width drives the joint state; ``None`` uses the
            default configuration.
        :return: The flex specification for the given flex state type, with the joint
            state interpolated from the configuration's grip position.
        """
        configuration = configuration or GriplinkGripperConfiguration()
        return cls(
            end_effector=end_effector,
            joint_state=cls._flex_joint_state(
                end_effector, configuration.grip_position, state_type
            ),
            configuration=configuration,
        )
