from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Self

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.exceptions import ConnectionsOutsideEndEffector
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import EndEffector

TGripperConfiguration = TypeVar("TGripperConfiguration", bound="GripperConfiguration")
"""
The configuration type a motion or mapping is bound to; concrete consumers bind it to
one configuration subclass.
"""

# %% Base configuration


@dataclass(eq=False)
class GripperConfiguration(SubClassSafeGeneric, ABC, Generic[TGripperConfiguration]):
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
