from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import Generic

from semantic_digital_twin.robots.robot_part_mixins import (
    HasTwoFingers,
    TGenericLeftFinger,
    TGenericRightFinger,
)
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
    JointServo,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@dataclass(eq=False)
class Robotiq85Gripper(
    EndEffector,
    HasTwoFingers[TGenericLeftFinger, TGenericRightFinger],
    Generic[TGenericLeftFinger, TGenericRightFinger],
    ABC,
):
    """
    A Robotiq 2F-85 parallel gripper: two fingers coupled by a mimic linkage, driven by
    the knuckle joint of the thumb.

    Knows what a grasp and a physical simulation need from it: the joint that drives
    the fingers and the servo driving it; the pads that meet an object are the tips of
    its thumb and finger.

    The servo is raised empirically, since no pre-tuned reference exists for the
    Robotiq 2F-85; its armature gives the coupled mechanism the numerical damping that
    keeps it from chattering under load.
    """

    @property
    def knuckle_joint(self) -> ActiveConnection1DOF:
        """
        The joint that actually drives the gripper; every other finger joint in the
        mimic linkage follows it.
        """
        return self.thumb.root.parent_connection

    def _setup_servos(self) -> None:
        for connection in self.active_connections:
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            self._declare_servo(
                connection,
                JointServo(
                    gains=ServoGains(stiffness=100.0, damping=10.0, torque_limit=10.0),
                    dynamics=JointDynamics(armature=0.05),
                ),
            )
        self._compensate_gravity()
