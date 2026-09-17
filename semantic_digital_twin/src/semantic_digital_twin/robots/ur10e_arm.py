from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import Dict, Generic

from semantic_digital_twin.robots.robot_part_mixins import TGenericEndEffector
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.world_description.connection_properties import (
    JointDynamics,
    JointServo,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@dataclass(eq=False)
class UR10eArm(Arm[TGenericEndEffector], Generic[TGenericEndEffector], ABC):
    """
    A Universal Robots UR10e arm.

    Its joints' servos are taken from MuJoCo Menagerie's
    ``universal_robots_ur10e/ur10e.xml``: a stiffness of 5000, a servo damping of 500
    and an armature of 0.1 apply to every joint regardless of size, and only the torque
    limit and the joint's passive damping differ per size class.
    """

    @staticmethod
    def _size_class(torque_limit: float, joint_damping: float) -> JointServo:
        """
        A servo at :attr:`servos_by_joint`'s shared stiffness and damping, sized to one
        joint's own torque limit and passive damping.

        :param torque_limit: The largest torque the servo may exert.
        :param joint_damping: The joint's own passive damping, on top of the servo's.
        :return: The servo.
        """
        return JointServo(
            gains=ServoGains(
                stiffness=5_000.0, damping=500.0, torque_limit=torque_limit
            ),
            dynamics=JointDynamics(armature=0.1, damping=joint_damping),
        )

    @property
    def servos_by_joint(self) -> Dict[str, JointServo]:
        """
        Each joint's servo, keyed by the joint's name without the arm's
        ``left_``/``right_`` prefix.

        The two shoulder joints carry the whole rest of the arm and need the most torque
        and passive damping to settle without ringing, the elbow less, and the three
        wrist joints, which carry only the gripper, the least.
        """
        shoulder = self._size_class(torque_limit=330.0, joint_damping=10.0)
        elbow = self._size_class(torque_limit=150.0, joint_damping=5.0)
        wrist = self._size_class(torque_limit=56.0, joint_damping=2.0)
        return {
            "shoulder_pan_joint": shoulder,
            "shoulder_lift_joint": shoulder,
            "elbow_joint": elbow,
            "wrist_1_joint": wrist,
            "wrist_2_joint": wrist,
            "wrist_3_joint": wrist,
        }

    def _setup_servos(self) -> None:
        for connection in self.active_connections:
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            joint_name = connection.raw_dof.name.name
            unprefixed = joint_name.removeprefix("left_").removeprefix("right_")
            self._declare_servo(connection, self.servos_by_joint[unprefixed])
        self._compensate_gravity()
