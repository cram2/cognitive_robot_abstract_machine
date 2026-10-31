from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from krrood.ormatic.utils import classproperty
from typing_extensions import Generic

from semantic_digital_twin.robots.robot_part_mixins import TGenericEndEffector
from semantic_digital_twin.robots.robot_parts import Arm, ServoedRobotPart
from semantic_digital_twin.world_description.connection_properties import (
    JointDrive,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@dataclass(eq=False)
class UR10eArm(
    Arm[TGenericEndEffector], ServoedRobotPart, Generic[TGenericEndEffector], ABC
):
    """
    A Universal Robots UR10e arm.

    The robot description the arm is parsed from is a URDF, which carries the joints but
    nothing about what drives them, and there is no MuJoCo description of this arm in
    this repository to read that from either. The arm therefore declares its own drives,
    the way :class:`~semantic_digital_twin.robots.robotiq_85_gripper.Robotiq85Gripper`
    declares the gripper's, as MuJoCo Menagerie tunes them in
    ``universal_robots_ur10e/ur10e.xml``.
    """

    @classproperty
    def holds_its_own_weight(cls) -> bool:
        return True

    @property
    def drive_of_each_joint(self) -> dict[str, JointDrive]:
        """
        The two shoulder joints carry the whole rest of the arm and need the most torque
        and passive damping to settle without ringing, the elbow less, and the three
        wrist joints, which carry only the gripper, the least.
        """
        shoulder = self._menagerie_drive(torque_limit=330.0, joint_damping=10.0)
        elbow = self._menagerie_drive(torque_limit=150.0, joint_damping=5.0)
        wrist = self._menagerie_drive(torque_limit=56.0, joint_damping=2.0)
        drive_by_unprefixed_name = {
            "shoulder_pan_joint": shoulder,
            "shoulder_lift_joint": shoulder,
            "elbow_joint": elbow,
            "wrist_1_joint": wrist,
            "wrist_2_joint": wrist,
            "wrist_3_joint": wrist,
        }
        return {
            connection.name.name: drive_by_unprefixed_name[
                connection.raw_dof.name.name.removeprefix("left_").removeprefix(
                    "right_"
                )
            ]
            for connection in self.active_connections
            if isinstance(connection, ActiveConnection1DOF)
        }

    @staticmethod
    def _menagerie_drive(torque_limit: float, joint_damping: float) -> JointDrive:
        """
        Stiffness, servo damping and armature are the same for every joint of the arm in
        MuJoCo Menagerie; only the torque limit and the joint's own passive damping
        differ between the shoulder, elbow and wrist joints.

        :param torque_limit: The largest torque the servo may exert.
        :param joint_damping: The joint's own passive damping, on top of the servo's.
        :return: The drive of one joint of the arm.
        """
        return JointDrive(
            gains=ServoGains(
                stiffness=5_000.0, damping=500.0, torque_limit=torque_limit
            ),
            armature=0.1,
            joint_damping=joint_damping,
        )
