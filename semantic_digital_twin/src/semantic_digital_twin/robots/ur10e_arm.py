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


@dataclass(frozen=True)
class UR10eJointDrive:
    """
    What drives one joint of a UR10e, as MuJoCo Menagerie tunes it in
    ``universal_robots_ur10e/ur10e.xml``.

    Stiffness, servo damping and armature are the same for every joint of the arm there,
    so they are written once here; only the torque limit and the joint's own passive
    damping differ between the shoulder, elbow and wrist joints.
    """

    torque_limit: float
    """
    The largest torque the servo may exert.
    """

    joint_damping: float
    """
    The joint's own passive damping, on top of the servo's.
    """

    stiffness: float = 5_000.0
    """
    How hard the servo pulls towards its commanded position.
    """

    servo_damping: float = 500.0
    """
    How hard the servo resists the joint's velocity.
    """

    armature: float = 0.1
    """
    The rotor inertia reflected through the gearbox.
    """

    @property
    def servo(self) -> JointServo:
        """
        The servo these values describe.
        """
        return JointServo(
            gains=ServoGains(
                stiffness=self.stiffness,
                damping=self.servo_damping,
                torque_limit=self.torque_limit,
            ),
            dynamics=JointDynamics(armature=self.armature, damping=self.joint_damping),
        )


@dataclass(eq=False)
class UR10eArm(Arm[TGenericEndEffector], Generic[TGenericEndEffector], ABC):
    """
    A Universal Robots UR10e arm.

    The robot description the arm is parsed from is a URDF, which carries the joints but
    nothing about what drives them, and there is no MuJoCo description of this arm in
    this repository to read that from either. The arm therefore declares its own drives,
    the way :class:`~semantic_digital_twin.robots.robotiq_85_gripper.Robotiq85Gripper`
    declares the gripper's, with the values of :class:`UR10eJointDrive`.
    """

    @property
    def drives_by_joint(self) -> Dict[str, UR10eJointDrive]:
        """
        What drives each joint, keyed by the joint's name without the arm's
        ``left_``/``right_`` prefix.

        The two shoulder joints carry the whole rest of the arm and need the most torque
        and passive damping to settle without ringing, the elbow less, and the three
        wrist joints, which carry only the gripper, the least.
        """
        shoulder = UR10eJointDrive(torque_limit=330.0, joint_damping=10.0)
        elbow = UR10eJointDrive(torque_limit=150.0, joint_damping=5.0)
        wrist = UR10eJointDrive(torque_limit=56.0, joint_damping=2.0)
        return {
            "shoulder_pan_joint": shoulder,
            "shoulder_lift_joint": shoulder,
            "elbow_joint": elbow,
            "wrist_1_joint": wrist,
            "wrist_2_joint": wrist,
            "wrist_3_joint": wrist,
        }

    @property
    def servos_by_joint(self) -> Dict[str, JointServo]:
        """
        Each joint's servo, keyed the same way as :attr:`drives_by_joint`.
        """
        return {
            joint_name: drive.servo
            for joint_name, drive in self.drives_by_joint.items()
        }

    def _setup_servos(self) -> None:
        for connection in self.active_connections:
            if not isinstance(connection, ActiveConnection1DOF):
                continue
            joint_name = connection.raw_dof.name.name
            unprefixed = joint_name.removeprefix("left_").removeprefix("right_")
            self._declare_servo(connection, self.servos_by_joint[unprefixed])
        self._compensate_gravity()
