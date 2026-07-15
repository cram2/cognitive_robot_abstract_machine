from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from giskardpy.tree.behaviors.joint_group_vel_controller_publisher import (
    MultiDOFVelocityCommand,
)
from giskardpy.model.world_config import WorldWithOmniDriveRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)

GARMI_LEFT_ARM_JOINTS = [
    "left_fr3_joint1",
    "left_fr3_joint2",
    "left_fr3_joint3",
    "left_fr3_joint4",
    "left_fr3_joint5",
    "left_fr3_joint6",
    "left_fr3_joint7",
]
"""Names of the seven left FR3 arm joints, ordered from base to tip."""

GARMI_RIGHT_ARM_JOINTS = [
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
]
"""Names of the seven right FR3 arm joints, ordered from base to tip."""

GARMI_HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
"""Names of the head pan and tilt joints."""

GARMI_LIFT_JOINTS = ["lift_0_lower_joint", "lift_0_upper_joint"]
"""Names of the two prismatic torso lift joints."""


@dataclass
class WorldWithGarmiConfig(WorldWithOmniDriveRobot):
    """
    World configuration for the GARMI robot.

    Builds a map -> odom_combined -> GARMI kinematic tree using an omni-drive base.
    """

    odom_body_name: PrefixedName = field(
        default_factory=lambda: PrefixedName("odom_combined")
    )
    urdf_view: Garmi = field(kw_only=True, default=Garmi)


class GarmiStandaloneInterface(StandAloneRobotInterfaceConfig):
    """
    Robot interface configuration for running GARMI in standalone (simulation) mode.

    Registers all hardware-controlled joints: mecanum wheels, lift, head, both FR3 arms, and grippers.
    """

    def __init__(self, drive_joint_name: str = "odom_combined_T_base_link"):
        super().__init__(
            [
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
                *GARMI_LIFT_JOINTS,
                *GARMI_HEAD_JOINTS,
                *GARMI_LEFT_ARM_JOINTS,
                *GARMI_RIGHT_ARM_JOINTS,
                "arm_0_gripper_fr3_finger_joint1",
                "arm_0_gripper_fr3_finger_joint2",
                "arm_1_gripper_fr3_finger_joint1",
                "arm_1_gripper_fr3_finger_joint2",
                drive_joint_name,
            ]
        )


class GarmiVelocityInterface(RobotInterfaceConfig):
    """
    Closed-loop velocity interface for the real GARMI robot.

    Synchronizes the world state from joint-state and odometry topics and sends joint
    velocities to per-subsystem group controllers as well as base twists to the drive.

    .. warning::
        The ROS topic and TF frame names below are placeholders for the online
        integration meeting and must be replaced with the names published by the
        GARMI hardware bring-up.
    """

    def setup(self) -> None:
        # self.sync_6dof_joint_with_tf_frame(
        #    joint=self.world.get_connections_by_type(Connection6DoF)[0],
        #    tf_parent_frame="placeholder_map_frame",
        #    tf_child_frame="placeholder_odom_frame",
        # )

        # omni_drive = self.world.get_connections_by_type(OmniDrive)[0]
        # self.sync_odometry_topic("/placeholder/base/odom", omni_drive)
        # self.add_base_cmd_velocity(
        #    cmd_vel_topic="/placeholder/base/cmd_vel", joint=omni_drive
        # )

        self.sync_joint_state_topic("/garmi/arms/joint_states")

        self.add_joint_velocity_group_controller(
            cmd_topic="/garmi/arms/left_arm_joint_velocity_controller/reference",
            connections=GARMI_LEFT_ARM_JOINTS,
            velocity_command=MultiDOFVelocityCommand(),
        )
        self.add_joint_velocity_group_controller(
            cmd_topic="/garmi/arms/right_arm_joint_velocity_controller/reference",
            connections=GARMI_RIGHT_ARM_JOINTS,
            velocity_command=MultiDOFVelocityCommand(),
        )

        # self.add_joint_velocity_group_controller(
        #    cmd_topic="/placeholder/head/velocity_controller/commands",
        #    connections=GARMI_HEAD_JOINTS,
        # )
        # self.add_joint_velocity_group_controller(
        #    cmd_topic="/placeholder/lift/velocity_controller/commands",
        #    connections=GARMI_LIFT_JOINTS,
        # )
