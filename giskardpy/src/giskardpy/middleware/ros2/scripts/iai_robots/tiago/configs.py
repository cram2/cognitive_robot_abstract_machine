"""
Robot interface configurations for the TIAGo robot in ROS 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from semantic_digital_twin.robots.tiago import TiagoJoint
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


@dataclass(kw_only=True)
class TiagoVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands the dual arms, torso, head, and mobile base of TIAGo through their velocity
    controllers.
    """

    @staticmethod
    def left_arm_velocity_joint_names() -> list[TiagoJoint]:
        """
        Joints driven by /arm_left_velocity_controller/command.

        Their order matches the controller's YAML command layout, so it is significant.
        """
        return [
            TiagoJoint.LEFT_ARM_1,
            TiagoJoint.LEFT_ARM_2,
            TiagoJoint.LEFT_ARM_3,
            TiagoJoint.LEFT_ARM_4,
            TiagoJoint.LEFT_ARM_5,
            TiagoJoint.LEFT_ARM_6,
            TiagoJoint.LEFT_ARM_7,
        ]

    @staticmethod
    def right_arm_velocity_joint_names() -> list[TiagoJoint]:
        """
        Joints driven by /arm_right_velocity_controller/command.

        Their order matches the controller's YAML command layout, so it is significant.
        """
        return [
            TiagoJoint.RIGHT_ARM_1,
            TiagoJoint.RIGHT_ARM_2,
            TiagoJoint.RIGHT_ARM_3,
            TiagoJoint.RIGHT_ARM_4,
            TiagoJoint.RIGHT_ARM_5,
            TiagoJoint.RIGHT_ARM_6,
            TiagoJoint.RIGHT_ARM_7,
        ]

    @staticmethod
    def torso_velocity_joint_names() -> list[TiagoJoint]:
        """
        Joints driven by /torso_velocity_controller/command.

        Their order matches the controller's YAML command layout, so it is significant.
        """
        return [
            TiagoJoint.TORSO_LIFT,
        ]

    @staticmethod
    def head_velocity_joint_names() -> list[TiagoJoint]:
        """
        Joints driven by /head_velocity_controller/command.

        Their order matches the controller's YAML command layout, so it is significant.
        """
        return [
            TiagoJoint.HEAD_1,
            TiagoJoint.HEAD_2,
        ]

    def setup(self) -> None:
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(connection_type=Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame=self.robot.mobile_base.odometry_body_name,
        )

        self.sync_robot_parts()

        diff_drive = self.world.get_connections_by_type(
            connection_type=DifferentialDrive
        )[0]
        self.add_drive_velocity_command_topic(
            command_topic="/cmd_vel",
            drive=diff_drive,
        )

        self.add_joint_group_velocity_command_topic(
            command_topic="/arm_left_velocity_controller/command",
            connections=self.left_arm_velocity_joint_names(),
            minimum_valid_velocity=0.0,
        )

        self.add_joint_group_velocity_command_topic(
            command_topic="/arm_right_velocity_controller/command",
            connections=self.right_arm_velocity_joint_names(),
            minimum_valid_velocity=0.0,
        )

        self.add_joint_group_velocity_command_topic(
            command_topic="/torso_velocity_controller/command",
            connections=self.torso_velocity_joint_names(),
            minimum_valid_velocity=0.0,
        )

        self.add_joint_group_velocity_command_topic(
            command_topic="/head_velocity_controller/command",
            connections=self.head_velocity_joint_names(),
            minimum_valid_velocity=0.0,
        )
