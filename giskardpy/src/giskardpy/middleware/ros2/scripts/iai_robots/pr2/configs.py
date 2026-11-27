"""
Robot interface configurations for the PR2 in ROS 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from giskardpy.middleware.ros2.command_publishing import MultiDOFCommandFormat
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)

# %% tf frames


class PR2LocalizationFrame(StrEnum):
    """
    The tf frames the PR2 is localized in.
    """

    MAP = "map"
    """
    The frame of the map the robot localizes itself in.
    """


# %% command topics


class PR2CommandTopic(StrEnum):
    """
    The topics the PR2's controllers take their velocity commands on.
    """

    RIGHT_ARM = "/r_arm_velocity_controller/reference"
    """
    The joint velocities of the right arm, as a ``control_msgs/MultiDOFCommand``.
    """

    LEFT_ARM = "/l_arm_velocity_controller/reference"
    """
    The joint velocities of the left arm, as a ``control_msgs/MultiDOFCommand``.
    """

    TORSO = "/torso_velocity_controller/reference"
    """
    The velocity of the torso lift, as a ``control_msgs/MultiDOFCommand``.
    """

    HEAD = "/head_velocity_controller/reference"
    """
    The joint velocities of the head, as a ``control_msgs/MultiDOFCommand``.
    """

    BASE = "/base_controller/command"
    """
    The velocity of the omnidirectional base, as a ``geometry_msgs/Twist``.
    """


# %% velocity interface


@dataclass(kw_only=True)
class PR2VelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands the arms, torso, head and base of the PR2 through their velocity
    controllers, localized by tf from ``map``.

    .. note:: The grippers are not commanded; Giskard only reads their joints.
    """

    @staticmethod
    def right_arm_velocity_joint_names() -> list[PR2Joint]:
        """
        Joints driven by :attr:`PR2CommandTopic.RIGHT_ARM`.

        Their order matches the controller's ``dof_names``, so it is significant.
        """
        return [
            PR2Joint.RIGHT_SHOULDER_PAN,
            PR2Joint.RIGHT_SHOULDER_LIFT,
            PR2Joint.RIGHT_UPPER_ARM_ROLL,
            PR2Joint.RIGHT_ELBOW_FLEX,
            PR2Joint.RIGHT_FOREARM_ROLL,
            PR2Joint.RIGHT_WRIST_FLEX,
            PR2Joint.RIGHT_WRIST_ROLL,
        ]

    @staticmethod
    def left_arm_velocity_joint_names() -> list[PR2Joint]:
        """
        Joints driven by :attr:`PR2CommandTopic.LEFT_ARM`.

        Their order matches the controller's ``dof_names``, so it is significant.
        """
        return [
            PR2Joint.LEFT_SHOULDER_PAN,
            PR2Joint.LEFT_SHOULDER_LIFT,
            PR2Joint.LEFT_UPPER_ARM_ROLL,
            PR2Joint.LEFT_ELBOW_FLEX,
            PR2Joint.LEFT_FOREARM_ROLL,
            PR2Joint.LEFT_WRIST_FLEX,
            PR2Joint.LEFT_WRIST_ROLL,
        ]

    @staticmethod
    def torso_velocity_joint_names() -> list[PR2Joint]:
        """
        Joints driven by :attr:`PR2CommandTopic.TORSO`.
        """
        return [PR2Joint.TORSO_LIFT]

    @staticmethod
    def head_velocity_joint_names() -> list[PR2Joint]:
        """
        Joints driven by :attr:`PR2CommandTopic.HEAD`.

        Their order matches the controller's ``dof_names``, so it is significant.
        """
        return [PR2Joint.HEAD_PAN, PR2Joint.HEAD_TILT]

    def setup(self) -> None:
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame=PR2LocalizationFrame.MAP,
            tf_child_frame=self.robot.mobile_base.odometry_body_name,
        )

        self.sync_robot_parts()

        self.add_drive_velocity_command_topic(
            command_topic=PR2CommandTopic.BASE,
            drive=self.world.get_connections_by_type(OmniDrive)[0],
        )

        for command_topic, joint_names in (
            (PR2CommandTopic.RIGHT_ARM, self.right_arm_velocity_joint_names()),
            (PR2CommandTopic.LEFT_ARM, self.left_arm_velocity_joint_names()),
            (PR2CommandTopic.TORSO, self.torso_velocity_joint_names()),
            (PR2CommandTopic.HEAD, self.head_velocity_joint_names()),
        ):
            self.add_joint_group_velocity_command_topic(
                command_topic=command_topic,
                connections=joint_names,
                command_format=MultiDOFCommandFormat(),
            )
