from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from giskardpy.middleware.ros2.command_publishing import (
    MultiDOFCommandFormat,
    TwistStampedFormat,
)
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    RootTipPair,
)
from giskardpy.model.world_config import WorldWithMobileRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.garmi import Garmi, GarmiJoint
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    OmniDrive,
)


class GarmiInteractiveMarkerChain(Enum):
    """
    The kinematic chains controllable via interactive markers.
    """

    LEFT_ARM = RootTipPair(root_link="map", tip_link="left_fr3_hand_tcp")
    RIGHT_ARM = RootTipPair(root_link="map", tip_link="right_fr3_hand_tcp")


@dataclass
class WorldWithGarmiConfig(WorldWithMobileRobot):
    """
    World configuration for the GARMI robot.

    Builds a map -> odom -> GARMI kinematic tree using an omni-drive base.
    """

    odometry_body_name: PrefixedName = field(
        default_factory=lambda: PrefixedName("odom")
    )
    robot_class: type[Garmi] = field(kw_only=True, default=Garmi, init=False)
    """
    The robot class that is applied to the parsed description.
    """


@dataclass
class GarmiVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Closed-loop velocity interface for the real GARMI robot.

    Synchronizes the world state from the merged whole-robot joint state topic and sends
    arm joint velocities to the per-arm group controllers, as documented in
    garmi_description's docs/real_robot.md (the public interface contract).
    """

    def setup(self) -> None:
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )
        omni_drive = self.world.get_connections_by_type(OmniDrive)[0]
        self.sync_odometry_topic("/r100_0603/platform/odom/filtered", omni_drive)
        self.add_drive_velocity_command_topic(
            command_topic="/r100_0603/joy_teleop/cmd_vel",
            drive=omni_drive,
            command_format=TwistStampedFormat(),
        )

        # The merged whole-robot joint state (~50 Hz): arms + lift, the platform's
        # wheels, and the head. The per-subsystem topics
        # stay available if a joint group needs faster sync.
        self.sync_joint_state_topic("/garmi/joint_states")

        self.add_joint_group_velocity_command_topic(
            command_topic="/garmi/arms/left_arm_joint_velocity_controller/reference",
            connections=GarmiJoint.left_arm(),
            command_format=MultiDOFCommandFormat(),
        )
        self.add_joint_group_velocity_command_topic(
            command_topic="/garmi/arms/right_arm_joint_velocity_controller/reference",
            connections=GarmiJoint.right_arm(),
            command_format=MultiDOFCommandFormat(),
        )
