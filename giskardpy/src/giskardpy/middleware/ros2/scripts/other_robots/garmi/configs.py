from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from giskardpy.middleware.ros2.command_publishing import MultiDOFCommandFormat
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    RootTipPair,
)
from giskardpy.model.world_config import WorldWithOmniDriveRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.garmi import Garmi, GarmiJoint
from semantic_digital_twin.world_description.connections import OmniDrive


class GarmiInteractiveMarkerChain(Enum):
    """
    The kinematic chains controllable via interactive markers.
    """

    LEFT_ARM = RootTipPair(
        root_link="arm_mount_left_link", tip_link="left_fr3_hand_tcp"
    )
    RIGHT_ARM = RootTipPair(root_link="map", tip_link="right_fr3_hand_tcp")


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


@dataclass
class GarmiStandaloneInterface(RobotInterfaceConfig):
    """
    Simulates the mecanum wheels, lift, head, both FR3 arms, grippers and drive of GARMI
    without talking to hardware.
    """

    def setup(self) -> None:
        self.register_controlled_joints(
            [
                *GarmiJoint.wheels(),
                *GarmiJoint.lift(),
                *GarmiJoint.head(),
                *GarmiJoint.left_arm(),
                *GarmiJoint.right_arm(),
                *GarmiJoint.left_fingers(),
                *GarmiJoint.right_fingers(),
                self.world.get_connections_by_type(OmniDrive)[0].name,
            ]
        )


@dataclass
class GarmiVelocityInterface(RobotInterfaceConfig):
    """
    Closed-loop velocity interface for the real GARMI robot.

    Synchronizes the world state from the arm joint-state topic and sends joint
    velocities to the two arm group controllers. The base, head and lift are not wired
    up yet.
    """

    def setup(self) -> None:
        self.sync_joint_state_topic("/garmi/arms/joint_states")

        self.add_joint_velocity_group_controller(
            cmd_topic="/garmi/arms/left_arm_joint_velocity_controller/reference",
            connections=GarmiJoint.left_arm(),
            command_format=MultiDOFCommandFormat(),
        )
        self.add_joint_velocity_group_controller(
            cmd_topic="/garmi/arms/right_arm_joint_velocity_controller/reference",
            connections=GarmiJoint.right_arm(),
            command_format=MultiDOFCommandFormat(),
        )
