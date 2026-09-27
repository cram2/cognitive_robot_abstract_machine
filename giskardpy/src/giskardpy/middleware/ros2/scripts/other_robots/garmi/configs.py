from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.middleware.ros2.command_publishing import MultiDOFCommandFormat
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
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
"""
Names of the seven left FR3 arm joints, ordered from base to tip.
"""

GARMI_RIGHT_ARM_JOINTS = [
    "right_fr3_joint1",
    "right_fr3_joint2",
    "right_fr3_joint3",
    "right_fr3_joint4",
    "right_fr3_joint5",
    "right_fr3_joint6",
    "right_fr3_joint7",
]
"""
Names of the seven right FR3 arm joints, ordered from base to tip.
"""

GARMI_FINGER_JOINTS = [
    "left_fr3_finger_joint1",
    "left_fr3_finger_joint2",
    "right_fr3_finger_joint1",
    "right_fr3_finger_joint2",
]
"""
Names of the two finger joints of each FR3 gripper.
"""

GARMI_WHEEL_JOINTS = [
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
]
"""
Names of the four mecanum wheel joints of the base.
"""

GARMI_HEAD_JOINTS = ["o1_motor_1", "o1_motor_2"]
"""
Names of the head pan and tilt joints.
"""

GARMI_LIFT_JOINTS = ["lift_0_lower_joint", "lift_0_upper_joint"]
"""
Names of the two prismatic torso lift joints.
"""

GARMI_INTERACTIVE_MARKER_ROOT_LINKS = ["map", "map"]
"""
Root links of the kinematic chains controllable via interactive markers.

Both arms are rooted at "map", so a drag expresses a whole-body goal and may
recruit the base and lift alongside the arm. Root an arm at its mount
("arm_mount_left_link" / "arm_mount_right_link") instead for arm-only goals --
that also renders in RViz when no map frame is on tf.
"""

GARMI_INTERACTIVE_MARKER_TIP_LINKS = ["left_fr3_hand_tcp", "right_fr3_hand_tcp"]
"""
Tip links (arm TCPs) corresponding to :data:`GARMI_INTERACTIVE_MARKER_ROOT_LINKS`.
"""


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
                *GARMI_WHEEL_JOINTS,
                *GARMI_LIFT_JOINTS,
                *GARMI_HEAD_JOINTS,
                *GARMI_LEFT_ARM_JOINTS,
                *GARMI_RIGHT_ARM_JOINTS,
                *GARMI_FINGER_JOINTS,
                self.world.get_connections_by_type(OmniDrive)[0].name,
            ]
        )


@dataclass
class GarmiVelocityInterface(RobotInterfaceConfig):
    """
    Closed-loop velocity interface for the GARMI robot.

    Synchronizes the world state from the merged whole-robot joint state topic and
    sends arm joint velocities to the per-arm group controllers, as documented in
    garmi_description's docs/real_robot.md (the public interface contract).
    """

    def setup(self) -> None:
        # Base localization and drive, per real_robot.md. Both producers run on
        # the current robot deployment: map->odom is on the global /tf (static
        # identity while localization is "none") and the legs EKF publishes the
        # filtered platform odometry at 50 Hz.
        # NOTE the base listens on /r100_0603/cmd_vel as geometry_msgs/TwistStamped,
        # while add_base_cmd_velocity publishes a plain Twist -- so the twist goes
        # out on the unstamped topic and garmi_giskard's twist_stamping_relay
        # (started by giskard_real.launch.py) stamps it onto /r100_0603/cmd_vel.
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )
        omni_drive = self.world.get_connections_by_type(OmniDrive)[0]
        self.sync_odometry_topic("/r100_0603/platform/odom/filtered", omni_drive)
        self.add_base_cmd_velocity(
            cmd_vel_topic="/r100_0603/cmd_vel_unstamped", joint=omni_drive
        )

        # The merged whole-robot joint state (~50 Hz): arms + lift, the platform's
        # wheels, and the head. The per-subsystem topics documented in
        # real_robot.md stay available if a joint group needs faster sync.
        self.sync_joint_state_topic("/garmi/joint_states")

        self.add_joint_velocity_group_controller(
            cmd_topic="/garmi/arms/left_arm_joint_velocity_controller/reference",
            connections=GARMI_LEFT_ARM_JOINTS,
            command_format=MultiDOFCommandFormat(),
        )
        self.add_joint_velocity_group_controller(
            cmd_topic="/garmi/arms/right_arm_joint_velocity_controller/reference",
            connections=GARMI_RIGHT_ARM_JOINTS,
            command_format=MultiDOFCommandFormat(),
        )

        # The head and the lift take POSITION streams, not velocities -- there is
        # no velocity controller to publish to on either path (see real_robot.md):
        #   head: sensor_msgs/JointState (position only) on
        #         /olive/olixO1/id004/head_goal;
        #   lift: std_msgs/Float64MultiArray on
        #         /garmi/arms/lift_0_position_controller/commands.
        # Wiring them into Giskard means a position-command publisher, not
        # add_joint_velocity_group_controller.
