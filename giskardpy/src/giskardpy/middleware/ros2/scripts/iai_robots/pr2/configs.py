from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithOmniDriveRobot
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
)


@dataclass
class WorldWithPR2Config(WorldWithOmniDriveRobot):
    odom_body_name: PrefixedName = PrefixedName("odom_combined")
    urdf_view: AbstractRobot = field(kw_only=True, default=PR2, init=False)


@dataclass
class PR2StandaloneInterface(RobotInterfaceConfig):
    """
    Simulates the arms, torso, head and drive of the PR2 without talking to hardware.
    """

    def setup(self):
        self.register_controlled_joints(
            [
                PR2Joint.TORSO_LIFT,
                PR2Joint.HEAD_PAN,
                PR2Joint.HEAD_TILT,
                PR2Joint.RIGHT_SHOULDER_PAN,
                PR2Joint.RIGHT_SHOULDER_LIFT,
                PR2Joint.RIGHT_UPPER_ARM_ROLL,
                PR2Joint.RIGHT_FOREARM_ROLL,
                PR2Joint.RIGHT_ELBOW_FLEX,
                PR2Joint.RIGHT_WRIST_FLEX,
                PR2Joint.RIGHT_WRIST_ROLL,
                PR2Joint.LEFT_SHOULDER_PAN,
                PR2Joint.LEFT_SHOULDER_LIFT,
                PR2Joint.LEFT_UPPER_ARM_ROLL,
                PR2Joint.LEFT_FOREARM_ROLL,
                PR2Joint.LEFT_ELBOW_FLEX,
                PR2Joint.LEFT_WRIST_FLEX,
                PR2Joint.LEFT_WRIST_ROLL,
                self.world.get_connections_by_type(OmniDrive)[0].name,
            ]
        )


@dataclass
class PR2VelocityMujocoInterface(RobotInterfaceConfig):
    """
    Commands a PR2 simulated in mujoco through its controller manager.
    """

    map_name: str = "map"
    """
    Name of the frame the localization is expressed in.
    """

    localization_joint_name: str = "localization"
    """
    Name of the 6 degree of freedom connection carrying the localization.
    """

    odom_link_name: str = "odom_combined"
    """
    Name of the body the drive moves relative to.
    """

    drive_joint_name: str = "brumbrum"
    """
    Name of the drive connection that the odometry topic is synced into.
    """

    def setup(self):
        self.discover_interfaces_from_controller_manager()
        self.sync_odometry_topic("/odom", self.drive_joint_name)
        self.add_base_cmd_velocity(cmd_vel_topic="/cmd_vel")
