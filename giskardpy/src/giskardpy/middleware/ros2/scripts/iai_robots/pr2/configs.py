from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithOmniDriveRobot
from giskardpy.middleware.ros2.giskard import RobotInterfaceConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
)


@dataclass
class WorldWithPR2Config(WorldWithOmniDriveRobot):
    odom_body_name: PrefixedName = PrefixedName("odom_combined")
    urdf_view: AbstractRobot = field(kw_only=True, default=PR2, init=False)


class PR2StandaloneInterface(RobotInterfaceConfig):
    def setup(self):
        self.register_controlled_joints(
            [
                "torso_lift_joint",
                "head_pan_joint",
                "head_tilt_joint",
                "r_shoulder_pan_joint",
                "r_shoulder_lift_joint",
                "r_upper_arm_roll_joint",
                "r_forearm_roll_joint",
                "r_elbow_flex_joint",
                "r_wrist_flex_joint",
                "r_wrist_roll_joint",
                "l_shoulder_pan_joint",
                "l_shoulder_lift_joint",
                "l_upper_arm_roll_joint",
                "l_forearm_roll_joint",
                "l_elbow_flex_joint",
                "l_wrist_flex_joint",
                "l_wrist_roll_joint",
                self.world.get_connections_by_type(OmniDrive)[0].name,
            ]
        )


class PR2JointTrajServerMujocoInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom_combined",
        drive_joint_name: str = "brumbrum",
    ):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint_name=self.localization_joint_name,
            tf_parent_frame=self.map_name,
            tf_child_frame=self.odom_link_name,
        )
        self.sync_joint_state_topic("/joint_states")
        self.sync_odometry_topic("/pr2/base_footprint", self.drive_joint_name)
        self.add_follow_joint_trajectory_server(namespace="/pr2/whole_body_controller")
        self.add_follow_joint_trajectory_server(
            namespace="/pr2/l_gripper_l_finger_controller"
        )
        self.add_follow_joint_trajectory_server(
            namespace="/pr2/r_gripper_l_finger_controller"
        )
        self.add_base_cmd_velocity(
            cmd_vel_topic="/pr2/cmd_vel",
            track_only_velocity=True,
            joint_name=self.drive_joint_name,
        )


class PR2VelocityMujocoInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom_combined",
        drive_joint_name: str = "brumbrum",
    ):
        self.map_name = map_name
        self.localization_joint_name = localization_joint_name
        self.odom_link_name = odom_link_name
        self.drive_joint_name = drive_joint_name

    def setup(self):
        self.discover_interfaces_from_controller_manager()
        self.sync_odometry_topic("/odom", self.drive_joint_name)
        self.add_base_cmd_velocity(cmd_vel_topic="/cmd_vel")
