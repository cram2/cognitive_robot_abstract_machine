from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithOmniDriveRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    Connection6DoF,
)


@dataclass
class WorldWithHSRConfig(WorldWithOmniDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=HSRB, init=False)


class HSRStandaloneInterface(RobotInterfaceConfig):
    def setup(self):
        self.register_controlled_joints(
            [
                "arm_flex_joint",
                "arm_lift_joint",
                "arm_roll_joint",
                "head_pan_joint",
                "head_tilt_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                self.world.get_connections_by_type(OmniDrive)[0].name,
            ]
        )


class HSRVelocityInterface(RobotInterfaceConfig):

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )

        omni_drive = self.world.get_connections_by_type(OmniDrive)[0]
        self.sync_odometry_topic(
            "/laser_odom",
            omni_drive,
        )

        self.add_base_cmd_velocity(
            cmd_vel_topic="/omni_base_controller/cmd_vel", joint=omni_drive
        )

        self.sync_joint_state_topic("/joint_states")
        joints_left = [
            "arm_flex_joint",
            "arm_lift_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/realtime_body_controller_real/command", connections=joints_left
        )


class HSRJointTrajInterfaceConfig(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom",
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
        self.sync_joint_state_topic("/hsrb/joint_states")
        self.sync_odometry_topic("/hsrb/odom", self.drive_joint_name)

        self.add_follow_joint_trajectory_server(
            namespace="/hsrb/head_trajectory_controller", fill_velocity_values=True
        )
        self.add_follow_joint_trajectory_server(
            namespace="/hsrb/arm_trajectory_controller", fill_velocity_values=True
        )
        self.add_follow_joint_trajectory_server(
            namespace="/hsrb/omni_base_controller",
            fill_velocity_values=True,
            path_tolerance={
                Derivatives.position: 1,
                Derivatives.velocity: 1,
                Derivatives.acceleration: 100,
            },
        )
        # self.add_base_cmd_velocity(cmd_vel_topic='/hsrb/command_velocity',
        #                            track_only_velocity=True,
        #                            joint_name=self.drive_joint_name)


class HSRMujocoVelocityInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom",
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
        self.sync_joint_state_topic("/hsrb4s/joint_states")
        self.sync_odometry_topic("/hsrb4s/base_footprint", self.drive_joint_name)

        self.add_joint_velocity_controller(
            namespaces=[
                "hsrb4s/arm_flex_joint_velocity_controller",
                "hsrb4s/arm_lift_joint_velocity_controller",
                "hsrb4s/arm_roll_joint_velocity_controller",
                "hsrb4s/head_pan_joint_velocity_controller",
                "hsrb4s/head_tilt_joint_velocity_controller",
                "hsrb4s/wrist_flex_joint_velocity_controller",
                "hsrb4s/wrist_roll_joint_velocity_controller",
            ]
        )

        self.add_base_cmd_velocity(
            cmd_vel_topic="/hsrb4s/cmd_vel", joint_name=self.drive_joint_name
        )


class HSRMujocoPositionInterface(RobotInterfaceConfig):
    map_name: str
    localization_joint_name: str
    odom_link_name: str
    drive_joint_name: str

    def __init__(
        self,
        map_name: str = "map",
        localization_joint_name: str = "localization",
        odom_link_name: str = "odom",
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
        self.sync_joint_state_topic("/hsrb4s/joint_states")
        self.sync_odometry_topic("/hsrb4s/base_footprint", self.drive_joint_name)

        self.add_joint_position_controller(
            namespaces=[
                "hsrb4s/arm_flex_joint_position_controller",
                # 'hsrb4s/arm_lift_joint_position_controller',
                "hsrb4s/arm_roll_joint_position_controller",
                "hsrb4s/head_pan_joint_position_controller",
                "hsrb4s/head_tilt_joint_position_controller",
                "hsrb4s/wrist_flex_joint_position_controller",
                "hsrb4s/wrist_roll_joint_position_controller",
            ]
        )

        self.add_joint_velocity_controller(
            namespaces=[
                # 'hsrb4s/arm_flex_joint_position_controller',
                "hsrb4s/arm_lift_joint_position_controller",
                # 'hsrb4s/arm_roll_joint_position_controller',
                # 'hsrb4s/head_pan_joint_position_controller',
                # 'hsrb4s/head_tilt_joint_position_controller',
                # 'hsrb4s/wrist_flex_joint_position_controller',
                # 'hsrb4s/wrist_roll_joint_position_controller'
            ]
        )

        self.add_base_cmd_velocity(
            cmd_vel_topic="/hsrb4s/cmd_vel", joint_name=self.drive_joint_name
        )
