from dataclasses import dataclass, field

from dataclasses import dataclass, field

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
    RobotInterfaceConfig,
)
from giskardpy.model.world_config import (
    WorldWithOmniDriveRobot,
    WorldWithDiffDriveRobot,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


class StretchStandaloneInterface(StandAloneRobotInterfaceConfig):

    def __init__(self, drive_joint_name: str = "brumbrum"):
        super().__init__(
            [
                drive_joint_name,
                "joint_gripper_finger_left",
                "joint_gripper_finger_right",
                "joint_right_wheel",
                "joint_left_wheel",
                "joint_lift",
                "joint_arm_l3",
                "joint_arm_l2",
                "joint_arm_l1",
                "joint_arm_l0",
                "joint_wrist_yaw",
                "joint_head_pan",
                "joint_head_tilt",
            ]
        )


class StretchVelocityInterface(RobotInterfaceConfig):

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )

        diff_drive = self.world.get_connections_by_type(DifferentialDrive)[0]
        self.sync_odometry_topic(
            "/odom",
            diff_drive,
        )

        self.add_base_cmd_velocity(cmd_vel_topic="/stretch/cmd_vel", joint=diff_drive)

        self.sync_joint_state_topic("/joint_states")
        joints = [
            "joint_arm_l0",
            "joint_lift",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
            "joint_head_pan",
            "joint_head_pan",
            "joint_gripper_finger_left",
            "joint_right_wheel",
            "joint_left_wheel",
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/joint_velocity_cmd", connections=joints
        )


@dataclass
class WorldWithStretchConfig(WorldWithOmniDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=Stretch, init=False)

    def setup_collision_config(self):
        pass


@dataclass
class WorldWithStretchConfigDiffDrive(WorldWithDiffDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=Stretch, init=False)

    def setup_collision_config(self):
        pass
