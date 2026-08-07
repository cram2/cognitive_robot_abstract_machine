from dataclasses import dataclass, field
from typing import List

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
    RobotInterfaceConfig,
)
from giskardpy.model.world_config import (
    WorldWithOmniDriveRobot,
    WorldWithDiffDriveRobot,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch, StretchJoint
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


@dataclass
class StretchStandaloneInterface(StandAloneRobotInterfaceConfig):
    """
    Simulates the arm, gripper, head and drive of Stretch without talking to hardware.
    """

    drive_joint_name: str = "brumbrum"
    """
    Name of the drive connection that is controlled alongside the other joints.
    """

    joint_names: List[str] = field(init=False, default_factory=list)
    """
    The drive joint plus the arm, gripper, wheel and head joints of Stretch.
    """

    def __post_init__(self) -> None:
        self.joint_names = [
            self.drive_joint_name,
            StretchJoint.GRIPPER_LEFT_FINGER,
            StretchJoint.GRIPPER_RIGHT_FINGER,
            StretchJoint.RIGHT_WHEEL,
            StretchJoint.LEFT_WHEEL,
            StretchJoint.LIFT,
            StretchJoint.ARM_L3,
            StretchJoint.ARM_L2,
            StretchJoint.ARM_L1,
            StretchJoint.ARM_L0,
            StretchJoint.WRIST_YAW,
            StretchJoint.HEAD_PAN,
            StretchJoint.HEAD_TILT,
        ]


@dataclass
class StretchVelocityInterface(RobotInterfaceConfig):
    """
    Commands the arm, head and drive of Stretch through their velocity controllers.
    """

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
            StretchJoint.ARM_L0,  # 0
            StretchJoint.LIFT,  # 1
            StretchJoint.WRIST_YAW,  # 2
            StretchJoint.WRIST_PITCH,  # 3
            StretchJoint.WRIST_ROLL,  # 4
            StretchJoint.HEAD_PAN,  # 5
            StretchJoint.HEAD_TILT,  # 6
            StretchJoint.GRIPPER_LEFT_FINGER,  # 7
            StretchJoint.RIGHT_WHEEL,  # 8
            StretchJoint.LEFT_WHEEL,  # 9
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/joint_velocity_cmd",
            connections=joints,
            minimum_valid_velocity=0.03,
            minimum_velocity_overrides={
                StretchJoint.LIFT: 0.0,
                StretchJoint.ARM_L0: 0.0,
                StretchJoint.GRIPPER_LEFT_FINGER: 0.0,
            },
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
