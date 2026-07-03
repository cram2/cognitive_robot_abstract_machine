from dataclasses import dataclass, field

from mercurial.namespaces import namespace

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
    RobotInterfaceConfig,
)
from giskardpy.model.world_config import (
    WorldWithDiffDriveRobot,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


@dataclass
class WorldWithTiagoConfig(WorldWithDiffDriveRobot):
    odom_body_name: PrefixedName = PrefixedName("odom_combined")
    urdf_view: AbstractRobot = field(kw_only=True, default=Tiago, init=False)


class TiagoStandaloneInterface(StandAloneRobotInterfaceConfig):

    def __init__(self):
        super().__init__(
            [
                "gripper_left_finger_joint",
                "gripper_right_finger_joint",
                "arm_left_1_joint",
                "arm_left_2_joint",
                "arm_left_3_joint",
                "arm_left_4_joint",
                "arm_left_5_joint",
                "arm_left_6_joint",
                "arm_left_7_joint",
                "arm_right_1_joint",
                "arm_right_2_joint",
                "arm_right_3_joint",
                "arm_right_4_joint",
                "arm_right_5_joint",
                "arm_right_6_joint",
                "arm_right_7_joint",
                "head_1_joint",
                "head_2_joint",
                "torso_lift_joint",
                "wheel_left_joint",
                "wheel_right_joint",
                "odom_combined_T_base_footprint",
            ]
        )


@dataclass
class WorldWithTiagoConfigDiffDrive(WorldWithDiffDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=Tiago, init=False)

    def setup_collision_config(self):
        pass
