from dataclasses import dataclass, field

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.model.world_config import WorldWithOmniDriveRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.garmi import Garmi


@dataclass
class WorldWithGarmiConfig(WorldWithOmniDriveRobot):
    """
    World configuration for the GARMI robot.

    Builds a map -> odom_combined -> GARMI kinematic tree using an omni-drive base.
    """

    odom_body_name: PrefixedName = field(
        default_factory=lambda: PrefixedName("odom_combined")
    )
    urdf_view: AbstractRobot = field(kw_only=True, default=Garmi, init=False)


class GarmiStandaloneInterface(StandAloneRobotInterfaceConfig):
    """
    Robot interface configuration for running GARMI in standalone (simulation) mode.

    Registers all hardware-controlled joints: mecanum wheels, lift, head, both FR3 arms, and grippers.
    """

    def __init__(self, drive_joint_name: str = "odom_combined_T_base_link"):
        super().__init__(
            [
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
                "lift_0_lower_joint",
                "lift_0_upper_joint",
                "head_pan_joint",
                "head_tilt_joint",
                "arm_0_fr3_joint1",
                "arm_0_fr3_joint2",
                "arm_0_fr3_joint3",
                "arm_0_fr3_joint4",
                "arm_0_fr3_joint5",
                "arm_0_fr3_joint6",
                "arm_0_fr3_joint7",
                "arm_1_fr3_joint1",
                "arm_1_fr3_joint2",
                "arm_1_fr3_joint3",
                "arm_1_fr3_joint4",
                "arm_1_fr3_joint5",
                "arm_1_fr3_joint6",
                "arm_1_fr3_joint7",
                "arm_0_gripper_fr3_finger_joint1",
                "arm_0_gripper_fr3_finger_joint2",
                "arm_1_gripper_fr3_finger_joint1",
                "arm_1_gripper_fr3_finger_joint2",
                drive_joint_name,
            ]
        )
