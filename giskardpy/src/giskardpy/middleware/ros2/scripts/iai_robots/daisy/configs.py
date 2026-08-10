from dataclasses import field
from typing import Optional

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
    RobotInterfaceConfig,
)
from giskardpy.model.world_config import WorldWithFixedRobot
from semantic_digital_twin.robots.daisy import DAiSy

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


class DAiSyVelocityInterface(RobotInterfaceConfig):

    def setup(self):
        self.sync_joint_state_topic("/joint_states")
        joints_left = [
            "left_shoulder_pan_joint",
            "left_shoulder_lift_joint",
            "left_elbow_joint",
            "left_wrist_1_joint",
            "left_wrist_2_joint",
            "left_wrist_3_joint",
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/left_forward_velocity_controller/commands",
            connections=joints_left,
        )
        joints_right = [
            "right_shoulder_pan_joint",
            "right_shoulder_lift_joint",
            "right_elbow_joint",
            "right_wrist_1_joint",
            "right_wrist_2_joint",
            "right_wrist_3_joint",
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/right_forward_velocity_controller/commands",
            connections=joints_right,
        )


class WorldWithDaisyConfig(WorldWithFixedRobot):
    """Minimal Tracy world config analogous to WorldWithPR2Config.

    - Fixed-base robot (no drive joint)
    - Accepts URDF via argument; if not provided, reads from ROS parameter server
    - Applies conservative default motion limits
    """

    def __init__(self, urdf: Optional[str] = None):
        super().__init__(
            urdf=urdf, root_name=PrefixedName(name="map2"), urdf_view=DAiSy
        )

    def setup_world(self, robot_name: Optional[str] = None) -> None:
        super().setup_world()
        self.robot = self.world.get_semantic_annotations_by_type(DAiSy)[0]


class DaisyStandAloneRobotInterfaceConfig(StandAloneRobotInterfaceConfig):
    def __init__(self):
        super().__init__(
            [
                "left_shoulder_pan_joint",
                "left_shoulder_lift_joint",
                "left_elbow_joint",
                "left_wrist_1_joint",
                "left_wrist_2_joint",
                "left_wrist_3_joint",
                "right_shoulder_pan_joint",
                "right_shoulder_lift_joint",
                "right_elbow_joint",
                "right_wrist_1_joint",
                "right_wrist_2_joint",
                "right_wrist_3_joint",
                "left_gripper_finger_joint",
                "left_gripper_right_finger_joint",
                "right_gripper_finger_joint",
                "right_gripper_right_finger_joint",
            ]
        )
