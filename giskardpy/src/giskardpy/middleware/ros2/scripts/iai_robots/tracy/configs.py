from typing import Optional, TYPE_CHECKING

from giskardpy.model.world_config import WorldWithFixedRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy

if TYPE_CHECKING:
    pass


class TracyVelocityInterface(RobotInterfaceConfig):

    def setup(self):
        self.sync_joint_state_topic("/left_arm/joint_states")
        self.sync_joint_state_topic("/right_arm/joint_states")
        self.sync_joint_state_topic("/right_gripper/joint_states")
        self.sync_joint_state_topic("/left_gripper/joint_states")
        joints_left = [
            "left_shoulder_pan_joint",
            "left_shoulder_lift_joint",
            "left_elbow_joint",
            "left_wrist_1_joint",
            "left_wrist_2_joint",
            "left_wrist_3_joint",
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/left_arm/forward_velocity_controller/commands",
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
            cmd_topic="/right_arm/forward_velocity_controller/commands",
            connections=joints_right,
        )


class WorldWithTracyConfig(WorldWithFixedRobot):
    """Minimal Tracy world config analogous to WorldWithPR2Config.

    - Fixed-base robot (no drive joint)
    - Accepts URDF via argument; if not provided, reads from ROS parameter server
    - Applies conservative default motion limits
    """

    def __init__(self, urdf: Optional[str] = None):
        super().__init__(urdf=urdf, root_name=PrefixedName("map2"), urdf_view=Tracy)

    def setup_world(self, robot_name: Optional[str] = None) -> None:
        super().setup_world()
        self.robot = self.world.get_semantic_annotations_by_type(Tracy)[0]


# class TracyCollisionAvoidanceConfig(LoadSelfCollisionMatrixConfig):
#     def __init__(self, collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb):
#         super().__init__('package://giskardpy_ros/self_collision_matrices/iai/tracy.srdf',
#                          collision_checker)


class TracyJointTrajServerMujocoInterface(RobotInterfaceConfig):
    def setup(self):
        self.sync_joint_state_topic("joint_states")
        self.add_follow_joint_trajectory_server(
            namespace="/left_arm/scaled_pos_joint_traj_controller_left"
        )
        self.add_follow_joint_trajectory_server(
            namespace="/right_arm/scaled_pos_joint_traj_controller_right"
        )


class TracyStandAloneRobotInterfaceConfig(StandAloneRobotInterfaceConfig):
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
            ]
        )
