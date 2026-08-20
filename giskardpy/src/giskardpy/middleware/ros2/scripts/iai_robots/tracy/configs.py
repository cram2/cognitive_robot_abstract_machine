from dataclasses import dataclass, field
from typing import List, Optional

from giskardpy.model.world_config import WorldWithFixedRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tracy import Tracy, TracyJoint


@dataclass
class TracyVelocityInterface(RobotInterfaceConfig):
    """
    Commands both arms of Tracy through their forward velocity controllers.
    """

    def setup(self):
        self.sync_joint_state_topic("/left_arm/joint_states")
        self.sync_joint_state_topic("/right_arm/joint_states")
        self.sync_joint_state_topic("/right_gripper/joint_states")
        self.sync_joint_state_topic("/left_gripper/joint_states")
        joints_left = [
            TracyJoint.LEFT_SHOULDER_PAN,
            TracyJoint.LEFT_SHOULDER_LIFT,
            TracyJoint.LEFT_ELBOW,
            TracyJoint.LEFT_WRIST_1,
            TracyJoint.LEFT_WRIST_2,
            TracyJoint.LEFT_WRIST_3,
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/left_arm/forward_velocity_controller/commands",
            connections=joints_left,
        )
        joints_right = [
            TracyJoint.RIGHT_SHOULDER_PAN,
            TracyJoint.RIGHT_SHOULDER_LIFT,
            TracyJoint.RIGHT_ELBOW,
            TracyJoint.RIGHT_WRIST_1,
            TracyJoint.RIGHT_WRIST_2,
            TracyJoint.RIGHT_WRIST_3,
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/right_arm/forward_velocity_controller/commands",
            connections=joints_right,
        )


@dataclass
class WorldWithTracyConfig(WorldWithFixedRobot):
    """
    A world containing only Tracy, whose base is fixed to the world root.
    """

    root_name: PrefixedName = field(default=PrefixedName("map2"))
    """
    Name of the body Tracy is attached to.
    """

    urdf_view: AbstractRobot = field(kw_only=True, default=Tracy, init=False)
    """
    Semantic view that is applied to the parsed urdf.
    """

    def setup_world(self, robot_name: Optional[str] = None) -> None:
        super().setup_world()
        self.robot = self.world.get_semantic_annotations_by_type(Tracy)[0]


# class TracyCollisionAvoidanceConfig(LoadSelfCollisionMatrixConfig):
#     def __init__(self, collision_checker: CollisionCheckerLib = CollisionCheckerLib.bpb):
#         super().__init__('package://giskardpy_ros/self_collision_matrices/iai/tracy.srdf',
#                          collision_checker)


@dataclass
class TracyStandAloneRobotInterfaceConfig(StandAloneRobotInterfaceConfig):
    """
    Simulates both arms of Tracy without talking to hardware.
    """

    joint_names: List[str] = field(
        init=False,
        default_factory=lambda: [
            TracyJoint.LEFT_SHOULDER_PAN,
            TracyJoint.LEFT_SHOULDER_LIFT,
            TracyJoint.LEFT_ELBOW,
            TracyJoint.LEFT_WRIST_1,
            TracyJoint.LEFT_WRIST_2,
            TracyJoint.LEFT_WRIST_3,
            TracyJoint.RIGHT_SHOULDER_PAN,
            TracyJoint.RIGHT_SHOULDER_LIFT,
            TracyJoint.RIGHT_ELBOW,
            TracyJoint.RIGHT_WRIST_1,
            TracyJoint.RIGHT_WRIST_2,
            TracyJoint.RIGHT_WRIST_3,
        ],
    )
    """
    The arm joints of Tracy.
    """
