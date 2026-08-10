from dataclasses import dataclass, field
from typing import List, Optional

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
    RobotInterfaceConfig,
)
from giskardpy.model.world_config import WorldWithFixedRobot
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyJoint
from semantic_digital_twin.robots.robot_parts import AbstractRobot

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


@dataclass
class DAiSyVelocityInterface(RobotInterfaceConfig):
    """
    Commands both arms of DAiSy through their forward velocity controllers.
    """

    def setup(self):
        self.sync_joint_state_topic("/joint_states")
        joints_left = [
            DAiSyJoint.LEFT_SHOULDER_PAN,
            DAiSyJoint.LEFT_SHOULDER_LIFT,
            DAiSyJoint.LEFT_ELBOW,
            DAiSyJoint.LEFT_WRIST_1,
            DAiSyJoint.LEFT_WRIST_2,
            DAiSyJoint.LEFT_WRIST_3,
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/left_forward_velocity_controller/commands",
            connections=joints_left,
        )
        joints_right = [
            DAiSyJoint.RIGHT_SHOULDER_PAN,
            DAiSyJoint.RIGHT_SHOULDER_LIFT,
            DAiSyJoint.RIGHT_ELBOW,
            DAiSyJoint.RIGHT_WRIST_1,
            DAiSyJoint.RIGHT_WRIST_2,
            DAiSyJoint.RIGHT_WRIST_3,
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/right_forward_velocity_controller/commands",
            connections=joints_right,
        )


@dataclass
class WorldWithDaisyConfig(WorldWithFixedRobot):
    """
    A world containing only DAiSy, whose base is fixed to the world root.
    """

    root_name: PrefixedName = field(default=PrefixedName(name="map2"))
    """
    Name of the body DAiSy is attached to.
    """

    urdf_view: AbstractRobot = field(kw_only=True, default=DAiSy, init=False)
    """
    Semantic view that is applied to the parsed urdf.
    """

    def setup_world(self, robot_name: Optional[str] = None) -> None:
        super().setup_world()
        self.robot = self.world.get_semantic_annotations_by_type(DAiSy)[0]


@dataclass
class DaisyStandAloneRobotInterfaceConfig(StandAloneRobotInterfaceConfig):
    """
    Simulates both arms and both grippers of DAiSy without talking to hardware.
    """

    joint_names: List[str] = field(
        init=False,
        default_factory=lambda: [
            DAiSyJoint.LEFT_SHOULDER_PAN,
            DAiSyJoint.LEFT_SHOULDER_LIFT,
            DAiSyJoint.LEFT_ELBOW,
            DAiSyJoint.LEFT_WRIST_1,
            DAiSyJoint.LEFT_WRIST_2,
            DAiSyJoint.LEFT_WRIST_3,
            DAiSyJoint.RIGHT_SHOULDER_PAN,
            DAiSyJoint.RIGHT_SHOULDER_LIFT,
            DAiSyJoint.RIGHT_ELBOW,
            DAiSyJoint.RIGHT_WRIST_1,
            DAiSyJoint.RIGHT_WRIST_2,
            DAiSyJoint.RIGHT_WRIST_3,
            DAiSyJoint.LEFT_GRIPPER_FINGER,
            DAiSyJoint.LEFT_GRIPPER_RIGHT_FINGER,
            DAiSyJoint.RIGHT_GRIPPER_FINGER,
            DAiSyJoint.RIGHT_GRIPPER_RIGHT_FINGER,
        ],
    )
    """
    The arm and gripper joints of DAiSy.
    """
