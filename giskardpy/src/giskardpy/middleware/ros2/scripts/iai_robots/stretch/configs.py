from dataclasses import dataclass, field

from typing_extensions import List, Union

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
    RobotInterfaceConfig,
)
from giskardpy.model.world_config import (
    WorldWithOmniDriveRobot,
    WorldWithDiffDriveRobot,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


class StretchStandaloneInterface(StandAloneRobotInterfaceConfig):
    """
    Standalone interface controlling Stretch's base drive and all of its actuated
    joints.
    """

    def __init__(self):
        super().__init__(
            [
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

    def controlled_joint_names(self, world: World) -> List[Union[str, PrefixedName]]:
        """
        The joints to control, with the base drive resolved from ``world``.

        :param world: The world holding the robot and its base drive connection.
        """
        drive = world.get_connections_by_type(DifferentialDrive)[0]
        return [drive.name, *self.joint_names]

    def setup(self):
        self.register_controlled_joints(self.controlled_joint_names(self.world))


class StretchVelocityInterface(RobotInterfaceConfig):
    """
    Interface for the real robot, driving it through velocity commands and reading its
    state back from the hardware's own topics.
    """

    def velocity_controlled_joint_names(self) -> List[str]:
        """
        The joints driven by the velocity group controller.

        Their order matches the controller's command layout, so it is significant.
        """
        return [
            "joint_arm_l0",  # 0
            "joint_lift",  # 1
            "joint_wrist_yaw",  # 2
            "joint_wrist_pitch",  # 3
            "joint_wrist_roll",  # 4
            "joint_head_pan",  # 5
            "joint_head_tilt",  # 6
            "joint_gripper_finger_left",  # 7
            "joint_right_wheel",  # 8
            "joint_left_wheel",  # 9
        ]

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
        self.add_joint_velocity_group_controller(
            cmd_topic="/joint_velocity_cmd",
            connections=self.velocity_controlled_joint_names(),
            minimum_valid_velocity=0.03,
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
