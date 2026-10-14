from dataclasses import dataclass, field

from typing_extensions import List

from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from giskardpy.model.world_config import WorldWithMobileRobot
from semantic_digital_twin.robots.stretch import Stretch, StretchJoint
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


@dataclass
class StretchVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands the arm, head and drive of Stretch through their velocity controllers.
    """

    @staticmethod
    def velocity_controlled_joint_names() -> List[StretchJoint]:
        """
        The joints driven by the velocity group controller.

        Their order matches the controller's command layout, so it is significant.
        """
        return [
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

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )

        self.sync_robot_parts()

        diff_drive = self.world.get_connections_by_type(DifferentialDrive)[0]
        self.add_base_cmd_velocity(cmd_vel_topic="/stretch/cmd_vel", joint=diff_drive)

        self.add_joint_velocity_group_controller(
            cmd_topic="/joint_velocity_cmd",
            connections=self.velocity_controlled_joint_names(),
            minimum_valid_velocity=0.03,
            minimum_velocity_overrides={
                StretchJoint.LIFT: 0.0,
                StretchJoint.ARM_L0: 0.0,
                StretchJoint.GRIPPER_LEFT_FINGER: 0.0,
            },
        )


@dataclass
class WorldWithStretchConfig(WorldWithMobileRobot):
    """
    A world containing only Stretch, modelled with an omnidirectional base.
    """

    robot_class: type[Stretch] = field(kw_only=True, default=Stretch, init=False)
    """
    The robot class that is applied to the parsed description.
    """


@dataclass
class WorldWithStretchConfigDiffDrive(WorldWithStretchConfig):
    """
    A world containing only Stretch, which drives on a differential base.
    """

    drive_class: type[DifferentialDrive] = field(
        kw_only=True, default=DifferentialDrive, init=False
    )
    """
    The kind of connection between the odometry body and the robot.
    """
