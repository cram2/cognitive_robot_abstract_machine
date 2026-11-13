from dataclasses import dataclass


from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from semantic_digital_twin.robots.stretch import StretchJoint
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    DifferentialDrive,
)


@dataclass(kw_only=True)
class StretchVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands the arm, head and drive of Stretch through their velocity controllers.
    """

    @staticmethod
    def velocity_controlled_joint_names() -> list[StretchJoint]:
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
            tf_child_frame=self.robot.mobile_base.odometry_body_name,
        )

        self.sync_robot_parts()

        diff_drive = self.world.get_connections_by_type(DifferentialDrive)[0]
        self.add_drive_velocity_command_topic(
            command_topic="/stretch/cmd_vel", drive=diff_drive
        )

        self.add_joint_group_velocity_command_topic(
            command_topic="/joint_velocity_cmd",
            connections=self.velocity_controlled_joint_names(),
            minimum_valid_velocity=0.03,
            minimum_velocity_overrides={
                StretchJoint.LIFT: 0.0,
                StretchJoint.ARM_L0: 0.0,
                StretchJoint.GRIPPER_LEFT_FINGER: 0.0,
            },
        )
