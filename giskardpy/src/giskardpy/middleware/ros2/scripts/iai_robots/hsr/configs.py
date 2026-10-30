from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithMobileRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    Connection6DoF,
)


@dataclass
class WorldWithHSRConfig(WorldWithMobileRobot):
    """
    A world containing only the HSR, which drives on an omnidirectional base.
    """

    robot_class: type[HSRB] = field(kw_only=True, default=HSRB, init=False)
    """
    The robot class that is applied to the parsed description.
    """


@dataclass
class HSRVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands the arm, head and drive of the HSR through their velocity controllers.
    """

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )

        self.sync_robot_parts()

        omni_drive = self.world.get_connections_by_type(OmniDrive)[0]
        self.add_drive_velocity_command_topic(
            command_topic="/omni_base_controller/cmd_vel", drive=omni_drive
        )

        joints_left = [
            HSRBJoint.ARM_FLEX,
            HSRBJoint.ARM_LIFT,
            HSRBJoint.ARM_ROLL,
            HSRBJoint.WRIST_FLEX,
            HSRBJoint.WRIST_ROLL,
            HSRBJoint.HEAD_PAN,
            HSRBJoint.HEAD_TILT,
        ]
        self.add_joint_group_velocity_command_topic(
            command_topic="/realtime_body_controller_real/command",
            connections=joints_left,
        )
