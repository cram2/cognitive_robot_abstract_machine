from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithOmniDriveRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    Connection6DoF,
)


@dataclass
class WorldWithHSRConfig(WorldWithOmniDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=HSRB, init=False)


@dataclass
class HSRStandaloneInterface(RobotInterfaceConfig):
    """
    Simulates the arm, head and drive of the HSR without talking to hardware.
    """

    def setup(self):
        self.register_controlled_joints(
            [
                HSRBJoint.ARM_FLEX,
                HSRBJoint.ARM_LIFT,
                HSRBJoint.ARM_ROLL,
                HSRBJoint.HEAD_PAN,
                HSRBJoint.HEAD_TILT,
                HSRBJoint.WRIST_FLEX,
                HSRBJoint.WRIST_ROLL,
                self.world.get_connections_by_type(OmniDrive)[0].name,
            ]
        )


@dataclass
class HSRVelocityInterface(RobotInterfaceConfig):
    """
    Commands the arm, head and drive of the HSR through their velocity controllers.
    """

    def setup(self):
        self.sync_6dof_joint_with_tf_frame(
            joint=self.world.get_connections_by_type(Connection6DoF)[0],
            tf_parent_frame="map",
            tf_child_frame="odom",
        )

        omni_drive = self.world.get_connections_by_type(OmniDrive)[0]
        self.sync_odometry_topic(
            "/laser_odom",
            omni_drive,
        )

        self.add_base_cmd_velocity(
            cmd_vel_topic="/omni_base_controller/cmd_vel", joint=omni_drive
        )

        self.sync_joint_state_topic("/joint_states")
        joints_left = [
            HSRBJoint.ARM_FLEX,
            HSRBJoint.ARM_LIFT,
            HSRBJoint.ARM_ROLL,
            HSRBJoint.WRIST_FLEX,
            HSRBJoint.WRIST_ROLL,
            HSRBJoint.HEAD_PAN,
            HSRBJoint.HEAD_TILT,
        ]
        self.add_joint_velocity_group_controller(
            cmd_topic="/realtime_body_controller_real/command", connections=joints_left
        )
