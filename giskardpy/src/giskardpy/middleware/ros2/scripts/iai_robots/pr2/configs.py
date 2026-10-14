from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithMobileRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2


@dataclass
class WorldWithPR2Config(WorldWithMobileRobot):
    """
    A world containing only the PR2, which drives on an omnidirectional base.
    """

    odom_body_name: PrefixedName = PrefixedName("odom_combined")
    """
    Name of the body the robot drives relative to.
    """

    robot_class: type[PR2] = field(kw_only=True, default=PR2, init=False)
    """
    The robot class that is applied to the parsed description.
    """


@dataclass
class PR2VelocityMujocoInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands a PR2 simulated in mujoco through its controller manager.
    """

    map_name: str = "map"
    """
    Name of the frame the localization is expressed in.
    """

    localization_joint_name: str = "localization"
    """
    Name of the 6 degree of freedom connection carrying the localization.
    """

    odom_link_name: str = "odom_combined"
    """
    Name of the body the drive moves relative to.
    """

    def setup(self):
        self.discover_interfaces_from_controller_manager()
        self.sync_robot_part(self.robot.mobile_base)
        self.add_base_cmd_velocity(cmd_vel_topic="/cmd_vel")
