#!/usr/bin/env python
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.scripts.iai_robots.tiago.configs import (
    WorldWithTiagoConfig,
    TiagoStandaloneInterface,
)
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.utils.utils import load_xacro
from semantic_digital_twin.adapters.ros import world_synchronizer


def main():
    rospy.init_node("giskard")
    robot_description = load_xacro(
        "package://iai_tiago_description/urdf/tiago_from_our_robot.urdf"
    )
    giskard = Giskard(
        world_config=WorldWithTiagoConfig(urdf=robot_description),
        robot_interface_config=TiagoStandaloneInterface(),
        behavior_tree_config=StandAloneBTConfig(debug_mode=True),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )
    giskard.live()


if __name__ == "__main__":
    main()
