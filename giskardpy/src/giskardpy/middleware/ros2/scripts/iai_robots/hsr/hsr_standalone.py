#!/usr/bin/env python
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs import (
    WorldWithHSRConfig,
    HSRStandaloneInterface,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard


def main():
    rospy.init_node("giskard")
    default_robot_desc = load_xacro(
        "package://hsr_description/robots/hsrb4s.urdf.xacro"
    )
    rospy.node.declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    robot_description = rospy.node.get_parameter_or("robot_description").value
    if robot_description is None:
        robot_description = default_robot_desc
    giskard = Giskard(
        world_config=WorldWithHSRConfig(urdf=robot_description),
        robot_interface_config=HSRStandaloneInterface(),
        behavior_tree_config=StandAloneBTConfig(debug_mode=True),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )
    giskard.live()


if __name__ == "__main__":
    main()
