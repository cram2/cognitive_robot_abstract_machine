#!/usr/bin/env python
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import WorldWithPR2Config
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.robots.pr2 import PR2Joint


def main():
    rospy.init_node("giskard")
    rospy.node.declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    robot_description = rospy.node.get_parameter_or("robot_description").value
    giskard = Giskard(
        world_config=WorldWithPR2Config(urdf=robot_description),
        robot_interface_config=StandAloneRobotInterfaceConfig(
            [
                PR2Joint.TORSO_LIFT,
                PR2Joint.HEAD_PAN,
                PR2Joint.HEAD_TILT,
                PR2Joint.RIGHT_SHOULDER_PAN,
                PR2Joint.RIGHT_SHOULDER_LIFT,
                PR2Joint.RIGHT_UPPER_ARM_ROLL,
                PR2Joint.RIGHT_FOREARM_ROLL,
                PR2Joint.RIGHT_ELBOW_FLEX,
                PR2Joint.RIGHT_WRIST_FLEX,
                PR2Joint.RIGHT_WRIST_ROLL,
                PR2Joint.LEFT_SHOULDER_PAN,
                PR2Joint.LEFT_SHOULDER_LIFT,
                PR2Joint.LEFT_UPPER_ARM_ROLL,
                PR2Joint.LEFT_FOREARM_ROLL,
                PR2Joint.LEFT_ELBOW_FLEX,
                PR2Joint.LEFT_WRIST_FLEX,
                PR2Joint.LEFT_WRIST_ROLL,
                "odom_combined_T_base_footprint",
            ]
        ),
        server_config=GiskardServerConfig(
            execution_mode=ExecutionMode.STANDALONE, debug_mode=True
        ),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )
    giskard.live()


if __name__ == "__main__":
    main()
