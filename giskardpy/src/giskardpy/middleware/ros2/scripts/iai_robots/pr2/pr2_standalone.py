#!/usr/bin/env python
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import WorldWithPR2Config
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2 import rospy


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
                "torso_lift_joint",
                "head_pan_joint",
                "head_tilt_joint",
                "r_shoulder_pan_joint",
                "r_shoulder_lift_joint",
                "r_upper_arm_roll_joint",
                "r_forearm_roll_joint",
                "r_elbow_flex_joint",
                "r_wrist_flex_joint",
                "r_wrist_roll_joint",
                "l_shoulder_pan_joint",
                "l_shoulder_lift_joint",
                "l_upper_arm_roll_joint",
                "l_forearm_roll_joint",
                "l_elbow_flex_joint",
                "l_wrist_flex_joint",
                "l_wrist_roll_joint",
                "odom_combined_T_base_footprint",
            ]
        ),
        behavior_tree_config=StandAloneBTConfig(debug_mode=True),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )
    giskard.live()


if __name__ == "__main__":
    main()
