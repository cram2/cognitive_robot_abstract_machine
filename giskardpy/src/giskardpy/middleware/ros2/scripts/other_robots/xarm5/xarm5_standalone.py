from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.scripts.other_robots.xarm5.configs import (
    WorldWithXArm5Config,
    XArm5StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from semantic_digital_twin.robots.xarm5 import XArm5


def main():
    rospy.init_node("giskard")
    rospy.node.declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    robot_description = load_xacro(XArm5.get_ros_file_path())

    giskard = Giskard(
        world_config=WorldWithXArm5Config(urdf=robot_description),
        robot_interface_config=XArm5StandAloneRobotInterfaceConfig(),
        behavior_tree_config=StandAloneBTConfig(debug_mode=True),
        qp_controller_config=QPControllerConfig(target_frequency=33),
    )
    giskard.live()


if __name__ == "__main__":
    main()
