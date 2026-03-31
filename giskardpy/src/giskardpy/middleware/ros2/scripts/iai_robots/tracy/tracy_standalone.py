from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    WorldWithTracyConfig,
    TracyStandAloneRobotInterfaceConfig,
)
from rclpy import Parameter

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard


def main():
    rospy.init_node("giskard")
    rospy.node.declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    robot_description = rospy.node.get_parameter_or("robot_description").value
    # robot_description = load_xacro("package://iai_tracy_description/urdf/tracy.urdf.xacro")

    giskard = Giskard(
        world_config=WorldWithTracyConfig(urdf=robot_description),
        robot_interface_config=TracyStandAloneRobotInterfaceConfig(),
        behavior_tree_config=StandAloneBTConfig(debug_mode=True),
        qp_controller_config=QPControllerConfig(target_frequency=33),
    )
    giskard.live()


if __name__ == "__main__":
    main()
