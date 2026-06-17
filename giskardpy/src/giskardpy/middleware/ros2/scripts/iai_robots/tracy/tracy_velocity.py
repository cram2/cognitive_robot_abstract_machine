from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    WorldWithTracyConfig,
    TracyVelocityInterface,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from rclpy import Parameter
from rclpy.exceptions import ParameterUninitializedException

from giskardpy.utils import objgraph_debug


def main():
    rospy.init_node("giskard")
    try:
        rospy.node.declare_parameters(
            namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
        )
        robot_description = rospy.node.get_parameter_or("robot_description").value
        if robot_description is None:
            robot_description = load_xacro(
                "package://iai_tracy_description/urdf/tracy.urdf.xacro"
            )
    except ParameterUninitializedException as e:
        robot_description = load_xacro(
            "package://iai_tracy_description/urdf/tracy.urdf.xacro"
        )

    giskard = Giskard(
        world_config=WorldWithTracyConfig(urdf=robot_description),
        robot_interface_config=TracyVelocityInterface(),
        behavior_tree_config=ClosedLoopBTConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=80, prediction_horizon=50
        ),
    )
    objgraph_debug.report_growth(label="giskard startup baseline")
    objgraph_debug.start_periodic()

    giskard.live()


if __name__ == "__main__":
    main()
