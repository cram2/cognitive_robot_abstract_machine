from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    WorldWithTracyConfig,
    TracyVelocityInterface,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from rclpy import Parameter
from rclpy.exceptions import ParameterUninitializedException


def main():
    rospy.init_node("giskard")
    try:
        rospy.node.declare_parameters(
            namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
        )
        robot_description = rospy.node.get_parameter_or("robot_description").value
    except ParameterUninitializedException as e:
        robot_description = load_xacro(
            "package://iai_tracy_description/urdf/tracy.urdf.xacro"
        )
    giskard = Giskard(
        world_config=WorldWithTracyConfig(urdf=robot_description),
        robot_interface_config=TracyVelocityInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
        qp_controller_config=QPControllerConfig(
            target_frequency=80, prediction_horizon=30
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
