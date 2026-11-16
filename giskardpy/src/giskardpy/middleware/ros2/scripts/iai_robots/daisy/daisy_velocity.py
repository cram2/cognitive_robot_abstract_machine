from datetime import timedelta

from giskardpy.middleware.ros2.giskard import Giskard
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.ros_graph import RosGraph
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    DAiSyVelocityInterface,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.robots.daisy import DAiSy


def create_giskard(node: Node) -> Giskard:
    """
    Build a Giskard that commands DAiSy, described by what the robot publishes on
    ``/robot_description``.

    :param node: The node Giskard runs on, which reads the robot description.
    """
    return Giskard(
        world_source=WorldSpecification(
            robots=[
                RobotSpecification.specify_real_robot(
                    DAiSy, urdf=RosGraph(node).robot_description()
                )
            ]
        ),
        robot_interface_config=DAiSyVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(
            # reproduces the braking of the former prediction_horizon=30
            target_frequency=80,
            braking_time=timedelta(seconds=0.36),
        ),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard=create_giskard)


if __name__ == "__main__":
    main()
