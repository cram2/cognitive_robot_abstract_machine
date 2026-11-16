#!/usr/bin/env python
from dataclasses import dataclass

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from rclpy.node import Node
from semantic_digital_twin.adapters.ros.ros_graph import RosGraph
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification


@dataclass(kw_only=True)
class ControllerManagerInterface(ClosedLoopRobotInterfaceConfig):
    """
    Reads and commands a robot through the active controllers of its ros2_control
    controller manager.
    """

    def setup(self) -> None:
        self.discover_interfaces_from_controller_manager()


def create_giskard(node: Node) -> Giskard:
    """
    Build a Giskard that commands the robot described on ``/robot_description`` through
    its controller manager.

    :param node: The node Giskard runs on, which reads the robot description.
    """
    return Giskard(
        world_source=WorldSpecification(
            robots=[
                RobotSpecification.specify_real_robot(
                    MinimalRobot, urdf=RosGraph(node).robot_description()
                )
            ]
        ),
        robot_interface_config=ControllerManagerInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard=create_giskard)


if __name__ == "__main__":
    main()
