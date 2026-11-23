#!/usr/bin/env python
"""
Launch Giskard with closed-loop velocity control for the TIAGo robot.
"""

from __future__ import annotations

from datetime import timedelta

from rclpy.node import Node
import re
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tiago.configs import (
    TiagoVelocityInterface,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.ros.ros_graph import RosGraph
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification


def create_giskard(node: Node) -> Giskard:
    """
    Build a Giskard that commands TIAGo, described by what the robot publishes on
    ``/robot_description``.

    :param node: The node Giskard runs on, which reads the robot description.
    :return: Configured closed-loop Giskard instance.
    """
    urdf = re.sub(
        r"(meshes/arm/[^\"<]+)\.dae",
        r"\1.stl",
        RosGraph(node).robot_description(),
    )
    return Giskard(
        world_source=WorldSpecification(
            robots=[RobotSpecification.specify_real_robot(Tiago, urdf=urdf)]
        ),
        robot_interface_config=TiagoVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=25,
            braking_time=timedelta(seconds=1.15),
        ),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard=create_giskard)


if __name__ == "__main__":
    main()
