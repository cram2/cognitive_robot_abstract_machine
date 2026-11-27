#!/usr/bin/env python
"""
Launch Giskard with closed-loop velocity control for the PR2.
"""

from __future__ import annotations

from datetime import timedelta

from rclpy.node import Node

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import (
    PR2VelocityInterface,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.ros.ros_graph import RosGraph
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification


def create_giskard(node: Node) -> Giskard:
    """
    Build a Giskard that commands the PR2, described by what the robot publishes on
    ``/robot_description``.

    :param node: The node Giskard runs on, which reads the robot description.
    :return: Configured closed-loop Giskard instance.
    """
    return Giskard(
        world_source=WorldSpecification(
            robots=[
                RobotSpecification.specify_real_robot(
                    PR2, urdf=RosGraph(node).robot_description()
                )
            ]
        ),
        robot_interface_config=PR2VelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(
            # below the 100 Hz of /joint_states; the braking keeps the stops slower than
            # the shoulders' resonance between 1 and 2 Hz
            target_frequency=80,
            braking_time=timedelta(seconds=0.5),
        ),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard=create_giskard)


if __name__ == "__main__":
    main()
