#!/usr/bin/env python
"""
Launch Giskard in standalone kinematic simulation mode for the TIAGo robot.
"""

from __future__ import annotations

from datetime import timedelta

from rclpy.node import Node

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification


def create_giskard(node: Node) -> Giskard:
    """
    Build a standalone Giskard instance configured for the TIAGo robot.

    :param node: The ROS 2 node Giskard runs on.
    :return: Configured standalone Giskard instance.
    """
    return Giskard.standalone(
        world_source=WorldSpecification(
            robots=[RobotSpecification.specify_real_robot(Tiago)]
        ),
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    )


def main() -> None:
    """
    Run the Giskard built by :func:`create_giskard` as a ROS 2 node until shutdown.
    """
    Giskard.launch(create_giskard=create_giskard)


if __name__ == "__main__":
    main()
