#!/usr/bin/env python
from datetime import timedelta

from rclpy.node import Node

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.specifications.robots import RobotSpecification


def create_giskard(node: Node) -> Giskard:
    """
    Build a Giskard that simulates Stretch, described by the file its robot class
    declares.

    :param node: The node Giskard runs on, unused because the robot is described by a
        file.
    """
    return Giskard.standalone(
        world_source=WorldSpecification(
            robots=[RobotSpecification.specify_real_robot(Stretch)]
        ),
        qp_controller_config=QPControllerConfig(
            # reproduces the braking of the former prediction_horizon=30
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
