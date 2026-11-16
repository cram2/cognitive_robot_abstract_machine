#!/usr/bin/env python
from rclpy.node import Node

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.specifications.robots import RobotSpecification


def create_giskard(node: Node) -> Giskard:
    """
    Build a Giskard that simulates the PR2, described by the file its robot class
    declares.

    :param node: The node Giskard runs on, unused because the robot is described by a
        file.
    """
    return Giskard.standalone(
        world_source=WorldSpecification(
            robots=[RobotSpecification.specify_real_robot(PR2)]
        ),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard=create_giskard)


if __name__ == "__main__":
    main()
