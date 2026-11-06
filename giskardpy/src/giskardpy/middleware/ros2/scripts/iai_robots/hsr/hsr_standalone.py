#!/usr/bin/env python
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.specifications.robots import RobotSpecification


def create_giskard() -> Giskard:
    """
    Build a Giskard that simulates the HSR, described by the file its robot class
    declares.
    """
    return Giskard.standalone(
        world_source=WorldSpecification(
            robots=[RobotSpecification.specify_real_robot(HSRB)]
        ),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
