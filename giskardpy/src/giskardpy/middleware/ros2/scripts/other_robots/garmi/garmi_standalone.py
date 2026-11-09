#!/usr/bin/env python
import argparse

from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.other_robots.garmi.configs import (
    GarmiInteractiveMarkerChain,
)
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerNode,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.robots.garmi import Garmi
from semantic_digital_twin.specifications.robots import RobotSpecification


def main() -> None:
    parser = argparse.ArgumentParser(description="GARMI Giskard standalone controller.")
    parser.add_argument(
        "--interactive-marker",
        action="store_true",
        help="Also start the interactive marker server for Cartesian control via RViz.",
    )
    # parse_known_args ignores ROS 2 arguments (--ros-args ...) that argparse does not know about.
    args, _ = parser.parse_known_args()

    rospy.init_node("giskard")

    giskard = Giskard.standalone(
        world_source=WorldSpecification(
            robots=[RobotSpecification.specify_real_robot(Garmi)]
        ),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )

    if args.interactive_marker:
        InteractiveMarkerNode.start_in_background_thread(
            chains=[chain.value for chain in GarmiInteractiveMarkerChain]
        )

    giskard.live()


if __name__ == "__main__":
    main()
