#!/usr/bin/env python
import argparse

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.other_robots.garmi.configs import (
    GARMI_INTERACTIVE_MARKER_ROOT_LINKS,
    GARMI_INTERACTIVE_MARKER_TIP_LINKS,
    GarmiVelocityInterface,
    WorldWithGarmiConfig,
)
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerNode,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="GARMI Giskard velocity controller.")
    parser.add_argument(
        "--interactive-marker",
        action="store_true",
        help="Also start the interactive marker server for Cartesian control via RViz.",
    )
    # parse_known_args ignores ROS 2 arguments (--ros-args ...) that argparse does not know about.
    args, _ = parser.parse_known_args()

    rospy.init_node("giskard")
    robot_description = load_xacro("package://garmi_description/urdf/garmi.urdf")

    giskard = Giskard(
        world_config=WorldWithGarmiConfig(urdf=robot_description),
        robot_interface_config=GarmiVelocityInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
        qp_controller_config=QPControllerConfig(
            target_frequency=20, prediction_horizon=15
        ),
    )

    if args.interactive_marker:
        InteractiveMarkerNode.start_in_background_thread(
            root_links=GARMI_INTERACTIVE_MARKER_ROOT_LINKS,
            tip_links=GARMI_INTERACTIVE_MARKER_TIP_LINKS,
        )

    giskard.live()


if __name__ == "__main__":
    main()
