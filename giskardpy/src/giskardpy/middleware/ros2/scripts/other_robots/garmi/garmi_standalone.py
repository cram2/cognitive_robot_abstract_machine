#!/usr/bin/env python
import argparse
from threading import Thread

from rclpy import Parameter

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.other_robots.garmi.configs import (
    GarmiStandaloneInterface,
    WorldWithGarmiConfig,
)
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerNode,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig

GARMI_ROOT_LINKS = ["arm_mount_left_link", "map"]
GARMI_TIP_LINKS = [
    "left_fr3_hand_tcp",  # left arm TCP
    "right_fr3_hand_tcp",  # right arm TCP
]


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
    robot_description = load_xacro("package://garmi_description/urdf/garmi.urdf")

    giskard = Giskard(
        world_config=WorldWithGarmiConfig(urdf=robot_description),
        robot_interface_config=GarmiStandaloneInterface(),
        behavior_tree_config=StandAloneBTConfig(debug_mode=True),
        qp_controller_config=QPControllerConfig(target_frequency=20),
    )

    if args.interactive_marker:
        Thread(
            target=lambda: InteractiveMarkerNode(
                root_links=GARMI_ROOT_LINKS,
                tip_links=GARMI_TIP_LINKS,
            ),
            daemon=True,
            name="interactive_marker",
        ).start()

    giskard.live()


if __name__ == "__main__":
    main()
