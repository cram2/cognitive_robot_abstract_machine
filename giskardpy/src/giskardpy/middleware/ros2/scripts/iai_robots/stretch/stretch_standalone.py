#!/usr/bin/env python
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchStandaloneInterface,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.model.world_config import WorldWithDiffDriveRobot


def main():
    rospy.init_node("giskard")
    robot_description = load_xacro(
        "package://stretch_description/urdf/stretch_description_RE2V0_tool_stretch_dex_wrist.xacro"
    )
    giskard = Giskard(
        world_config=WorldWithDiffDriveRobot(urdf=robot_description),
        robot_interface_config=StretchStandaloneInterface(),
        behavior_tree_config=StandAloneBTConfig(debug_mode=True),
    )
    giskard.live()


if __name__ == "__main__":
    main()
