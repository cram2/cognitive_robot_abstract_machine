from dataclasses import dataclass
from enum import Enum

from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    RootTipPair,
)
from semantic_digital_twin.robots.daisy import DAiSyController


class DAiSyInteractiveMarkerChain(Enum):
    """
    The kinematic chains controllable via interactive markers.
    """

    LEFT_ARM = RootTipPair(root_link="map", tip_link="left_gripper_tool_frame")
    RIGHT_ARM = RootTipPair(root_link="map", tip_link="right_gripper_tool_frame")


@dataclass(kw_only=True)
class DAiSyVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands both arms of DAiSy through their forward velocity controllers.
    """

    def setup(self):
        self.sync_robot_parts()
        self.discover_interfaces_from_controller_manager(
            whitelist=[
                DAiSyController.LEFT_FORWARD_VELOCITY,
                DAiSyController.RIGHT_FORWARD_VELOCITY,
            ]
        )
