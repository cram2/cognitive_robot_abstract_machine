from dataclasses import dataclass

from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from semantic_digital_twin.robots.tracy import (
    TracyController,
    TracyControllerManager,
)


@dataclass(kw_only=True)
class TracyVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands both arms of Tracy through their forward velocity controllers.
    """

    def setup(self):
        self.sync_robot_parts()
        for controller_manager in TracyControllerManager:
            self.discover_interfaces_from_controller_manager(
                controller_manager_name=controller_manager,
                whitelist=[TracyController.FORWARD_VELOCITY],
            )
