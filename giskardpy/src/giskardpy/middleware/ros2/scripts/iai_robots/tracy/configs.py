from dataclasses import dataclass, field

from giskardpy.model.world_config import WorldWithFixedRobot
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import (
    Tracy,
    TracyController,
    TracyControllerManager,
)


@dataclass
class TracyVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands both arms of Tracy through their forward velocity controllers.
    """

    def setup(self):
        self.sync_robot_parts()
        for controller_manager in TracyControllerManager:
            self.discover_interfaces_from_controller_manager(
                controller_manager, whitelist=[TracyController.FORWARD_VELOCITY]
            )


@dataclass
class WorldWithTracyConfig(WorldWithFixedRobot):
    """
    A world containing only Tracy, whose base is fixed to the world root.
    """

    root_name: PrefixedName = field(default=PrefixedName("map2"))
    """
    Name of the body Tracy is attached to.
    """

    robot_class: type[Tracy] = field(kw_only=True, default=Tracy, init=False)
    """
    The robot class that is applied to the parsed description.
    """
