from dataclasses import dataclass, field

from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from giskardpy.model.world_config import WorldWithFixedRobot
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyController

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


@dataclass
class DAiSyVelocityInterface(ClosedLoopRobotInterfaceConfig):
    """
    Commands both arms of DAiSy through their forward velocity controllers.
    """

    def setup(self):
        self.sync_robot_parts()
        self.add_velocity_controllers_from_controller_manager(
            whitelist=[
                DAiSyController.LEFT_FORWARD_VELOCITY,
                DAiSyController.RIGHT_FORWARD_VELOCITY,
            ]
        )


@dataclass
class WorldWithDaisyConfig(WorldWithFixedRobot):
    """
    A world containing only DAiSy, whose base is fixed to the world root.
    """

    root_name: PrefixedName = field(default=PrefixedName(name="map2"))
    """
    Name of the body DAiSy is attached to.
    """

    robot_class: type[DAiSy] = field(kw_only=True, default=DAiSy, init=False)
    """
    The robot class that is applied to the parsed description.
    """
