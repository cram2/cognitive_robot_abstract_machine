#!/usr/bin/env python
from dataclasses import dataclass

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.ros2_interface import get_robot_description
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.model.world_config import WorldWithFixedRobot
from giskardpy.qp.qp_controller_config import QPControllerConfig


@dataclass
class ControllerManagerInterface(ClosedLoopRobotInterfaceConfig):
    """
    Reads and commands a robot through the active controllers of its ros2_control
    controller manager.
    """

    def setup(self) -> None:
        self.discover_interfaces_from_controller_manager()


def create_giskard() -> Giskard:
    """
    Build a Giskard that commands the robot described on ``/robot_description`` through
    its controller manager.

    .. note:: The ros node has to be initialized first, because the robot description is
        read from its topic.
    """
    return Giskard(
        world_config=WorldWithFixedRobot(urdf=get_robot_description()),
        robot_interface_config=ControllerManagerInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )


def main() -> None:
    """
    Run the Giskard :func:`create_giskard` builds as a ros node until ROS shuts down.
    """
    Giskard.launch(create_giskard)


if __name__ == "__main__":
    main()
