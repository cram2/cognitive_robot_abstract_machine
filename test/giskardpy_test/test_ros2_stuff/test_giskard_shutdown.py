import signal
from dataclasses import dataclass

import pytest
from typing_extensions import Never
import rclpy

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.exceptions import DataclassException
from semantic_digital_twin.input_synchronization import InputSynchronizer
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_source import WorldSource

# %% being interrupted while waiting for goals


@dataclass
class InterruptingInput(InputSynchronizer):
    """
    Asks the process to shut down as soon as it is read.
    """

    def apply(self) -> bool:
        signal.raise_signal(signal.SIGINT)
        return False


@dataclass
class InterruptedWhileIdleInterface(StandAloneRobotInterfaceConfig):
    """
    Lets the first idle cycle of Giskard be interrupted.
    """

    def connect(self) -> None:
        self.motion_server.inputs.add_input(InterruptingInput(world=self.world))


@dataclass
class WorldWithMinimalRobot(WorldSource):
    """
    Hands out an existing world, treating everything in it as the robot.
    """

    world: World
    """
    The world handed out.
    """

    def create_world(self) -> World:
        MinimalRobot.from_world(self.world)
        return self.world


def test_an_interrupt_while_waiting_for_goals_shuts_ros_down(
    init_background_node, mini_world: World
) -> None:
    giskard = Giskard(
        world_source=WorldWithMinimalRobot(world=mini_world),
        robot_interface_config=InterruptedWhileIdleInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    giskard.live()

    assert not rclpy.ok()


# %% failing while starting


@dataclass
class RobotUnreachableError(DataclassException):
    """
    Raised by an interface that cannot connect to its robot.
    """

    def error_message(self) -> str:
        return "The robot cannot be reached."

    def suggest_correction(self) -> str:
        return "Start the robot before Giskard."


@dataclass
class WorldUnbuildableError(DataclassException):
    """
    Raised by a world source that cannot create its world.
    """

    def error_message(self) -> str:
        return "The world cannot be built."

    def suggest_correction(self) -> str:
        return "Check the robot description the world source loads."


@dataclass
class UnreachableRobotInterface(StandAloneRobotInterfaceConfig):
    """
    Fails as soon as Giskard connects to the robot.
    """

    def connect(self) -> Never:
        raise RobotUnreachableError()


@dataclass
class UnbuildableWorld(WorldSource):
    """
    Fails as soon as Giskard creates the world.
    """

    def create_world(self) -> Never:
        raise WorldUnbuildableError()


def test_an_interface_that_fails_to_connect_makes_giskard_fail(
    init_background_node, mini_world: World
) -> None:
    giskard = Giskard(
        world_source=WorldWithMinimalRobot(world=mini_world),
        robot_interface_config=UnreachableRobotInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    with pytest.raises(RobotUnreachableError):
        giskard.live()

    assert not rclpy.ok()


def test_a_world_that_fails_to_build_makes_giskard_fail_with_that_error(
    init_background_node,
) -> None:
    giskard = Giskard(
        world_source=UnbuildableWorld(),
        robot_interface_config=StandAloneRobotInterfaceConfig(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    with pytest.raises(WorldUnbuildableError):
        giskard.live()
