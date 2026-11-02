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
from giskardpy.model.world_config import WorldConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.exceptions import DataclassException
from semantic_digital_twin.input_synchronization import InputSynchronizer
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.world import World

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
class WorldWithMinimalRobot(WorldConfig):
    """
    Treats everything in an existing world as the robot.
    """

    def setup_world(self) -> None:
        MinimalRobot.from_world(self.world)


def test_an_interrupt_while_waiting_for_goals_shuts_ros_down(
    init_rospy, mini_world: World
) -> None:
    giskard = Giskard(
        world_config=WorldWithMinimalRobot(world=mini_world),
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
    Raised by a world config that cannot build its world.
    """

    def error_message(self) -> str:
        return "The world cannot be built."

    def suggest_correction(self) -> str:
        return "Check the robot description the world config loads."


@dataclass
class UnreachableRobotInterface(StandAloneRobotInterfaceConfig):
    """
    Fails as soon as Giskard connects to the robot.
    """

    def connect(self) -> Never:
        raise RobotUnreachableError()


@dataclass
class UnbuildableWorld(WorldConfig):
    """
    Fails as soon as Giskard builds the world.
    """

    def setup_world(self) -> Never:
        raise WorldUnbuildableError()


def test_an_interface_that_fails_to_connect_makes_giskard_fail(
    init_rospy, mini_world: World
) -> None:
    giskard = Giskard(
        world_config=WorldWithMinimalRobot(world=mini_world),
        robot_interface_config=UnreachableRobotInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    with pytest.raises(RobotUnreachableError):
        giskard.live()

    assert not rclpy.ok()


def test_a_world_that_fails_to_build_makes_giskard_fail_with_that_error(
    init_rospy,
) -> None:
    giskard = Giskard(
        world_config=UnbuildableWorld(),
        robot_interface_config=StandAloneRobotInterfaceConfig(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    with pytest.raises(WorldUnbuildableError):
        giskard.live()
