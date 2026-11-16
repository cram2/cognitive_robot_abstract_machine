"""
Coverage for what the launcher scripts of the robots share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import rclpy
from rclpy.node import Node

from semantic_digital_twin.adapters.ros import background_node
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World

from .test_giskard_shutdown import (
    InterruptedWhileIdleInterface,
    WorldWithMinimalRobot,
)

# %% building a Giskard that simulates


def test_a_standalone_giskard_simulates_the_robot_of_its_world(
    pr2_world_specification,
) -> None:
    qp_controller_config = QPControllerConfig(target_frequency=20)

    giskard = Giskard.standalone(
        world_source=pr2_world_specification, qp_controller_config=qp_controller_config
    )

    assert giskard.world_source is pr2_world_specification
    assert giskard.qp_controller_config is qp_controller_config
    assert type(giskard.robot_interface_config) is StandAloneRobotInterfaceConfig
    assert giskard.server_config == GiskardServerConfig(debug_mode=True)


# %% launching a Giskard


@dataclass(kw_only=True)
class GiskardBuiltOnItsNode:
    """
    Builds a Giskard that is interrupted as soon as it waits for goals, and remembers
    the ros node it was built on.
    """

    world: World
    """
    The world the built Giskard controls.
    """

    node_names_when_built: list[str] = field(default_factory=list)
    """
    The name of the ros node each Giskard was built on.
    """

    def __call__(self, node: Node) -> Giskard:
        self.node_names_when_built.append(node.get_name())
        return Giskard(
            world_source=WorldWithMinimalRobot(world=self.world),
            robot_interface_config=InterruptedWhileIdleInterface(),
            server_config=GiskardServerConfig(),
            qp_controller_config=QPControllerConfig(target_frequency=50),
        )


@pytest.fixture()
def shut_ros_down_afterwards():
    """
    Shuts down the ros node a launched Giskard leaves behind.
    """
    if background_node is None:
        pytest.skip("ROS not installed")
    yield
    background_node.shutdown()


def test_launching_builds_giskard_on_its_node_and_runs_it_until_shutdown(
    shut_ros_down_afterwards, mini_world: World
) -> None:
    create_giskard = GiskardBuiltOnItsNode(world=mini_world)
    node_name = "launched_giskard"

    Giskard.launch(create_giskard=create_giskard, node_name=node_name)

    assert create_giskard.node_names_when_built == [node_name]
    assert not rclpy.ok()
