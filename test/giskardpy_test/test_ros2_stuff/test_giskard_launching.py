"""
Coverage for what the launcher scripts of the robots share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import rclpy

from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import WorldWithPR2Config
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.model.exceptions import RobotDescriptionMissingError
from giskardpy.model.world_config import WorldWithFixedRobot, WorldWithRobot
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world import World

from .test_giskard_shutdown import (
    InterruptedWhileIdleInterface,
    WorldWithMinimalRobot,
)

# %% describing the robot of a world


def body_names_after_setup(world_config: WorldWithRobot) -> set[PrefixedName]:
    """
    :return: The names of the bodies of the world the config builds.
    """
    with world_config.world.modify_world():
        world_config.setup_world()
    return {body.name for body in world_config.world.bodies}


def test_a_world_without_a_urdf_holds_the_robot_its_robot_class_declares() -> None:
    described_by_robot_class = body_names_after_setup(WorldWithPR2Config())

    assert described_by_robot_class == body_names_after_setup(
        WorldWithPR2Config(urdf=load_xacro(PR2.get_ros_file_path()))
    )


def test_a_world_without_a_urdf_needs_a_robot_class_that_declares_a_description() -> (
    None
):
    world_config = WorldWithFixedRobot()

    with pytest.raises(RobotDescriptionMissingError) as raised:
        body_names_after_setup(world_config)

    assert raised.value.robot_class is world_config.robot_class


# %% building a Giskard that simulates


def test_a_standalone_giskard_simulates_the_robot_of_its_world() -> None:
    world_config = WorldWithPR2Config()
    qp_controller_config = QPControllerConfig(target_frequency=20)

    giskard = Giskard.standalone(
        world_config=world_config, qp_controller_config=qp_controller_config
    )

    assert giskard.world_config is world_config
    assert giskard.qp_controller_config is qp_controller_config
    assert type(giskard.robot_interface_config) is StandAloneRobotInterfaceConfig
    assert giskard.server_config == GiskardServerConfig(debug_mode=True)


# %% launching a Giskard


@dataclass
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
    The name of the ros node at each time a Giskard was built.
    """

    def __call__(self) -> Giskard:
        self.node_names_when_built.append(rospy.get_node().get_name())
        return Giskard(
            world_config=WorldWithMinimalRobot(world=self.world),
            robot_interface_config=InterruptedWhileIdleInterface(),
            server_config=GiskardServerConfig(),
            qp_controller_config=QPControllerConfig(target_frequency=50),
        )


@pytest.fixture()
def shut_ros_down_afterwards():
    """
    Shuts down the ros node a launched Giskard leaves behind.
    """
    if rospy is None:
        pytest.skip("ROS not installed")
    yield
    rospy.shutdown()


def test_launching_builds_giskard_on_its_node_and_runs_it_until_shutdown(
    shut_ros_down_afterwards, mini_world: World
) -> None:
    create_giskard = GiskardBuiltOnItsNode(world=mini_world)
    node_name = "launched_giskard"

    Giskard.launch(create_giskard, node_name=node_name)

    assert create_giskard.node_names_when_built == [node_name]
    assert not rclpy.ok()
