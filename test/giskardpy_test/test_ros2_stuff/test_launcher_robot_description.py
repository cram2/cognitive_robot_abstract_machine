"""
Coverage for where the launcher scripts of the robots take the robot description from.
"""

from __future__ import annotations

import inspect
from importlib.resources import files
from types import ModuleType
from typing_extensions import Iterator

import pytest
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String

from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.ros2_interface import get_robot_description
from giskardpy.middleware.ros2.scripts.iai_robots.daisy import (
    daisy_standalone,
    daisy_velocity,
)
from giskardpy.middleware.ros2.scripts.iai_robots.hsr import (
    hsr_standalone,
    hsr_velocity,
)
from giskardpy.middleware.ros2.scripts.iai_robots.pr2 import pr2_standalone
from giskardpy.middleware.ros2.scripts.iai_robots.stretch import (
    stretch_standalone,
    stretch_velocity,
)
from giskardpy.middleware.ros2.scripts.iai_robots.tracy import (
    tracy_standalone,
    tracy_velocity,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.utils import daisy_installed, tracy_installed

# %% commanding a real robot


@pytest.fixture()
def published_robot_description(init_rospy) -> Iterator[str]:
    """
    Publishes a robot description the way a robot does and yields it.

    .. note:: The description is of none of the robots that have a launcher, so it cannot
        be confused with the one a robot class declares.
    """
    description = (
        files("semantic_digital_twin")
        .parent.parent.joinpath("resources", "urdf", "simple_two_arm_robot.urdf")
        .read_text()
    )
    topic = inspect.signature(get_robot_description).parameters["topic"].default
    publisher = rospy.get_node().create_publisher(
        String,
        topic,
        QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
    )
    publisher.publish(String(data=description))
    yield description
    rospy.get_node().destroy_publisher(publisher)


@pytest.mark.parametrize(
    "launcher", [tracy_velocity, daisy_velocity, hsr_velocity, stretch_velocity]
)
def test_a_velocity_launcher_uses_the_description_the_robot_publishes(
    launcher: ModuleType, published_robot_description: str
) -> None:
    giskard = launcher.create_giskard()

    assert giskard.world_config.urdf == published_robot_description


# %% simulating a robot


@pytest.mark.parametrize(
    "launcher, robot_class",
    [
        pytest.param(
            tracy_standalone,
            Tracy,
            marks=pytest.mark.skipif(
                not tracy_installed(), reason="Tracy not installed"
            ),
        ),
        pytest.param(
            daisy_standalone,
            DAiSy,
            marks=pytest.mark.skipif(
                not daisy_installed(), reason="DAiSy not installed"
            ),
        ),
        (hsr_standalone, HSRB),
        (pr2_standalone, PR2),
        (stretch_standalone, Stretch),
    ],
)
def test_a_standalone_launcher_uses_the_description_its_robot_class_declares(
    launcher: ModuleType, robot_class: type[AbstractRobot]
) -> None:
    giskard = launcher.create_giskard()

    assert giskard.world_config.urdf is None
    assert giskard.world_config.robot_class is robot_class
