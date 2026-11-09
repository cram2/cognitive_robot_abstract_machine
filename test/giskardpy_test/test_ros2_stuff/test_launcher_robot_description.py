"""
Coverage for where the launcher scripts of the robots take the robot description from.
"""

from __future__ import annotations

import inspect
from importlib.resources import files
from types import ModuleType

import pytest
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
from typing_extensions import Iterator

from semantic_digital_twin.adapters.ros.ros2_interface import get_robot_description
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
from semantic_digital_twin.adapters.ros import rospy
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.specifications.worlds import WorldSpecification


# %% reading the robot of a world specification


def urdfs_read_by(specification: WorldSpecification) -> list[str]:
    """
    :return: The URDFs the environment and the robots of ``specification`` are read
        from, as their parsers normalized them.
    """
    parsers = [specification.world_source] + [
        robot.description_parser for robot in specification.robots
    ]
    return [parser.urdf for parser in parsers if isinstance(parser, URDFParser)]


def robot_classes_of(specification: WorldSpecification) -> list[type[AbstractRobot]]:
    """
    :return: The robot classes ``specification`` annotates.
    """
    return [robot.semantic_annotation_type for robot in specification.robots]


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
    giskard = launcher.create_giskard(rospy.get_node())

    assert urdfs_read_by(giskard.world_source) == [
        URDFParser(urdf=published_robot_description).urdf
    ]


# %% simulating a robot


@pytest.mark.parametrize(
    "launcher, robot_class",
    [
        pytest.param(
            tracy_standalone,
            Tracy,
            marks=pytest.mark.skipif(
                not Tracy.is_description_installed(), reason="Tracy not installed"
            ),
        ),
        pytest.param(
            daisy_standalone,
            DAiSy,
            marks=pytest.mark.skipif(
                not DAiSy.is_description_installed(), reason="DAiSy not installed"
            ),
        ),
        (hsr_standalone, HSRB),
        (pr2_standalone, PR2),
        (stretch_standalone, Stretch),
    ],
)
def test_a_standalone_launcher_uses_the_description_its_robot_class_declares(
    launcher: ModuleType, robot_class: type[AbstractRobot], init_rospy
) -> None:
    giskard = launcher.create_giskard(rospy.get_node())

    assert urdfs_read_by(giskard.world_source) == [
        URDFParser.from_file(robot_class.get_ros_file_path()).urdf
    ]
    assert robot_classes_of(giskard.world_source) == [robot_class]
