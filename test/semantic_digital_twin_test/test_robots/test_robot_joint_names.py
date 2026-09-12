from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, field
from enum import StrEnum

import pytest

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.armar7 import Armar7, Armar7Joint
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyJoint
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.robots.icub3 import ICub3, ICub3Joint
from semantic_digital_twin.robots.justin import Justin, JustinJoint
from semantic_digital_twin.robots.mmp_dresden import MMPDresden, MMPDresdenJoint
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch, StretchJoint
from semantic_digital_twin.robots.tiago import Tiago, TiagoJoint
from semantic_digital_twin.robots.tracy import Tracy, TracyJoint
from semantic_digital_twin.robots.unitree_g1 import UnitreeG1, UnitreeG1Joint
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection

# %% robots paired with their joint-name enum

ROBOTS_WITH_JOINT_ENUM: list[tuple[type[AbstractRobot], type[StrEnum]]] = [
    (PR2, PR2Joint),
    (HSRB, HSRBJoint),
    (Tiago, TiagoJoint),
    (Stretch, StretchJoint),
    (Tracy, TracyJoint),
    (DAiSy, DAiSyJoint),
    (Armar7, Armar7Joint),
    (ICub3, ICub3Joint),
    (Justin, JustinJoint),
    (UnitreeG1, UnitreeG1Joint),
    (MMPDresden, MMPDresdenJoint),
]
"""
Every robot whose description can be resolved, together with the enum naming its joints.

Garmi and :class:`TiagoMujoco` are absent because their descriptions are unavailable, so
their joint names cannot be checked against a parsed world.
"""

ROBOT_IDENTIFIERS = [robot.__name__ for robot, _ in ROBOTS_WITH_JOINT_ENUM]
"""
Test identifiers naming the robot under test.
"""


@dataclass
class ParsedRobotDescriptions:
    """
    The worlds parsed from robot descriptions, reused by everything holding the same
    instance.

    .. note:: The worlds live exactly as long as the instance, so whoever holds it decides
        how long the parsed descriptions stay in memory.
    """

    worlds_by_robot: dict[type[AbstractRobot], World] = field(default_factory=dict)
    """
    The world already parsed for a robot.
    """

    def world_of(self, robot_type: type[AbstractRobot]) -> World:
        """
        The world holding the robot's description, parsing it on first request.

        :param robot_type: The robot whose description is parsed.
        """
        if robot_type not in self.worlds_by_robot:
            self.worlds_by_robot[robot_type] = URDFParser.from_file(
                robot_type.get_ros_file_path()
            ).parse()
        return self.worlds_by_robot[robot_type]


@pytest.fixture(scope="module")
def parsed_robot_descriptions() -> ParsedRobotDescriptions:
    """
    Descriptions parsed once for this module, and released when it is done with them.
    """
    return ParsedRobotDescriptions()


# %% joint-name enums against the parsed description


@pytest.mark.parametrize(
    "robot_type, joint_enum", ROBOTS_WITH_JOINT_ENUM, ids=ROBOT_IDENTIFIERS
)
def test_joint_enum_members_name_connections_of_the_robot(
    robot_type: type[AbstractRobot],
    joint_enum: type[StrEnum],
    parsed_robot_descriptions: ParsedRobotDescriptions,
):
    """
    Every member must spell a connection name that the robot's description contains.
    """
    world = parsed_robot_descriptions.world_of(robot_type)
    connection_names = {connection.name.name for connection in world.connections}

    assert {joint.value for joint in joint_enum} - connection_names == set()


@pytest.mark.parametrize(
    "robot_type, joint_enum", ROBOTS_WITH_JOINT_ENUM, ids=ROBOT_IDENTIFIERS
)
def test_joint_enum_members_name_actuated_connections(
    robot_type: type[AbstractRobot],
    joint_enum: type[StrEnum],
    parsed_robot_descriptions: ParsedRobotDescriptions,
):
    """
    Every member must name an actuated connection, since only those accept a joint goal.
    """
    world = parsed_robot_descriptions.world_of(robot_type)
    actuated_connection_names = {
        connection.name.name
        for connection in world.connections
        if isinstance(connection, ActiveConnection)
    }

    assert {joint.value for joint in joint_enum} - actuated_connection_names == set()


# %% lifetime of the parsed descriptions


def test_parsed_descriptions_are_released_with_their_holder():
    """
    Nothing may keep a parsed world alive once the descriptions holding it are gone, so
    that a module does not pin worlds for the rest of the test session.
    """
    descriptions = ParsedRobotDescriptions()
    world_reference = weakref.ref(descriptions.world_of(PR2))

    del descriptions
    gc.collect()

    assert world_reference() is None
