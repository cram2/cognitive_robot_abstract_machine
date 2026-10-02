from __future__ import annotations

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


@pytest.fixture(scope="module")
def robot_world(request: pytest.FixtureRequest) -> World:
    """
    The world parsed from the description of the robot under test, shared by this
    module's tests of that robot and released once they have run.
    """
    robot_type: type[AbstractRobot] = request.param
    return URDFParser.from_file(robot_type.get_ros_file_path()).parse()


# %% joint-name enums against the parsed description


@pytest.mark.parametrize(
    "robot_world, joint_enum",
    ROBOTS_WITH_JOINT_ENUM,
    ids=ROBOT_IDENTIFIERS,
    indirect=["robot_world"],
)
def test_joint_enum_members_name_connections_of_the_robot(
    robot_world: World, joint_enum: type[StrEnum]
):
    """
    Every member must spell a connection name that the robot's description contains.
    """
    connection_names = {connection.name.name for connection in robot_world.connections}

    assert {joint.value for joint in joint_enum} - connection_names == set()


@pytest.mark.parametrize(
    "robot_world, joint_enum",
    ROBOTS_WITH_JOINT_ENUM,
    ids=ROBOT_IDENTIFIERS,
    indirect=["robot_world"],
)
def test_joint_enum_members_name_actuated_connections(
    robot_world: World, joint_enum: type[StrEnum]
):
    """
    Every member must name an actuated connection, since only those accept a joint goal.
    """
    actuated_connection_names = {
        connection.name.name
        for connection in robot_world.connections
        if isinstance(connection, ActiveConnection)
    }

    assert {joint.value for joint in joint_enum} - actuated_connection_names == set()
