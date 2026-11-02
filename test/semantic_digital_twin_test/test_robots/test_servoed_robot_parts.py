"""
Tests for the robot parts that declare a drive for each of their joints, and for the
servo a drive puts on a joint.
"""

from __future__ import annotations

import pytest
from typing_extensions import Callable

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot, ServoedRobotPart
from semantic_digital_twin.robots.tiago import TiagoMujoco
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    JointDrive,
    JointDynamics,
    ServoGains,
)
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body, PositionServo

# %% the servo a drive puts on a joint


@pytest.fixture
def described_joint() -> RevoluteConnection:
    """
    A joint whose description states its armature, damping and dry friction.
    """
    world = World()
    with world.modify_world():
        parent = Body(name=PrefixedName("parent"))
        child = Body(name=PrefixedName("child"))
        world.add_kinematic_structure_entity(parent)
        world.add_kinematic_structure_entity(child)
        joint = RevoluteConnection.create_with_dofs(
            world=world, parent=parent, child=child, axis=Vector3.Z()
        )
        world.add_connection(joint)
    joint.dynamics = JointDynamics(armature=0.2, damping=3.0, dry_friction=0.5)
    return joint


def test_a_drive_that_overrides_nothing_keeps_the_described_dynamics(described_joint):
    drive = JointDrive(gains=ServoGains(stiffness=1.0, damping=1.0, torque_limit=1.0))

    assert drive.servo_for(described_joint).dynamics == described_joint.dynamics


def test_a_drive_overrides_only_the_armature_and_joint_damping_it_states(
    described_joint,
):
    drive = JointDrive(
        gains=ServoGains(stiffness=1.0, damping=1.0, torque_limit=1.0),
        armature=0.7,
        joint_damping=9.0,
    )

    assert drive.servo_for(described_joint).dynamics == JointDynamics(
        armature=drive.armature,
        damping=drive.joint_damping,
        dry_friction=described_joint.dynamics.dry_friction,
    )


# %% every servo of a robot comes from the drive of one of its parts


def _parse_urdf(robot_type: type[AbstractRobot]) -> World:
    """
    :return: The world ``robot_type``'s URDF describes, without any servo declared.
    """
    return URDFParser.from_file(robot_type.get_ros_file_path()).create_world()


def _parse_mjcf(robot_type: type[AbstractRobot]) -> World:
    """
    :return: The world ``robot_type``'s MJCF describes, without any servo declared.
    """
    return MJCFParser(robot_type.get_ros_file_path()).create_world()


@pytest.mark.parametrize(
    "robot_type, parse_description",
    [
        (HSRB, _parse_urdf),
        (PR2, _parse_urdf),
        (Tracy, _parse_urdf),
        pytest.param(
            TiagoMujoco,
            _parse_mjcf,
            marks=pytest.mark.skipif(
                not TiagoMujoco.is_description_installed(),
                reason="The MuJoCo model of the Tiago is not installed",
            ),
        ),
    ],
    ids=lambda value: value.__name__,
)
def test_every_position_servo_of_a_robot_is_the_drive_of_one_of_its_parts(
    robot_type: type[AbstractRobot],
    parse_description: Callable[[type[AbstractRobot]], World],
):
    """
    Each joint a robot drives is driven with the gains of the drive one of its parts
    declares for it, and turns with the dynamics that drive puts on the joint as
    described.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body(name=PrefixedName("floor")))
    RobotSpecification(robot_type).spawn(world)
    description = parse_description(robot_type)
    gains_by_degree_of_freedom: dict[DegreeOfFreedom, ServoGains] = {
        degree_of_freedom: actuator.gains
        for actuator in world.actuators
        if isinstance(actuator, PositionServo)
        for degree_of_freedom in actuator.dofs
    }

    driven: set[DegreeOfFreedom] = set()
    for part in world.get_semantic_annotations_by_type(ServoedRobotPart):
        for joint_name, drive in part.drive_of_each_joint.items():
            connection = world.get_connection_by_name(joint_name)
            described = description.get_connection_by_name(joint_name)
            assert gains_by_degree_of_freedom[connection.raw_dof] == drive.gains
            assert connection.dynamics == drive.servo_for(described).dynamics
            driven.add(connection.raw_dof)

    assert driven == set(gains_by_degree_of_freedom)
