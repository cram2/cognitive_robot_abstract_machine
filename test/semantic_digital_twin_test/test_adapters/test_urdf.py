import os.path
from dataclasses import dataclass
from math import pi

import numpy as np
import pytest
from urdf_parser_py import urdf as urdfpy

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class URDFPaths:
    """
    Data class to hold paths to URDF files used in tests.
    """

    table: str
    kitchen: str
    apartment: str
    pr2: str


@pytest.fixture
def urdf_paths():
    """
    Fixture providing paths to various URDF files.
    """
    urdf_directory = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "urdf",
    )
    return URDFPaths(
        table=os.path.join(urdf_directory, "table.urdf"),
        kitchen=os.path.join(urdf_directory, "kitchen-small.urdf"),
        apartment=os.path.join(urdf_directory, "apartment.urdf"),
        pr2=PR2.get_ros_file_path(),
    )


@pytest.fixture
def table_parser(urdf_paths):
    """
    Fixture providing a URDFParser for the table model.
    """
    return URDFParser.from_file(file_path=urdf_paths.table)


@pytest.fixture
def kitchen_parser(urdf_paths):
    """
    Fixture providing a URDFParser for the kitchen model.
    """
    return URDFParser.from_file(file_path=urdf_paths.kitchen)


@pytest.fixture
def apartment_parser(urdf_paths):
    """
    Fixture providing a URDFParser for the apartment model.
    """
    return URDFParser.from_file(file_path=urdf_paths.apartment)


@pytest.fixture
def pr2_parser():
    """
    Fixture providing a URDFParser for the PR2 model.
    """
    return URDFParser.from_file(file_path=PR2.get_ros_file_path())


@pytest.fixture
def tiago_parser():
    """
    Fixture providing a URDFParser for the Tiago model.
    """
    return URDFParser.from_file(file_path=Tiago.get_ros_file_path())


def test_table_parsing(table_parser):
    world = table_parser.parse()
    world.validate()
    assert len(world.kinematic_structure_entities) == 6

    origin_left_front_leg_joint = world.get_connection(
        world.root, world.kinematic_structure_entities[1]
    )
    assert isinstance(origin_left_front_leg_joint, FixedConnection)


def test_kitchen_parsing(kitchen_parser):
    world = kitchen_parser.parse()
    world.validate()
    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_apartment_parsing(apartment_parser):
    world = apartment_parser.parse()
    world.validate()
    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_pr2_parsing(pr2_parser):
    world = pr2_parser.parse()
    world.validate()
    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0
    assert world.root.name.name == "base_footprint"


def test_mimic_joints(pr2_parser):
    world = pr2_parser.parse()
    joint_to_be_mimicked = world.get_connection_by_name("l_gripper_l_finger_joint")
    mimic_joint = world.get_connection_by_name("l_gripper_r_finger_joint")

    assert joint_to_be_mimicked.dofs == mimic_joint.dofs


def test_declared_joint_dynamics_are_imported(tiago_parser):
    world = tiago_parser.parse()
    dynamics = world.get_connection_by_name("arm_left_1_joint").dynamics
    assert dynamics.damping == 40.0
    assert dynamics.dry_friction == 1.0


def test_undeclared_joint_dynamics_default_to_zero(pr2_parser):
    world = pr2_parser.parse()
    dynamics = world.get_connection_by_name("r_gripper_motor_slider_joint").dynamics
    assert dynamics.damping == 0.0
    assert dynamics.dry_friction == 0.0
    assert dynamics.armature == 0.0


def test_declared_link_inertial_is_imported(pr2_parser):
    world = pr2_parser.parse()
    inertial = world.get_body_by_name("l_elbow_flex_link").inertial
    assert inertial.mass == 1.90327
    assert inertial.center_of_mass.to_np()[:3].tolist() == [0.01014, 0.00032, -0.01211]
    assert inertial.inertia.to_values() == (
        0.00346541989,
        0.00441606455,
        0.00359156824,
        0.00004066825,
        0.00043171614,
        -0.00003968914,
    )


def test_inertia_tensor_is_expressed_in_the_link_frame(table_parser):
    link = urdfpy.Link(
        name="rotated_inertial_link",
        inertial=urdfpy.Inertial(
            mass=2.0,
            inertia=urdfpy.Inertia(
                ixx=0.002, iyy=0.003, izz=0.004, ixy=0.0, ixz=0.0, iyz=0.0
            ),
            origin=urdfpy.Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, pi / 2]),
        ),
    )
    body = Body(name=PrefixedName("rotated_inertial_link"))

    inertial = table_parser.parse_inertial(link, body)
    inertia_rotated_by_ninety_degrees_around_z = np.diag([0.003, 0.002, 0.004])
    assert np.allclose(
        inertial.inertia.data, inertia_rotated_by_ninety_degrees_around_z, atol=1e-12
    )


def test_undeclared_link_inertial_keeps_the_default(table_parser):
    inertial = table_parser.parse().root.inertial
    assert inertial.mass == 1.0
    assert inertial.inertia.to_values() == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)


def test_xacro():
    path = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    parser = URDFParser.from_xacro(path)
    world = parser.parse()
    world.validate()
    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0
    assert world.root.name.name == "base_footprint"
