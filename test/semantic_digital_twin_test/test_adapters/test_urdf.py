import os.path
from dataclasses import dataclass
from math import pi

import numpy as np
import pytest
from urdf_parser_py import urdf as urdfpy

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.robots.tiago import Tiago, TiagoJoint
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


VISUAL_ONLY_LINK_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset", "visual_only_link.urdf"
)
"""
Description whose ``cover_link`` carries a visual and no collision, next to a
``base_link`` that carries both.
"""


def _body_named(world, name):
    """
    :return: The single body of ``world`` with that name.
    """
    [body] = [
        body for body in world.kinematic_structure_entities if body.name.name == name
    ]
    return body


def test_link_without_collision_geometry_has_none_by_default():
    """
    A link that describes no collision keeps none, so a description is read as written.
    """
    world = URDFParser.from_file(VISUAL_ONLY_LINK_URDF).parse()

    cover = _body_named(world, "cover_link")
    assert len(cover.visual.shapes) == 1
    assert len(cover.collision.shapes) == 0


def test_visual_geometry_stands_in_when_a_link_has_no_collision_geometry():
    """
    Cosmetic links are often drawn but not described for contact, which leaves them
    invisible to collision avoidance; the visual then stands in for the missing
    collision geometry.
    """
    world = URDFParser.from_file(
        VISUAL_ONLY_LINK_URDF, use_visual_as_collision_backup=True
    ).parse()

    cover = _body_named(world, "cover_link")
    [collision_shape] = cover.collision.shapes
    [visual_shape] = cover.visual.shapes
    assert collision_shape.scale == visual_shape.scale
    assert collision_shape is not visual_shape


def test_visual_geometry_is_left_out_where_the_link_already_collides():
    """
    The visual is a stand-in, not an addition: a link that describes its own collision
    keeps exactly that geometry.
    """
    as_written = URDFParser.from_file(VISUAL_ONLY_LINK_URDF).parse()
    with_backup = URDFParser.from_file(
        VISUAL_ONLY_LINK_URDF, use_visual_as_collision_backup=True
    ).parse()

    base_as_written = _body_named(as_written, "base_link")
    base_with_backup = _body_named(with_backup, "base_link")
    assert [shape.scale for shape in base_with_backup.collision.shapes] == [
        shape.scale for shape in base_as_written.collision.shapes
    ]


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
    joint_to_be_mimicked = world.get_connection_by_name(
        PR2Joint.LEFT_GRIPPER_LEFT_FINGER
    )
    mimic_joint = world.get_connection_by_name(PR2Joint.LEFT_GRIPPER_RIGHT_FINGER)

    assert joint_to_be_mimicked.dofs == mimic_joint.dofs


def test_declared_joint_dynamics_are_imported(tiago_parser):
    world = tiago_parser.parse()
    dynamics = world.get_connection_by_name(TiagoJoint.LEFT_ARM_1).dynamics
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
