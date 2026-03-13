import os.path

import pytest

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.world_description.connections import FixedConnection


MJCF_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "semantic_digital_twin",
    "resources",
    "mjcf",
)


@pytest.fixture
def table_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "table.xml"))


@pytest.fixture
def kitchen_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "kitchen-small.xml"))


@pytest.fixture
def apartment_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "iai_apartment.xml"))


@pytest.fixture
def pr2_xml_parser():
    return MJCFParser(os.path.join(MJCF_DIR, "pr2_kinematic_tree.xml"))


def test_table_parsing(table_xml_parser):
    body_num = 7
    world = table_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) == body_num

    origin_left_front_leg_joint = world.get_connection(
        world.root, world.kinematic_structure_entities[1]
    )
    assert isinstance(origin_left_front_leg_joint, FixedConnection)


def test_kitchen_parsing(kitchen_xml_parser):
    world = kitchen_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_apartment_parsing(apartment_xml_parser):
    world = apartment_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0


def test_pr2_parsing(pr2_xml_parser):
    world = pr2_xml_parser.parse()
    world.validate()

    assert len(world.kinematic_structure_entities) > 0
    assert len(world.connections) > 0
    assert world.root.name.name == "world"