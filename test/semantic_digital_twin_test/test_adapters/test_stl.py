from pathlib import Path

import pytest

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from semantic_digital_twin.world_description.geometry import Mesh

# %% meshes of the iai_objects package


@pytest.fixture
def milk_uri() -> str:
    """
    :return: The package URI of the milk carton mesh.
    """
    return "package://iai_objects/objects/milk.stl"


@pytest.fixture
def cup_uri() -> str:
    """
    :return: The package URI of the jeroen cup mesh.
    """
    return "package://iai_objects/objects/jeroen_cup.stl"


# %% parsing


def test_stl_parsing_construct(milk_uri):
    stl_parser = STLParser(milk_uri)
    assert stl_parser.file_path == milk_uri


def test_stl_parsing(milk_uri):
    world = STLParser(milk_uri).parse()
    world.validate()

    assert len(world.kinematic_structure_entities) == 1
    assert len(world.kinematic_structure_entities[0].collision) == 1
    assert len(world.kinematic_structure_entities[0].visual) == 1
    assert type(world.kinematic_structure_entities[0].collision[0]) is Mesh


def test_a_mesh_named_by_a_package_uri_is_read_from_that_package(milk_uri):
    world = STLParser(milk_uri).parse()

    [mesh] = world.kinematic_structure_entities[0].collision
    assert mesh.local_file == Path(PackageUriResolver().resolve(milk_uri))


def test_parse_and_merge(milk_uri, cup_uri):
    milk_world = STLParser(milk_uri).parse()
    cup_world = STLParser(cup_uri).parse()

    milk_world.merge_world(cup_world)
    assert len(milk_world.kinematic_structure_entities) == 2
    assert len(milk_world.kinematic_structure_entities[0].collision) == 1
