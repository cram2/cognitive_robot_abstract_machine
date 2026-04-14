import os

import pytest

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.mesh_decomposition.box_decomposer import (
    BoxDecomposer,
)
from semantic_digital_twin.pipeline.mesh_decomposition.coacd import COACDMeshDecomposer
from semantic_digital_twin.pipeline.mesh_decomposition.vhacd import VHACDMeshDecomposer
from semantic_digital_twin.pipeline.pipeline import Pipeline
from semantic_digital_twin.world_description.geometry import Box


@pytest.fixture(scope="function")
def jeroen_cup_world_fixture():
    stl_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "stl",
    )
    world = STLParser(os.path.join(stl_dir, "jeroen_cup.stl")).parse()
    world.root.name = PrefixedName("root")
    return world


def test_coacd(jeroen_cup_world_fixture):
    [cup] = jeroen_cup_world_fixture.bodies
    old_collision_length = len(cup.collision.shapes)

    pipeline = Pipeline([COACDMeshDecomposer(threshold=0.2)])
    pipeline.apply(jeroen_cup_world_fixture)

    assert len(cup.collision.shapes) > old_collision_length


def test_vhacd(jeroen_cup_world_fixture):
    [cup] = jeroen_cup_world_fixture.bodies
    old_collision_length = len(cup.collision.shapes)

    pipeline = Pipeline([VHACDMeshDecomposer(max_convex_hulls=10)])
    pipeline.apply(jeroen_cup_world_fixture)

    assert len(cup.collision.shapes) > old_collision_length


def test_box_decomposer(jeroen_cup_world_fixture):
    [cup] = jeroen_cup_world_fixture.bodies
    old_collision_length = len(cup.collision.shapes)

    pipeline = Pipeline([BoxDecomposer()])

    pipeline.apply(jeroen_cup_world_fixture)

    assert len(cup.collision.shapes) > old_collision_length
    assert all([isinstance(shape, Box) for shape in cup.collision.shapes])
