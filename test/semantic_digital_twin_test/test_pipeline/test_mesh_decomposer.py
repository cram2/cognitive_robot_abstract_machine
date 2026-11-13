import os

import numpy as np
import pytest
import trimesh

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.mesh_decomposition.box_decomposer import (
    BoxDecomposer,
)
from semantic_digital_twin.pipeline.mesh_decomposition.coacd import COACDMeshDecomposer
from semantic_digital_twin.pipeline.mesh_decomposition.vhacd import VHACDMeshDecomposer
from semantic_digital_twin.pipeline.pipeline import Pipeline
from semantic_digital_twin.world_description.geometry import Box, Mesh


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
    world = STLParser(os.path.join(stl_dir, "jeroen_cup.stl")).create_world()
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


# %% flat mesh decomposition


@pytest.fixture
def millimetre_wall_strip() -> trimesh.Trimesh:
    """
    A flat, zero-volume quad one millimetre wide and 2.8 metres tall, in which V-HACD
    finds no convex parts.
    """
    return trimesh.Trimesh(
        vertices=np.array(
            [
                [0.001, 0.0, 2.8],
                [0.0, 0.0, 2.8],
                [0.001, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        faces=np.array([[2, 1, 0], [1, 2, 3]]),
        process=False,
    )


def test_vhacd_decomposes_flat_mesh_into_its_convex_hull(millimetre_wall_strip):
    """
    A mesh in which V-HACD finds no convex parts is decomposed into its convex hull.
    """
    [part] = VHACDMeshDecomposer().apply_to_mesh(
        Mesh.from_trimesh(mesh=millimetre_wall_strip)
    )

    convex_hull = millimetre_wall_strip.convex_hull
    assert np.allclose(part.mesh.vertices, convex_hull.vertices)
    assert np.array_equal(part.mesh.faces, convex_hull.faces)
