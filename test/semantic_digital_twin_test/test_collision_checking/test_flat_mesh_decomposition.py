"""
Tests that a world accepts a body whose collision geometry is a flat, zero-volume mesh.
"""

import numpy as np
import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% flat mesh collision geometry


def test_world_accepts_body_with_flat_mesh_collision():
    """
    A one-millimetre-wide wall strip, as architectural USD scenes contain, is a planar
    quad: VHACD finds no convex parts in it, which must not stop the world from building.
    """
    wall_strip = trimesh.Trimesh(
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
    body = Body(
        name=PrefixedName("wall_strip"),
        collision=ShapeCollection([Mesh.from_trimesh(mesh=wall_strip)]),
    )
    world = World()

    with world.modify_world():
        world.add_body(body)

    assert world.bodies == [body]
