"""
Tests that a world accepts a body whose collision geometry is a flat, zero-volume mesh.
"""

import trimesh

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% flat mesh collision geometry


def test_world_builds_collision_object_for_flat_mesh(
    millimetre_wall_strip: trimesh.Trimesh,
):
    """
    V-HACD finds no convex parts in the wall strip, which must not keep the collision
    detector from building a collision object for its body.
    """
    body = Body(
        name=PrefixedName("wall_strip"),
        collision=ShapeCollection([Mesh.from_trimesh(mesh=millimetre_wall_strip)]),
    )
    world = World()

    with world.modify_world():
        world.add_body(body)

    collision_detector = world.collision_manager.collision_detector
    assert list(collision_detector.body_to_bullet_object) == [body]
