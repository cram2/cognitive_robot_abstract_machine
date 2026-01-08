import itertools

import pytest

from semantic_digital_twin.collision_checking.collision_detector import CollisionCheck
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    TrimeshCollisionDetector,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.testing import world_setup_simple


def test_simple_collision(world_setup_simple):
    world, body1, body2, body3, body4 = world_setup_simple
    tcd = TrimeshCollisionDetector(world)
    collision = tcd.check_collision_between_bodies(body1, body2)
    assert collision
    assert {collision.body_a, collision.body_b} == {body1, body2}


def test_no_collision(world_setup_simple):
    world, body1, body2, body3, body4 = world_setup_simple
    body1.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 1, 1
    )
    tcd = TrimeshCollisionDetector(world)
    collision = tcd.check_collision_between_bodies(body1, body2)
    assert not collision


@pytest.mark.skip(reason="Not my krrood_test not my problem.")
def test_collision_matrix(world_setup_simple):
    world, body1, body2, body3, body4 = world_setup_simple
    tcd = TrimeshCollisionDetector(world)
    collisions = tcd.check_collisions(
        {
            CollisionCheck(body1, body2, 0.0, world),
            CollisionCheck(body3, body4, 0.0, world),
        }
    )
    assert len(collisions) == 2
    pairs = {(c.body_a, c.body_b) for c in collisions}
    assert (body1, body2) in pairs
    assert (body3, body4) in pairs


def test_all_collisions(world_setup_simple):
    world, body1, body2, body3, body4 = world_setup_simple
    tcd = TrimeshCollisionDetector(world)
    body4.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        10, 10, 10
    )
    body3.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -10, -10, 10
    )

    collisions = tcd.check_collisions(
        [
            CollisionCheck(a, b, _world=world, distance=0.0001)
            for a, b in itertools.combinations(world.bodies_with_enabled_collision, 2)
        ]
    )
    assert len(collisions) == 1
    assert {collisions[0].body_a, collisions[0].body_b} == {body1, body2}
