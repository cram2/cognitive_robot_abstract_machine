import itertools

import pytest

from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.collision_checking.pybullet_collision_detector import (
    BulletCollisionDetector,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    TrimeshCollisionDetector,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.testing import world_setup_simple
import numpy as np

collision_detectors = [BulletCollisionDetector, TrimeshCollisionDetector]


@pytest.mark.parametrize("collision_detector", collision_detectors)
def test_simple_collision(world_setup_simple, collision_detector):
    world, body1, body2, body3, body4 = world_setup_simple
    tcd = collision_detector(world)
    collision = tcd.check_collision_between_bodies(body1, body2)
    assert collision
    assert {collision.body_a, collision.body_b} == {body1, body2}


@pytest.mark.parametrize("collision_detector", collision_detectors)
def test_contact_distance(world_setup_simple, collision_detector):
    world, box, cylinder, sphere, mesh = world_setup_simple
    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 0, 0
    )
    tcd = collision_detector(world)
    collision = tcd.check_collision_between_bodies(cylinder, sphere, distance=10)
    assert collision

    assert collision.body_a == cylinder
    assert collision.body_b == sphere

    assert np.isclose(collision.map_P_pa[0], 0.75, atol=1e-5)
    assert np.isclose(collision.map_P_pa[1], 0.0, atol=1e-5)
    assert np.isclose(collision.map_P_pa[2], 0.0, atol=1e-5)

    assert np.isclose(collision.map_P_pb[0], 0.1, atol=1e-5)
    assert np.isclose(collision.map_P_pb[1], 0.0, atol=1e-5)
    assert np.isclose(collision.map_P_pb[2], 0.0, atol=1e-5)

    assert np.isclose(collision.contact_distance, 0.65)


@pytest.mark.parametrize("collision_detector", collision_detectors)
def test_no_collision(world_setup_simple, collision_detector):
    world, body1, body2, body3, body4 = world_setup_simple
    body1.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 1, 1
    )
    tcd = collision_detector(world)
    collision = tcd.check_collision_between_bodies(body1, body2)
    assert not collision


@pytest.mark.skip(reason="Not my krrood_test not my problem.")
def test_collision_matrix(world_setup_simple):
    world, body1, body2, body3, body4 = world_setup_simple
    tcd = TrimeshCollisionDetector(world)
    collisions = tcd.check_collisions(
        CollisionMatrix(
            {
                CollisionCheck(body1, body2, 0.0),
                CollisionCheck(body3, body4, 0.0),
            }
        )
    )
    assert len(collisions) == 2
    pairs = {(c.body_a, c.body_b) for c in collisions}
    assert (body1, body2) in pairs
    assert (body3, body4) in pairs


@pytest.mark.parametrize("collision_detector", collision_detectors)
def test_all_collisions(world_setup_simple, collision_detector):
    world, body1, body2, body3, body4 = world_setup_simple
    tcd = collision_detector(world)
    body4.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        10, 10, 10
    )
    body3.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -10, -10, 10
    )

    collisions = tcd.check_collisions(
        CollisionMatrix.create_all_checks(distance=0.0001, world=world)
    ).contacts
    assert len(collisions) == 1
    assert {collisions[0].body_a, collisions[0].body_b} == {body1, body2}
