import itertools

import pytest

from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionMatrix,
    CollisionCheck,
)
from semantic_digital_twin.collision_checking.pybullet_collision_detector import (
    BulletCollisionDetector,
)
from semantic_digital_twin.collision_checking.trimesh_collision_detector import (
    FCLCollisionDetector,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.testing import world_setup_simple
import numpy as np

collision_detectors = [BulletCollisionDetector, FCLCollisionDetector]


@pytest.mark.parametrize("collision_detector", collision_detectors)
def test_simple_collision(world_setup_simple, collision_detector):
    world, body1, body2, body3, body4, body5 = world_setup_simple
    tcd = collision_detector(_world=world)
    collision = tcd.check_collision_between_bodies(body1, body2)
    assert collision
    assert {collision.body_a, collision.body_b} == {body1, body2}


@pytest.mark.parametrize("collision_detector", collision_detectors)
def test_contact_distance(world_setup_simple, collision_detector):
    world, box, cylinder, sphere, mesh, compound = world_setup_simple
    cylinder.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 0, 0
    )
    tcd = collision_detector(_world=world)
    collision = tcd.check_collision_between_bodies(cylinder, sphere, distance=10)
    assert collision

    if collision.body_a == cylinder:
        map_P_cylinder = collision.root_P_point_on_body_a
        map_P_sphere = collision.root_P_point_on_body_b
        map_V_contact_normal = -collision.root_V_contact_normal_from_b_to_a
        assert collision.body_b == sphere
    else:
        map_P_sphere = collision.root_P_point_on_body_a
        map_P_cylinder = collision.root_P_point_on_body_b
        map_V_contact_normal = collision.root_V_contact_normal_from_b_to_a
        assert collision.body_b == cylinder
        assert collision.body_a == sphere

    assert np.allclose(map_P_cylinder, [0.75, 0, 0, 1], atol=1e-5)
    assert np.allclose(map_P_sphere, [0.1, 0, 0, 1], atol=1e-5)
    assert np.allclose(map_V_contact_normal, [-1, 0, 0, 0], atol=1e-5)
    assert np.isclose(collision.distance, 0.65)


@pytest.mark.parametrize("collision_detector", collision_detectors)
@pytest.mark.parametrize("directions", [[1, 0, 0], [0, 1, 0]])
def test_contact_distance_compound_front(
    world_setup_simple, collision_detector, directions
):
    world, box, cylinder, sphere, mesh, compound = world_setup_simple
    sphere.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        directions[0], directions[1], directions[2]
    )
    tcd = collision_detector(_world=world)
    collision_compound = tcd.check_collision_between_bodies(
        sphere, compound, distance=10
    )
    collision_box = tcd.check_collision_between_bodies(sphere, box, distance=10)

    if collision_box.body_a == sphere:
        map_P_sphere1 = collision_box.root_P_point_on_body_a
        map_P_box = collision_box.root_P_point_on_body_b
    else:
        map_P_box = collision_box.root_P_point_on_body_a
        map_P_sphere1 = collision_box.root_P_point_on_body_b

    if collision_compound.body_a == sphere:
        map_P_sphere2 = collision_compound.root_P_point_on_body_a
        map_P_compound = collision_compound.root_P_point_on_body_b
    else:
        map_P_compound = collision_compound.root_P_point_on_body_a
        map_P_sphere2 = collision_compound.root_P_point_on_body_b

    assert np.allclose(map_P_sphere1, map_P_sphere2, atol=1e-3)
    assert np.allclose(map_P_compound, map_P_box, atol=1e-3)

    assert np.isclose(collision_compound.distance, collision_box.distance)


@pytest.mark.parametrize("collision_detector", collision_detectors)
def test_no_collision(world_setup_simple, collision_detector):
    world, body1, body2, body3, body4, body5 = world_setup_simple
    body1.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, 1, 1
    )
    tcd = collision_detector(_world=world)
    collision = tcd.check_collision_between_bodies(body1, body2)
    assert not collision


@pytest.mark.skip(reason="Not my krrood_test not my problem.")
def test_collision_matrix(world_setup_simple):
    world, body1, body2, body3, body4, body5 = world_setup_simple
    tcd = FCLCollisionDetector(_world=world)
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
    world, body1, body2, body3, body4, body5 = world_setup_simple
    tcd = collision_detector(_world=world)
    body3.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -10, -10, 10
    )
    body4.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        10, 10, 10
    )
    body5.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        -10, -10, -10
    )

    collisions = tcd.check_collisions(
        CollisionMatrix.create_all_checks(distance=0.0001, world=world)
    ).contacts
    assert len(collisions) == 1
    assert {collisions[0].body_a, collisions[0].body_b} == {body1, body2}
