import numpy as np
import pytest

from semantic_digital_twin.testing import world_setup_simple


def test_closest_points(world_setup_simple):
    world, body1, body2, body3, _ = world_setup_simple

    world.get_connection(world.root, body1).origin

    world.get_connection(world.root, body1).origin = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    point_body1, point_body2, dist = body1.compute_closest_points_multi([body2])

    # Check that y and z coordinates are the same since we only moved in the x direction
    assert point_body1[0][1] == point_body2[0][1]
    assert point_body1[0][2] == point_body2[0][2]
    # Check that the distance between the points is the distance between the two boxes
    assert dist == pytest.approx(0.75, abs=1e-2)


def test_closest_points_approximate(world_setup_simple):
    world, body1, body2, body3, _ = world_setup_simple

    world.get_connection(world.root, body3).origin = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0.5], [0, 0, 1, 0.125], [0, 0, 0, 1]]
    )

    point_body3, point_body1, dist = body3.compute_closest_points_multi(
        [body1], sample_size=20
    )

    # Check that the distance between the found points to be within 2 cm
    assert dist == pytest.approx(
        0.385, abs=2e-2
    )  # sphere is 0.385 meter away from the surface of the box plus the radius of the sphere (0.01 m)


def test_closest_points_transform(world_setup_simple):
    world, body1, body2, body3, body4 = world_setup_simple

    world.get_connection(world.root, body4).origin = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
    )

    point_body3, point_body4, dist = body3.compute_closest_points_multi(
        [body4], sample_size=20
    )

    assert point_body4[0][0] == pytest.approx(
        1.0, abs=1e-2
    )  # 1 meter away from the surface of the box plus the radius of the sphere (0.01 m)
    assert point_body4[0][1] == pytest.approx(1.0, abs=1e-2)
    assert point_body4[0][2] == pytest.approx(
        1.0, abs=1e-2
    )  # 1 meter away from the surface of the box plus the radius of the sphere (0.01 m)


def test_closest_points_multi(world_setup_simple):
    world, body1, body2, body3, body4 = world_setup_simple

    world.get_connection(world.root, body3).origin = np.array(
        [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    world.get_connection(world.root, body4).origin = np.array(
        [[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    point_body1, point_other_bodies, dist_1 = body1.compute_closest_points_multi(
        [body3, body4], sample_size=20
    )

    assert dist_1[0] == pytest.approx(
        0.875, abs=2e-2
    )  # sphere is 2.385 meter away from the surface of the box plus the radius of the sphere (0.01 m)
    assert dist_1[1] == pytest.approx(
        0.875, abs=2e-2
    )  # sphere is 0.385 meter away from the surface of the box plus the radius of the sphere (0.01 m)

    assert point_body1[0][0] == pytest.approx(0.125, abs=1e-2)
    assert point_body1[1][0] == pytest.approx(-0.125, abs=1e-2)
