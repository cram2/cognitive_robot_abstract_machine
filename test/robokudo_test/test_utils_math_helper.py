import math
from typing import Optional

import numpy as np
import pytest

from robokudo.utils.math_helper import (
    does_line_intersect_sphere,
    intersection_point,
    distance,
    intersecting_spheres,
    compute_line_intersection_point,
    compute_direction_vector_angle,
    intersecting_cuboids,
)


class TestUtilsMathHelper(object):
    @pytest.mark.parametrize(
        ["point1", "point2", "t", "expected_result"],
        [
            # Validate interpolation
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 0.25, (1.25, 1.25, 1.25)),
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 0.5, (1.5, 1.5, 1.5)),
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 0.75, (1.75, 1.75, 1.75)),
            # point1 == point2
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0.5, (1.0, 1.0, 1.0)),
            # t==0 => return point1
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 0.0, (1.0, 1.0, 1.0)),
            # t==1 => return point2
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 1.0, (2.0, 2.0, 2.0)),
        ],
    )
    def test_intersection_point_valid_input(
        self,
        point1: tuple[float, float, float],
        point2: tuple[float, float, float],
        t: float,
        expected_result: tuple[float, float, float],
    ):
        assert intersection_point(point1, point2, t) == expected_result

    @pytest.mark.parametrize(
        ["point1", "point2", "t", "expected_result"],
        [
            # t < 0
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), -1.0, (1.0, 1.0, 1.0)),
            # t > 1
            ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), 2.0, (1.0, 1.0, 1.0)),
        ],
    )
    def test_intersection_point_invalid_t(
        self,
        point1: tuple[float, float, float],
        point2: tuple[float, float, float],
        t: float,
        expected_result: tuple[float, float, float],
    ):
        assert pytest.raises(ValueError, intersection_point, point1, point2, t)

    @pytest.mark.parametrize(
        ["point1", "point2", "expected_distance"],
        [
            # point1 == point2
            ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0.0),
            # point1 < point2
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), math.sqrt(3)),
            # point1 > point2
            ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), math.sqrt(3)),
            # point1 == point2 == (0.0, 0.0, 0.0)
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        ],
    )
    def test_distance(
        self,
        point1: tuple[float, float, float],
        point2: tuple[float, float, float],
        expected_distance: float,
    ):
        assert distance(point1, point2) == expected_distance

    def test_intersecting_spheres_empty_list(self):
        # Empty spheres list
        point1, point2 = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        intersections = intersecting_spheres(point1, point2, [])
        assert len(intersections) == 0

    def test_intersecting_spheres_input_order_kept(self):
        # Line intersects both spheres, sphere2 closer than sphere 1
        point1, point2 = (0.0, 1.0, 1.0), (1.5, 1.0, 1.0)
        spheres = [("sphere1", (1.0, 1.0, 1.0), 1.0), ("sphere2", (1.5, 1.0, 1.0), 1.0)]
        intersections = intersecting_spheres(point1, point2, spheres)

        # Order should be kept
        assert intersections[0][1] == spheres[0][0]
        assert intersections[1][1] == spheres[1][0]

        assert intersections[0][0] == 0.0
        assert intersections[1][0] == 0.5

    def test_intersecting_spheres_input_order_reversed(self):
        # Line intersects both spheres, sphere2 closer than sphere 1
        point1, point2 = (0.0, 1.0, 1.0), (1.5, 1.0, 1.0)
        spheres = [("sphere1", (1.5, 1.0, 1.0), 1.0), ("sphere2", (1.0, 1.0, 1.0), 1.0)]
        intersections = intersecting_spheres(point1, point2, spheres)

        # Order should be reversed
        assert intersections[0][1] == spheres[1][0]
        assert intersections[1][1] == spheres[0][0]

        assert intersections[0][0] == 0.0
        assert intersections[1][0] == 0.5

    def test_intersecting_spheres_one_intersection(self):
        # Line one sphere only
        point1, point2 = (0.0, 1.0, 1.0), (1.5, 1.0, 1.0)
        spheres = [("sphere1", (1.5, 1.0, 1.0), 1.0), ("sphere2", (5.0, 5.0, 5.0), 1.0)]
        intersections = intersecting_spheres(point1, point2, spheres)

        # Only sphere one remains
        assert intersections[0][1] == spheres[0][0]
        assert intersections[0][0] == 0.5

    def test_intersecting_spheres_invalid_sphere_radius(self):
        point1, point2 = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        spheres = [
            ("sphere1", (1.0, 1.0, 1.0), -1.0),
        ]
        assert pytest.raises(ValueError, intersecting_spheres, point1, point2, spheres)

    @pytest.mark.parametrize(
        ["point1", "point2", "sphere_center", "sphere_radius", "intersects"],
        [
            # point1 == point2 == sphere_center, no radius
            ((1, 1, 1), (1, 1, 1), (1, 1, 1), 0.0, True),
            # point1 == point2 != sphere_center
            ((1, 1, 1), (1, 1, 1), (0.95, 0.95, 0.95), 0.1, True),
            # point1 != point2 == sphere_center, no radius
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0.0, True),
            # Line intersects sphere exactly at the center
            ((0, 0, 0), (2, 2, 2), (1, 1, 1), 0.0, True),
            # Line ends exactly at sphere surface
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), 1.0, True),
            # Line tangent to sphere
            ((1.0, 2.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, True),
            # Line entirely inside sphere
            ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5), (1.0, 1.0, 1.0), 1.0, True),
            # Line segment outside sphere
            ((3.0, 3.0, 3.0), (4.0, 4.0, 4.0), (1.0, 1.0, 1.0), 1.0, False),
            # Line segment parallel to sphere surface
            ((0.0, 2.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 0.0), 1.0, False),
        ],
    )
    def test_does_line_intersect_sphere(
        self,
        point1: tuple[float, float, float],
        point2: tuple[float, float, float],
        sphere_center: tuple[float, float, float],
        sphere_radius: float,
        intersects: bool,
    ):
        assert (
            does_line_intersect_sphere(point1, point2, sphere_center, sphere_radius)
            == intersects
        )

    def test_does_line_intersect_sphere_invalid_radius(self):
        assert pytest.raises(
            ValueError,
            does_line_intersect_sphere,
            (1, 1, 1),
            (1, 1, 1),
            (1.0, 1.0, 1.0),
            -1.0,
        )

    @pytest.mark.parametrize(
        ["point1", "point2", "sphere_center", "sphere_radius", "intersects_at"],
        [
            # point1 == point2 == sphere_center, no radius
            ((1, 1, 1), (1, 1, 1), (1, 1, 1), 0.0, [0.0]),
            # point1 == point2 != sphere_center
            ((1, 1, 1), (1, 1, 1), (0.95, 0.95, 0.95), 0.1, [0.0]),
            # point1 == point2, outside sphere
            ((2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, None),
            # point1 != point2 == sphere_center, no radius
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0.0, [1.0, 1.0]),
            # Line intersects sphere exactly at the center
            ((0, 0, 0), (2, 2, 2), (1, 1, 1), 0.0, [0.5, 0.5]),
            # Line ends exactly at sphere surface
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), 1.0, [1.0]),
            # Line starts inside sphere and exits once
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, [0.5]),
            # Line tangent to sphere
            ((1.0, 2.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, [1.0, 1.0]),
            # Line entirely inside sphere
            ((0.5, 0.5, 0.5), (1.5, 1.5, 1.5), (1.0, 1.0, 1.0), 1.0, [0.0, 1.0]),
            # Line segment outside sphere
            ((3.0, 3.0, 3.0), (4.0, 4.0, 4.0), (1.0, 1.0, 1.0), 1.0, None),
            # Line segment parallel to sphere surface
            ((0.0, 2.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 0.0), 1.0, None),
        ],
    )
    def test_compute_line_intersection_point(
        self,
        point1: tuple[float, float, float],
        point2: tuple[float, float, float],
        sphere_center: tuple[float, float, float],
        sphere_radius: float,
        intersects_at: Optional[list[float]],
    ):
        assert (
            compute_line_intersection_point(
                point1, point2, sphere_center, sphere_radius
            )
            == intersects_at
        )

    @pytest.mark.parametrize(
        "direction_vector", [np.array([0, 0, 0]), np.array([1e-10, 0, 0])]
    )
    def test_compute_direction_vector_angle_small_vector(
        self, direction_vector: np.ndarray
    ):
        assert pytest.raises(
            ValueError, compute_direction_vector_angle, direction_vector
        )

    def test_zero_floor_vector(self):
        angle = compute_direction_vector_angle(np.array([1, 0, 0]), np.array([0, 0, 0]))
        assert np.isnan(angle)

    def test_vertical_direction_vector(self):
        angle = compute_direction_vector_angle(np.array([0, 1, 0]))
        assert angle == 90.0

    def test_negative_vertical_direction_vector(self):
        angle = compute_direction_vector_angle(np.array([0, -1, 0]))
        assert angle == -90.0

    def test_horizontal_direction_vector_x(self):
        angle = compute_direction_vector_angle(np.array([1, 0, 0]))
        assert angle == 0.0

    def test_horizontal_direction_vector_z(self):
        angle = compute_direction_vector_angle(np.array([0, 0, 1]))
        assert angle == 0.0

    def test_angle_45_degrees(self):
        angle = compute_direction_vector_angle(np.array([1, 1, 0]))
        assert angle == 45.0

    def test_non_unity_floor_vector(self):
        angle = compute_direction_vector_angle(np.array([1, 1, 0]), np.array([0, 2, 0]))
        assert angle == 45.0

    def test_non_standard_floor_vector(self):
        angle = compute_direction_vector_angle(np.array([1, 0, 0]), np.array([1, 0, 0]))
        assert angle == 90.0

    def test_direction_perpendicular_to_floor(self):
        angle = compute_direction_vector_angle(np.array([0, 1, 0]), np.array([1, 0, 0]))
        assert angle == 0.0

    def test_negative_components(self):
        angle = compute_direction_vector_angle(np.array([-1, 0, 0]))
        assert angle == 0.0

    def test_3d_direction_vector(self):
        angle = compute_direction_vector_angle(np.array([1, 1, 1]))
        expected_angle = np.degrees(np.arctan2(1, np.sqrt(2)))
        assert angle == expected_angle

    @pytest.mark.parametrize(
        "dimensions",
        [
            (-1.0, 1.0, 1.0),  # Invalid width
            (1.0, -1.0, 1.0),  # Invalid height
            (1.0, 1.0, -1.0),  # Invalid depth
            (-1.0, -1.0, -1.0),  # Invalid everything
        ],
    )
    def test_intersecting_cuboids_invalid_cuboid_dimensions(
        self, dimensions: tuple[float, float, float]
    ):
        assert pytest.raises(
            ValueError,
            intersecting_cuboids,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            [
                ("cuboid1", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), dimensions),
            ],
        )

    def test_intersecting_cuboids_invalid_rotation(self):
        point1, point2 = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        cuboids = [("cuboid1", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
        assert pytest.raises(ValueError, intersecting_cuboids, point1, point2, cuboids)

    def test_intersecting_cuboids_all_zero(self):
        point1, point2 = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        cuboids = [("cuboid1", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 1
        assert intersections[0][1] == cuboids[0][0]

    def test_intersecting_cuboids_zero_length_line_segment_contained_inside_cuboid(
        self,
    ):
        point1, point2 = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        cuboids = [("cuboid1", (1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 1
        assert intersections[0][0] == 0.0

    def test_intersecting_cuboids_line_segment_contained_inside_cuboid(self):
        point1, point2 = (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
        cuboids = [("cuboid1", (0, 0, 0), (0.0, 0.0, 0.0, 1.0), (2, 2, 2))]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 1
        assert intersections[0][0] == 0.0

    def test_intersecting_cuboids_line_outside_cuboids(self):
        point1, point2 = (2.0, 2.0, 2.0), (-2.0, -2.0, 2.0)
        cuboids = [
            ("cuboid1", (0, 0, 0), (0.0, 0.0, 0.0, 1.0), (2, 2, 2)),
            ("cuboid2", (2.0, 0, 0), (0.0, 0.0, 0.0, 1.0), (2, 2, 2)),
        ]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 0

    def test_intersecting_cuboids_line_intersects_with_end_inside_cuboid(self):
        point1, point2 = (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        cuboids = [("cuboid1", (0, 0, 0), (0.0, 0.0, 0.0, 1.0), (1, 1, 1))]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 1
        assert intersections[0][0] == 1.5

    def test_intersecting_cuboids_line_intersects_with_start_inside_cuboid(self):
        point1, point2 = (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)
        cuboids = [("cuboid1", (0, 0, 0), (0.0, 0.0, 0.0, 1.0), (1, 1, 1))]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 1
        assert intersections[0][0] == 0.5

    def test_intersecting_cuboids_no_intersection_beyond_point2(self):
        point1, point2 = (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)
        cuboids = [("cuboid1", (3.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 0

    def test_intersecting_cuboids_line_intersects_multiple_cuboids_incorrect_input_order(
        self,
    ):
        point1, point2 = (2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)
        cuboids = [
            ("cuboid1", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1, 1, 1)),
            ("cuboid2", (1.0, 0, 0), (0.0, 0.0, 0.0, 1.0), (1, 1, 1)),
        ]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 2

        # Order must be reversed
        assert intersections[0][1] == cuboids[1][0]
        assert intersections[1][1] == cuboids[0][0]

        assert intersections[0][0] == 0.5
        assert intersections[1][0] == 1.5

    def test_intersecting_cuboids_line_intersects_multiple_cuboids_correct_input_order(
        self,
    ):
        point1, point2 = (2.0, 0.0, 0.0), (-2.0, 0.0, 0.0)
        cuboids = [
            ("cuboid1", (1.0, 0, 0), (0.0, 0.0, 0.0, 1.0), (1, 1, 1)),
            ("cuboid2", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1, 1, 1)),
        ]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 2

        # Order must be kept
        assert intersections[0][1] == cuboids[0][0]
        assert intersections[1][1] == cuboids[1][0]

        assert intersections[0][0] == 0.5
        assert intersections[1][0] == 1.5

    @pytest.mark.parametrize(
        ["point1", "point2"],
        [
            # Top edges
            ((1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)),
            ((1.0, 1.0, 1.0), (1.0, -1.0, 1.0)),
            ((-1.0, -1.0, 1.0), (1.0, -1.0, 1.0)),
            ((-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)),
            # Vertical edges
            ((1.0, 1.0, 1.0), (1.0, 1.0, -1.0)),
            ((-1.0, 1.0, 1.0), (-1.0, 1.0, -1.0)),
            ((-1.0, -1.0, 1.0), (-1.0, -1.0, -1.0)),
            ((1.0, -1.0, 1.0), (1.0, -1.0, -1.0)),
            # Bottom edges
            ((1.0, 1.0, -1.0), (-1.0, 1.0, -1.0)),
            ((1.0, 1.0, -1.0), (1.0, -1.0, -1.0)),
            ((-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0)),
            ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0)),
            # Top face
            ((1.0, 1.0, 1.0), (-1.0, -1.0, 1.0)),
            # Bottom face
            ((1.0, 1.0, -1.0), (-1.0, -1.0, -1.0)),
            # Vertical faces
            ((1.0, 1.0, 1.0), (-1.0, 1.0, -1.0)),
            ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0)),
            ((-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0)),
            ((-1.0, -1.0, 1.0), (1.0, -1.0, -1.0)),
        ],
    )
    def test_intersecting_cuboids_segment_on_edge(
        self, point1: tuple[float, float, float], point2: tuple[float, float, float]
    ):
        cuboids = [("cuboid1", (0, 0, 0), (0.0, 0.0, 0.0, 1.0), (2, 2, 2))]
        intersections = intersecting_cuboids(point1, point2, cuboids)
        assert len(intersections) == 1
        assert intersections[0][0] == 0.0
