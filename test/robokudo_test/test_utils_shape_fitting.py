import math

import numpy as np

from robokudo.utils.shape_fitting import (
    CuboidFit,
    CylinderFit,
    SphereFit,
    compute_fit_score,
    fit_cuboid,
    fit_cylinder,
    fit_sphere,
    select_best_shape,
)


class TestShapeFitting:
    def test_fit_sphere_with_noisy_surface_points(self):
        random_generator = np.random.default_rng(4)

        center = np.asarray([0.2, -0.1, 0.4], dtype=np.float64)
        radius = 0.12

        directions = random_generator.normal(size=(500, 3))
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        points = center + radius * directions
        points += random_generator.normal(scale=0.002, size=points.shape)

        fit_result = fit_sphere(points=points, distance_threshold=0.01)

        assert fit_result is not None
        assert np.linalg.norm(fit_result.center - center) < 0.02
        assert abs(fit_result.radius - radius) < 0.01
        assert fit_result.inlier_ratio > 0.85

    def test_fit_cylinder_with_noisy_surface_points(self):
        random_generator = np.random.default_rng(21)

        radius = 0.06
        height = 0.24
        center = np.asarray([0.3, 0.05, -0.2], dtype=np.float64)

        theta = random_generator.uniform(0.0, 2.0 * math.pi, size=700)
        z = random_generator.uniform(-height / 2.0, height / 2.0, size=700)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points = np.stack([x, y, z], axis=1) + center
        points += random_generator.normal(scale=0.0015, size=points.shape)

        fit_result = fit_cylinder(points=points, distance_threshold=0.01)

        assert fit_result is not None
        assert abs(fit_result.radius - radius) < 0.01
        assert abs(float(np.dot(fit_result.axis_direction, np.array([0, 0, 1])))) > 0.97
        assert fit_result.height > 0.18
        assert fit_result.inlier_ratio > 0.8

    def test_fit_cuboid_with_surface_points(self):
        random_generator = np.random.default_rng(2)

        extents = np.asarray([0.20, 0.10, 0.06], dtype=np.float64)
        half_extents = extents / 2.0
        center = np.asarray([0.05, -0.1, 0.3], dtype=np.float64)
        angle = math.radians(25.0)
        rotation_matrix = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        local_points = []
        sample_count_per_face = 120
        for axis_index in range(3):
            for sign in (-1.0, 1.0):
                face_points = random_generator.uniform(
                    low=-half_extents,
                    high=half_extents,
                    size=(sample_count_per_face, 3),
                )
                face_points[:, axis_index] = sign * half_extents[axis_index]
                local_points.append(face_points)
        local_points_array = np.concatenate(local_points, axis=0)
        points = local_points_array @ rotation_matrix.T + center
        points += random_generator.normal(scale=0.001, size=points.shape)

        fit_result = fit_cuboid(points=points, distance_threshold=0.01)

        assert fit_result is not None
        assert np.linalg.norm(fit_result.center - center) < 0.03
        assert np.allclose(
            np.sort(fit_result.extents),
            np.sort(extents),
            atol=0.03,
        )
        assert fit_result.inlier_ratio > 0.7

    def test_select_best_shape_returns_highest_score(self):
        first = SphereFit(
            center=np.zeros(3),
            radius=1.0,
            inlier_indices=np.array([0, 1]),
            inlier_ratio=0.5,
            root_mean_square_error=0.01,
            score=0.21,
        )
        second = SphereFit(
            center=np.zeros(3),
            radius=1.0,
            inlier_indices=np.array([0, 1, 2]),
            inlier_ratio=0.9,
            root_mean_square_error=0.02,
            score=0.42,
        )
        selected = select_best_shape([first, second])
        assert selected is second

    def test_compute_fit_score_prefers_higher_inlier_ratio(self):
        score_a = compute_fit_score(
            inlier_ratio=0.5,
            root_mean_square_error=0.001,
            distance_threshold=0.01,
            complexity_penalty=0.0,
        )
        score_b = compute_fit_score(
            inlier_ratio=0.8,
            root_mean_square_error=0.001,
            distance_threshold=0.01,
            complexity_penalty=0.0,
        )
        assert score_b > score_a

    def test_select_best_shape_with_empty_input(self):
        assert select_best_shape([]) is None

    def test_select_best_shape_prefers_smaller_volume_when_scores_are_close(self):
        sphere = SphereFit(
            center=np.zeros(3),
            radius=0.4,
            inlier_indices=np.arange(10),
            inlier_ratio=0.8,
            root_mean_square_error=0.005,
            score=0.71,
        )
        cuboid = CuboidFit(
            center=np.zeros(3),
            rotation_matrix=np.eye(3),
            extents=np.array([0.18, 0.11, 0.07]),
            inlier_indices=np.arange(10),
            inlier_ratio=0.79,
            root_mean_square_error=0.006,
            score=0.69,
        )

        selected = select_best_shape([sphere, cuboid], score_tolerance=0.05)
        assert selected is cuboid

    def test_select_best_shape_keeps_best_score_when_gap_is_large(self):
        sphere = SphereFit(
            center=np.zeros(3),
            radius=0.4,
            inlier_indices=np.arange(10),
            inlier_ratio=0.9,
            root_mean_square_error=0.003,
            score=0.83,
        )
        cuboid = CuboidFit(
            center=np.zeros(3),
            rotation_matrix=np.eye(3),
            extents=np.array([0.18, 0.11, 0.07]),
            inlier_indices=np.arange(10),
            inlier_ratio=0.75,
            root_mean_square_error=0.010,
            score=0.60,
        )

        selected = select_best_shape([sphere, cuboid], score_tolerance=0.05)
        assert selected is sphere

    def test_select_best_shape_prefers_box_like_cuboid_when_scores_are_close(self):
        cylinder = CylinderFit(
            axis_center=np.zeros(3),
            axis_direction=np.array([0.0, 0.0, 1.0]),
            radius=0.08,
            height=0.24,
            inlier_indices=np.arange(10),
            inlier_ratio=0.82,
            root_mean_square_error=0.005,
            score=0.78,
        )
        cuboid = CuboidFit(
            center=np.zeros(3),
            rotation_matrix=np.eye(3),
            extents=np.array([0.24, 0.13, 0.08]),
            inlier_indices=np.arange(10),
            inlier_ratio=0.80,
            root_mean_square_error=0.006,
            score=0.75,
        )

        selected = select_best_shape(
            candidates=[cylinder, cuboid],
            prefer_cuboid_when_close=True,
            cuboid_preference_score_margin=0.05,
            cuboid_preference_inlier_ratio_tolerance=0.05,
        )
        assert selected is cuboid

    def test_select_best_shape_does_not_prefer_cuboid_for_cylinder_like_profile(self):
        cylinder = CylinderFit(
            axis_center=np.zeros(3),
            axis_direction=np.array([0.0, 0.0, 1.0]),
            radius=0.08,
            height=0.24,
            inlier_indices=np.arange(10),
            inlier_ratio=0.83,
            root_mean_square_error=0.004,
            score=0.79,
        )
        cylinder_like_cuboid = CuboidFit(
            center=np.zeros(3),
            rotation_matrix=np.eye(3),
            extents=np.array([0.16, 0.159, 0.30]),
            inlier_indices=np.arange(10),
            inlier_ratio=0.82,
            root_mean_square_error=0.005,
            score=0.76,
        )

        selected = select_best_shape(
            candidates=[cylinder, cylinder_like_cuboid],
            prefer_cuboid_when_close=True,
            cuboid_preference_score_margin=0.05,
            cuboid_preference_inlier_ratio_tolerance=0.05,
            cuboid_box_like_cross_section_asymmetry_threshold=0.12,
            cuboid_box_like_cube_axis_similarity_tolerance=0.12,
        )
        assert selected is cylinder

    def test_fit_sphere_rejects_radius_over_one_meter(self):
        random_generator = np.random.default_rng(9)

        center = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        radius = 1.25
        directions = random_generator.normal(size=(450, 3))
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        points = center + radius * directions
        points += random_generator.normal(scale=0.003, size=points.shape)

        fit_result = fit_sphere(
            points=points,
            distance_threshold=0.03,
            max_radius=1.0,
        )
        assert fit_result is None

    def test_fit_sphere_rejects_implausible_radius_for_small_patch(self):
        random_generator = np.random.default_rng(11)

        radius = 1.0
        xy = random_generator.uniform(-0.04, 0.04, size=(400, 2))
        z = np.sqrt(np.maximum(radius**2 - np.sum(np.square(xy), axis=1), 1e-9))
        points = np.column_stack((xy[:, 0], xy[:, 1], z))
        points += random_generator.normal(scale=0.0008, size=points.shape)

        unconstrained_fit = fit_sphere(
            points=points,
            distance_threshold=0.01,
            max_radius=1.2,
        )
        constrained_fit = fit_sphere(
            points=points,
            distance_threshold=0.01,
            max_radius=1.2,
            max_radius_to_bbox_diagonal_ratio=1.8,
            max_center_distance_to_bbox_diagonal_ratio=1.8,
        )

        assert unconstrained_fit is not None
        assert constrained_fit is None

    def test_fit_sphere_rejects_large_radius_for_small_observed_extent(self):
        random_generator = np.random.default_rng(15)

        radius = 0.36
        phi = random_generator.uniform(0.0, 0.18, size=600)
        theta = random_generator.uniform(0.0, 2.0 * math.pi, size=600)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        points = np.column_stack((x, y, z))
        points += random_generator.normal(scale=0.0008, size=points.shape)

        unconstrained_fit = fit_sphere(
            points=points,
            distance_threshold=0.01,
            max_radius=0.4,
            max_radius_to_bbox_diagonal_ratio=10.0,
            max_center_distance_to_bbox_diagonal_ratio=10.0,
        )
        constrained_fit = fit_sphere(
            points=points,
            distance_threshold=0.01,
            max_radius=0.4,
            max_radius_to_bbox_diagonal_ratio=10.0,
            max_radius_to_observed_extent_ratio=0.8,
            max_center_distance_to_bbox_diagonal_ratio=10.0,
        )

        assert unconstrained_fit is not None
        assert unconstrained_fit.radius > 0.2
        assert constrained_fit is None

    def test_fit_cylinder_rejects_radius_over_limit(self):
        random_generator = np.random.default_rng(33)

        radius = 0.52
        height = 0.35
        theta = random_generator.uniform(0.0, 2.0 * math.pi, size=500)
        z = random_generator.uniform(-height / 2.0, height / 2.0, size=500)
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
        points += random_generator.normal(scale=0.002, size=points.shape)

        fit_result = fit_cylinder(
            points=points,
            distance_threshold=0.02,
            max_radius=0.4,
            min_inlier_ratio=0.8,
        )
        assert fit_result is None

    def test_fit_cylinder_rejects_large_radius_for_small_cross_section_arc(self):
        random_generator = np.random.default_rng(55)

        radius = 1.1
        height = 0.28
        theta = random_generator.uniform(-0.12, 0.12, size=900)
        z = random_generator.uniform(-height / 2.0, height / 2.0, size=900)
        points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
        points += random_generator.normal(scale=0.001, size=points.shape)

        unconstrained_fit = fit_cylinder(
            points=points,
            distance_threshold=0.01,
            max_radius=2.0,
            max_radius_to_bbox_diagonal_ratio=10.0,
            max_axis_center_distance_to_bbox_diagonal_ratio=10.0,
        )
        constrained_fit = fit_cylinder(
            points=points,
            distance_threshold=0.01,
            max_radius=2.0,
            max_radius_to_bbox_diagonal_ratio=10.0,
            max_radius_to_cross_section_extent_ratio=0.95,
            max_axis_center_distance_to_bbox_diagonal_ratio=10.0,
        )

        assert unconstrained_fit is not None
        assert unconstrained_fit.radius > 0.5
        assert constrained_fit is None

    def test_fit_cuboid_rejects_extent_over_limit(self):
        random_generator = np.random.default_rng(41)

        extents = np.asarray([0.82, 0.22, 0.18], dtype=np.float64)
        half_extents = extents / 2.0
        samples = random_generator.uniform(
            low=-half_extents,
            high=half_extents,
            size=(1000, 3),
        )
        points = samples + random_generator.normal(scale=0.001, size=samples.shape)

        fit_result = fit_cuboid(
            points=points,
            distance_threshold=0.02,
            max_extent=0.6,
        )
        assert fit_result is None
