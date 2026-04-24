"""Utilities for fitting primitive shapes to 3D points.

This module provides robust least-squares based fitting for sphere and
cylinder models, plus a cuboid approximation based on oriented bounding boxes.
The implementation is intended for segmented object-level point sets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from typing_extensions import Optional, List, Union, Tuple

MAX_ACCEPTABLE_RADIUS_GROWTH_WITHOUT_STRONG_INLIER_GAIN = 1.35
MIN_INLIER_GAIN_TO_ACCEPT_LARGER_RADIUS = 0.03
MIN_ACCEPTABLE_CYLINDER_SCORE = 0.0


@dataclass
class SphereFit:
    """Fitted sphere model with inlier and quality information."""

    center: np.ndarray
    radius: float
    inlier_indices: np.ndarray
    inlier_ratio: float
    root_mean_square_error: float
    score: float


@dataclass
class CylinderFit:
    """Fitted cylinder model with inlier and quality information."""

    axis_center: np.ndarray
    axis_direction: np.ndarray
    radius: float
    height: float
    inlier_indices: np.ndarray
    inlier_ratio: float
    root_mean_square_error: float
    score: float


@dataclass
class CuboidFit:
    """Fitted cuboid model with inlier and quality information."""

    center: np.ndarray
    rotation_matrix: np.ndarray
    extents: np.ndarray
    inlier_indices: np.ndarray
    inlier_ratio: float
    root_mean_square_error: float
    score: float


FittedShape = Union[SphereFit, CylinderFit, CuboidFit]


@dataclass(frozen=True)
class CylinderFitConstraints:
    """Numerical constraints used during cylinder fitting and validation."""

    distance_threshold: float
    robust_loss: str
    max_radius: float
    max_height: float
    max_radius_to_bbox_diagonal_ratio: float
    max_radius_to_cross_section_extent_ratio: float
    max_axis_center_distance_to_bbox_diagonal_ratio: float


@dataclass(frozen=True)
class CylinderInitializationSettings:
    """Search settings controlling cylinder multi-start initialization."""

    max_initializations: int
    consensus_trials: int
    inlier_polishing_iterations: int


def sphere_residuals(parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute sphere residuals for optimization."""
    center = parameters[:3]
    radius = max(float(parameters[3]), 1e-9)
    point_distances = np.linalg.norm(points - center, axis=1)
    return point_distances - radius


def cylinder_residuals(parameters: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Compute cylinder residuals for optimization."""
    axis_point = parameters[:3]
    axis_direction = _normalize_vector(parameters[3:6])
    radius = max(float(parameters[6]), 1e-9)

    point_offsets = points - axis_point
    projected_offsets = np.outer(point_offsets @ axis_direction, axis_direction)
    radial_offsets = point_offsets - projected_offsets
    radial_distances = np.linalg.norm(radial_offsets, axis=1)
    return radial_distances - radius


def fit_sphere(
    points: np.ndarray,
    distance_threshold: float,
    robust_loss: str = "soft_l1",
    max_radius: float = np.inf,
    max_radius_to_bbox_diagonal_ratio: float = np.inf,
    max_radius_to_observed_extent_ratio: float = np.inf,
    max_center_distance_to_bbox_diagonal_ratio: float = np.inf,
    min_inlier_ratio: float = 0.0,
) -> Optional[SphereFit]:
    """Fit a sphere model and return model quality and inliers."""
    if len(points) < 4:
        return None
    if max_radius <= 0.0:
        return None
    if max_radius_to_observed_extent_ratio <= 0.0:
        return None
    if min_inlier_ratio < 0.0:
        return None

    initial_center = points.mean(axis=0)
    initial_radius = float(np.median(np.linalg.norm(points - initial_center, axis=1)))
    if initial_radius <= 0.0:
        return None
    if np.isfinite(max_radius):
        if max_radius <= 1e-6:
            return None
        initial_radius = min(initial_radius, max_radius * 0.999999)

    initial_parameters = np.asarray(
        [
            initial_center[0],
            initial_center[1],
            initial_center[2],
            initial_radius,
        ],
        dtype=np.float64,
    )

    lower_bounds = np.asarray(
        [-np.inf, -np.inf, -np.inf, 1e-6],
        dtype=np.float64,
    )
    upper_bounds = np.asarray([np.inf, np.inf, np.inf, max_radius], dtype=np.float64)

    optimization_result = least_squares(
        sphere_residuals,
        x0=initial_parameters,
        bounds=(lower_bounds, upper_bounds),
        args=(points,),
        loss=robust_loss,
        f_scale=distance_threshold,
        max_nfev=300,
    )

    absolute_residuals = np.abs(sphere_residuals(optimization_result.x, points))
    inlier_indices = np.where(absolute_residuals <= distance_threshold)[0]
    if len(inlier_indices) < 4:
        return None

    radius = float(optimization_result.x[3])
    if radius > max_radius:
        return None

    observed_max_extent = _point_cloud_max_extent(points[inlier_indices])
    if observed_max_extent > 1e-9:
        if radius / observed_max_extent > max_radius_to_observed_extent_ratio:
            return None

    bbox_diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if bbox_diagonal > 1e-9:
        if radius / bbox_diagonal > max_radius_to_bbox_diagonal_ratio:
            return None

        point_centroid = points.mean(axis=0)
        center_distance = float(
            np.linalg.norm(optimization_result.x[:3] - point_centroid)
        )
        if center_distance / bbox_diagonal > max_center_distance_to_bbox_diagonal_ratio:
            return None

    root_mean_square_error = float(
        np.sqrt(np.mean(np.square(absolute_residuals[inlier_indices])))
    )
    inlier_ratio = float(len(inlier_indices) / len(points))
    if inlier_ratio < min_inlier_ratio:
        return None
    score = compute_fit_score(
        inlier_ratio=inlier_ratio,
        root_mean_square_error=root_mean_square_error,
        distance_threshold=distance_threshold,
        complexity_penalty=0.0,
    )

    return SphereFit(
        center=optimization_result.x[:3].astype(np.float64),
        radius=radius,
        inlier_indices=inlier_indices.astype(np.int64),
        inlier_ratio=inlier_ratio,
        root_mean_square_error=root_mean_square_error,
        score=score,
    )


def fit_cylinder(
    points: np.ndarray,
    distance_threshold: float,
    robust_loss: str = "soft_l1",
    max_radius: float = np.inf,
    max_height: float = np.inf,
    max_radius_to_bbox_diagonal_ratio: float = np.inf,
    max_radius_to_cross_section_extent_ratio: float = np.inf,
    max_axis_center_distance_to_bbox_diagonal_ratio: float = np.inf,
    min_inlier_ratio: float = 0.0,
    max_initializations: int = 8,
    consensus_trials: int = 24,
    inlier_polishing_iterations: int = 0,
) -> Optional[CylinderFit]:
    """Fit a cylinder model and return model quality and inliers."""
    cylinder_fit_constraints = CylinderFitConstraints(
        distance_threshold=distance_threshold,
        robust_loss=robust_loss,
        max_radius=max_radius,
        max_height=max_height,
        max_radius_to_bbox_diagonal_ratio=max_radius_to_bbox_diagonal_ratio,
        max_radius_to_cross_section_extent_ratio=(
            max_radius_to_cross_section_extent_ratio
        ),
        max_axis_center_distance_to_bbox_diagonal_ratio=(
            max_axis_center_distance_to_bbox_diagonal_ratio
        ),
    )
    cylinder_initialization_settings = CylinderInitializationSettings(
        max_initializations=max_initializations,
        consensus_trials=consensus_trials,
        inlier_polishing_iterations=inlier_polishing_iterations,
    )
    if len(points) < 8:
        return None
    if cylinder_fit_constraints.max_radius <= 0.0:
        return None
    if cylinder_fit_constraints.max_height <= 0.0:
        return None
    if cylinder_fit_constraints.max_radius_to_cross_section_extent_ratio <= 0.0:
        return None
    if min_inlier_ratio < 0.0:
        return None
    if cylinder_initialization_settings.max_initializations <= 0:
        return None
    if cylinder_initialization_settings.consensus_trials < 0:
        return None
    if cylinder_initialization_settings.inlier_polishing_iterations < 0:
        return None
    if (
        np.isfinite(cylinder_fit_constraints.max_radius)
        and cylinder_fit_constraints.max_radius <= 1e-6
    ):
        return None

    initial_parameter_sets = _generate_cylinder_initial_parameter_sets(
        points=points,
        constraints=cylinder_fit_constraints,
        initialization_settings=cylinder_initialization_settings,
    )
    if len(initial_parameter_sets) == 0:
        return None

    lower_bounds = np.asarray(
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 1e-6], dtype=np.float64
    )
    upper_bounds = np.asarray(
        [
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            cylinder_fit_constraints.max_radius,
        ],
        dtype=np.float64,
    )

    best_fit: Optional[CylinderFit] = None
    for initial_parameters in initial_parameter_sets:
        candidate_fit = _fit_cylinder_from_initialization(
            points=points,
            initial_parameters=initial_parameters,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            constraints=cylinder_fit_constraints,
            initialization_settings=cylinder_initialization_settings,
        )
        if candidate_fit is None:
            continue

        if best_fit is None or _is_better_cylinder_fit(
            candidate_fit, best_fit, min_inlier_ratio
        ):
            best_fit = candidate_fit

    if best_fit is None:
        return None

    if best_fit.inlier_ratio < min_inlier_ratio:
        return None
    return best_fit


def _fit_cylinder_from_initialization(
    points: np.ndarray,
    initial_parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    constraints: CylinderFitConstraints,
    initialization_settings: CylinderInitializationSettings,
) -> Optional[CylinderFit]:
    """Run one cylinder optimization from one initialization and validate the result."""
    optimization_result = least_squares(
        cylinder_residuals,
        x0=initial_parameters,
        bounds=(lower_bounds, upper_bounds),
        args=(points,),
        loss=constraints.robust_loss,
        f_scale=constraints.distance_threshold,
        max_nfev=500,
    )
    optimized_parameters = np.asarray(optimization_result.x, dtype=np.float64)

    for _ in range(initialization_settings.inlier_polishing_iterations):
        residuals = np.abs(cylinder_residuals(optimized_parameters, points))
        inlier_indices = np.where(residuals <= constraints.distance_threshold)[0]
        if len(inlier_indices) < 8:
            break

        polishing_result = least_squares(
            cylinder_residuals,
            x0=optimized_parameters,
            bounds=(lower_bounds, upper_bounds),
            args=(points[inlier_indices],),
            loss="linear",
            f_scale=constraints.distance_threshold,
            max_nfev=250,
        )
        optimized_parameters = np.asarray(polishing_result.x, dtype=np.float64)

    return _build_cylinder_fit_from_parameters(
        parameters=optimized_parameters,
        points=points,
        constraints=constraints,
    )


def _build_cylinder_fit_from_parameters(
    parameters: np.ndarray,
    points: np.ndarray,
    constraints: CylinderFitConstraints,
) -> Optional[CylinderFit]:
    """Validate one optimized cylinder parameter vector and construct a fit object."""
    absolute_residuals = np.abs(cylinder_residuals(parameters, points))
    inlier_indices = np.where(absolute_residuals <= constraints.distance_threshold)[0]
    if len(inlier_indices) < 8:
        return None

    axis_direction = _normalize_vector(parameters[3:6])
    axis_point = np.asarray(parameters[:3], dtype=np.float64)
    axis_coordinates = (points - axis_point) @ axis_direction
    minimum_axis_coordinate = float(axis_coordinates.min())
    maximum_axis_coordinate = float(axis_coordinates.max())
    height = max(maximum_axis_coordinate - minimum_axis_coordinate, 1e-6)
    axis_center = axis_point + axis_direction * (
        0.5 * (maximum_axis_coordinate + minimum_axis_coordinate)
    )
    radius = float(parameters[6])
    if radius > constraints.max_radius:
        return None
    if height > constraints.max_height:
        return None

    bbox_diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    if bbox_diagonal > 1e-9:
        if radius / bbox_diagonal > constraints.max_radius_to_bbox_diagonal_ratio:
            return None
        point_centroid = points.mean(axis=0)
        center_distance = float(np.linalg.norm(axis_center - point_centroid))
        if (
            center_distance / bbox_diagonal
            > constraints.max_axis_center_distance_to_bbox_diagonal_ratio
        ):
            return None

    cross_section_max_extent = _cross_section_max_extent(
        points=points[inlier_indices],
        axis_center=axis_center,
        axis_direction=axis_direction,
    )
    if cross_section_max_extent > 1e-9:
        if (
            radius / cross_section_max_extent
            > constraints.max_radius_to_cross_section_extent_ratio
        ):
            return None

    root_mean_square_error = float(
        np.sqrt(np.mean(np.square(absolute_residuals[inlier_indices])))
    )
    inlier_ratio = float(len(inlier_indices) / len(points))
    score = compute_fit_score(
        inlier_ratio=inlier_ratio,
        root_mean_square_error=root_mean_square_error,
        distance_threshold=constraints.distance_threshold,
        complexity_penalty=0.02,
    )
    if score <= MIN_ACCEPTABLE_CYLINDER_SCORE:
        return None

    return CylinderFit(
        axis_center=axis_center.astype(np.float64),
        axis_direction=axis_direction.astype(np.float64),
        radius=radius,
        height=float(height),
        inlier_indices=inlier_indices.astype(np.int64),
        inlier_ratio=inlier_ratio,
        root_mean_square_error=root_mean_square_error,
        score=score,
    )


def _generate_cylinder_initial_parameter_sets(
    points: np.ndarray,
    constraints: CylinderFitConstraints,
    initialization_settings: CylinderInitializationSettings,
) -> List[np.ndarray]:
    """Generate distinct initialization vectors for cylinder multi-start optimization."""
    centroid = points.mean(axis=0)
    median_point = np.median(points, axis=0)

    axis_candidates: List[np.ndarray] = []
    for principal_axis in _principal_axes(points):
        _append_axis_if_distinct(axis_candidates, principal_axis)

    _append_axis_if_distinct(axis_candidates, np.asarray([1.0, 0.0, 0.0]))
    _append_axis_if_distinct(axis_candidates, np.asarray([0.0, 1.0, 0.0]))
    _append_axis_if_distinct(axis_candidates, np.asarray([0.0, 0.0, 1.0]))

    consensus_axis_candidates = _consensus_axis_candidates(
        points=points,
        axis_point=centroid,
        distance_threshold=constraints.distance_threshold,
        max_radius=constraints.max_radius,
        trial_count=initialization_settings.consensus_trials,
    )
    for consensus_axis, _, _ in consensus_axis_candidates:
        _append_axis_if_distinct(axis_candidates, consensus_axis)

    initial_parameter_sets: List[np.ndarray] = []
    axis_points = [centroid, median_point]
    for axis_direction in axis_candidates:
        for axis_point in axis_points:
            initial_radius = _initial_cylinder_radius(
                points=points,
                axis_point=axis_point,
                axis_direction=axis_direction,
                max_radius=constraints.max_radius,
            )
            if initial_radius is None:
                continue
            initial_parameter_sets.append(
                np.asarray(
                    [
                        axis_point[0],
                        axis_point[1],
                        axis_point[2],
                        axis_direction[0],
                        axis_direction[1],
                        axis_direction[2],
                        initial_radius,
                    ],
                    dtype=np.float64,
                )
            )
            if (
                len(initial_parameter_sets)
                >= initialization_settings.max_initializations
            ):
                return initial_parameter_sets
    return initial_parameter_sets


def _consensus_axis_candidates(
    points: np.ndarray,
    axis_point: np.ndarray,
    distance_threshold: float,
    max_radius: float,
    trial_count: int,
) -> List[Tuple[np.ndarray, float, float]]:
    """Return axis candidates ranked by provisional inlier ratio and residual error."""
    if trial_count <= 0 or len(points) < 8:
        return []

    random_generator = np.random.default_rng(0)
    provisional_candidates: List[Tuple[np.ndarray, float, float]] = []
    point_count = len(points)
    for _ in range(trial_count):
        first_index = int(random_generator.integers(0, point_count))
        second_index = int(random_generator.integers(0, point_count))
        if first_index == second_index:
            continue

        axis_direction = points[second_index] - points[first_index]
        axis_norm = float(np.linalg.norm(axis_direction))
        if axis_norm < 1e-9:
            continue
        axis_direction = axis_direction / axis_norm

        initial_radius = _initial_cylinder_radius(
            points=points,
            axis_point=axis_point,
            axis_direction=axis_direction,
            max_radius=max_radius,
        )
        if initial_radius is None:
            continue

        inlier_ratio, inlier_rmse = _provisional_cylinder_axis_quality(
            points=points,
            axis_point=axis_point,
            axis_direction=axis_direction,
            radius=initial_radius,
            distance_threshold=distance_threshold,
        )
        if inlier_ratio <= 0.0:
            continue

        provisional_candidates.append((axis_direction, inlier_ratio, inlier_rmse))

    provisional_candidates.sort(key=lambda candidate: (-candidate[1], candidate[2]))
    return provisional_candidates


def _provisional_cylinder_axis_quality(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    radius: float,
    distance_threshold: float,
) -> Tuple[float, float]:
    """Return provisional inlier ratio and error for one axis and radius hypothesis."""
    axis = _normalize_vector(axis_direction)
    point_offsets = points - axis_point
    projected_offsets = np.outer(point_offsets @ axis, axis)
    radial_offsets = point_offsets - projected_offsets
    radial_distances = np.linalg.norm(radial_offsets, axis=1)
    residuals = np.abs(radial_distances - radius)
    inlier_indices = np.where(residuals <= distance_threshold)[0]
    if len(inlier_indices) < 8:
        return 0.0, np.inf
    inlier_ratio = float(len(inlier_indices) / len(points))
    inlier_root_mean_square_error = float(
        np.sqrt(np.mean(np.square(residuals[inlier_indices])))
    )
    return inlier_ratio, inlier_root_mean_square_error


def _initial_cylinder_radius(
    points: np.ndarray,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    max_radius: float,
) -> Optional[float]:
    """Estimate a robust initial cylinder radius for one axis hypothesis."""
    normalized_axis = _normalize_vector(axis_direction)
    point_offsets = points - axis_point
    projected_offsets = np.outer(point_offsets @ normalized_axis, normalized_axis)
    radial_offsets = point_offsets - projected_offsets
    radius = float(np.median(np.linalg.norm(radial_offsets, axis=1)))
    if radius <= 0.0:
        return None
    if np.isfinite(max_radius):
        radius = min(radius, max_radius * 0.999999)
    return radius


def _append_axis_if_distinct(
    axis_candidates: List[np.ndarray],
    axis_direction: np.ndarray,
    min_angle_degrees: float = 8.0,
) -> None:
    """Append a normalized axis direction unless it duplicates an existing direction."""
    normalized_axis = _normalize_vector(axis_direction)
    min_cosine = float(np.cos(np.deg2rad(min_angle_degrees)))
    for existing_axis in axis_candidates:
        if abs(float(np.dot(normalized_axis, existing_axis))) >= min_cosine:
            return
    axis_candidates.append(normalized_axis)


def _is_better_cylinder_fit(
    candidate_fit: CylinderFit,
    best_fit: CylinderFit,
    minimum_target_inlier_ratio: float,
) -> bool:
    """Return whether one cylinder fit is better under inlier-target-aware ranking."""
    candidate_meets_inlier_target = (
        candidate_fit.inlier_ratio >= minimum_target_inlier_ratio
    )
    best_meets_inlier_target = best_fit.inlier_ratio >= minimum_target_inlier_ratio
    if candidate_meets_inlier_target and not best_meets_inlier_target:
        return True
    if best_meets_inlier_target and not candidate_meets_inlier_target:
        return False

    if candidate_meets_inlier_target and best_meets_inlier_target:
        score_delta = candidate_fit.score - best_fit.score
        if score_delta > 1e-4:
            return True
        if score_delta < -1e-4:
            return False
        return candidate_fit.root_mean_square_error < best_fit.root_mean_square_error

    inlier_ratio_delta = candidate_fit.inlier_ratio - best_fit.inlier_ratio
    if inlier_ratio_delta > 1e-4:
        if (
            candidate_fit.radius
            > (
                best_fit.radius
                * MAX_ACCEPTABLE_RADIUS_GROWTH_WITHOUT_STRONG_INLIER_GAIN
            )
            and inlier_ratio_delta < MIN_INLIER_GAIN_TO_ACCEPT_LARGER_RADIUS
        ):
            return False
        return True
    if inlier_ratio_delta < -1e-4:
        return False

    rmse_delta = candidate_fit.root_mean_square_error - best_fit.root_mean_square_error
    if rmse_delta < -1e-6:
        return True
    if rmse_delta > 1e-6:
        return False
    return candidate_fit.score > best_fit.score


def fit_cuboid(
    points: np.ndarray,
    distance_threshold: float,
    max_extent: float = np.inf,
    min_inlier_ratio: float = 0.0,
) -> Optional[CuboidFit]:
    """Fit a cuboid model and return model quality and inliers."""
    if len(points) < 8:
        return None
    if max_extent <= 0.0:
        return None
    if min_inlier_ratio < 0.0:
        return None

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    oriented_bounding_box = point_cloud.get_oriented_bounding_box(robust=True)

    center = np.asarray(oriented_bounding_box.center, dtype=np.float64)
    rotation_matrix = np.asarray(oriented_bounding_box.R, dtype=np.float64)
    extents = np.asarray(oriented_bounding_box.extent, dtype=np.float64)
    if np.any(extents > max_extent):
        return None

    surface_distances = point_to_oriented_box_surface_distance(
        points=points,
        center=center,
        rotation_matrix=rotation_matrix,
        extents=extents,
    )

    inlier_indices = np.where(surface_distances <= distance_threshold)[0]
    if len(inlier_indices) < 8:
        return None

    root_mean_square_error = float(
        np.sqrt(np.mean(np.square(surface_distances[inlier_indices])))
    )
    inlier_ratio = float(len(inlier_indices) / len(points))
    if inlier_ratio < min_inlier_ratio:
        return None
    score = compute_fit_score(
        inlier_ratio=inlier_ratio,
        root_mean_square_error=root_mean_square_error,
        distance_threshold=distance_threshold,
        complexity_penalty=0.02,
    )

    return CuboidFit(
        center=center,
        rotation_matrix=rotation_matrix,
        extents=extents,
        inlier_indices=inlier_indices.astype(np.int64),
        inlier_ratio=inlier_ratio,
        root_mean_square_error=root_mean_square_error,
        score=score,
    )


def select_best_shape(
    candidates: List[FittedShape],
    score_tolerance: float = 0.05,
    prefer_cuboid_when_close: bool = False,
    cuboid_preference_score_margin: float = 0.06,
    cuboid_preference_inlier_ratio_tolerance: float = 0.05,
    cuboid_box_like_cross_section_asymmetry_threshold: float = 0.12,
    cuboid_box_like_cube_axis_similarity_tolerance: float = 0.12,
) -> Optional[FittedShape]:
    """Return the best shape candidate with volume-aware tie-breaking."""
    if len(candidates) == 0:
        return None

    best_candidate_by_score = max(candidates, key=lambda candidate: candidate.score)
    best_score = max(candidate.score for candidate in candidates)

    if prefer_cuboid_when_close:
        cuboid_candidates = [
            candidate for candidate in candidates if isinstance(candidate, CuboidFit)
        ]
        if len(cuboid_candidates) > 0:
            best_cuboid = max(cuboid_candidates, key=lambda candidate: candidate.score)
            cuboid_score_gap = best_score - best_cuboid.score
            cuboid_inlier_ratio_gap = (
                best_candidate_by_score.inlier_ratio - best_cuboid.inlier_ratio
            )
            if (
                cuboid_score_gap <= cuboid_preference_score_margin
                and cuboid_inlier_ratio_gap <= cuboid_preference_inlier_ratio_tolerance
                and _is_box_like_extent_profile(
                    extents=best_cuboid.extents,
                    cross_section_asymmetry_threshold=(
                        cuboid_box_like_cross_section_asymmetry_threshold
                    ),
                    cube_axis_similarity_tolerance=(
                        cuboid_box_like_cube_axis_similarity_tolerance
                    ),
                )
            ):
                return best_cuboid

    close_candidates = [
        candidate
        for candidate in candidates
        if (best_score - candidate.score) <= score_tolerance
    ]
    if len(close_candidates) == 1:
        return close_candidates[0]

    return min(
        close_candidates,
        key=lambda candidate: (
            _fitted_shape_volume(candidate),
            -candidate.score,
        ),
    )


def point_to_oriented_box_surface_distance(
    points: np.ndarray,
    center: np.ndarray,
    rotation_matrix: np.ndarray,
    extents: np.ndarray,
) -> np.ndarray:
    """Compute absolute distance from points to cuboid surface."""
    half_extents = extents / 2.0
    local_points = (points - center) @ rotation_matrix
    absolute_local_points = np.abs(local_points)

    outside_offset = np.maximum(absolute_local_points - half_extents, 0.0)
    outside_distance = np.linalg.norm(outside_offset, axis=1)

    inside_margin = half_extents - absolute_local_points
    inside_distance = np.min(inside_margin, axis=1)
    signed_distance = np.where(
        np.any(absolute_local_points > half_extents, axis=1),
        outside_distance,
        -inside_distance,
    )
    return np.abs(signed_distance)


def compute_fit_score(
    inlier_ratio: float,
    root_mean_square_error: float,
    distance_threshold: float,
    complexity_penalty: float,
) -> float:
    """Compute a scalar quality score for model comparison."""
    normalization = max(distance_threshold, 1e-6)
    normalized_error = root_mean_square_error / normalization
    return inlier_ratio - 0.35 * normalized_error - complexity_penalty


def _principal_axes(points: np.ndarray) -> List[np.ndarray]:
    centered_points = points - points.mean(axis=0)
    covariance_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return [
        _normalize_vector(eigenvectors[:, axis_index]) for axis_index in sorted_indices
    ]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _fitted_shape_volume(candidate: FittedShape) -> float:
    """Return primitive volume for volume-aware candidate comparison."""
    if isinstance(candidate, SphereFit):
        return float((4.0 / 3.0) * np.pi * candidate.radius**3)
    if isinstance(candidate, CylinderFit):
        return float(np.pi * candidate.radius**2 * candidate.height)
    return float(np.prod(candidate.extents))


def _cross_section_max_extent(
    points: np.ndarray,
    axis_center: np.ndarray,
    axis_direction: np.ndarray,
) -> float:
    """Return largest observed cross-section extent orthogonal to the cylinder axis."""
    basis_a, basis_b = _orthogonal_basis(axis_direction)
    centered_points = points - axis_center
    coordinate_a = centered_points @ basis_a
    coordinate_b = centered_points @ basis_b
    extent_a = float(coordinate_a.max() - coordinate_a.min())
    extent_b = float(coordinate_b.max() - coordinate_b.min())
    return max(extent_a, extent_b)


def _orthogonal_basis(axis_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create two unit vectors orthogonal to the provided axis direction."""
    axis = _normalize_vector(axis_direction)
    helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, axis))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

    basis_a = np.cross(axis, helper)
    basis_a = basis_a / max(np.linalg.norm(basis_a), 1e-9)
    basis_b = np.cross(axis, basis_a)
    basis_b = basis_b / max(np.linalg.norm(basis_b), 1e-9)
    return basis_a.astype(np.float64), basis_b.astype(np.float64)


def _point_cloud_max_extent(points: np.ndarray) -> float:
    """Return largest principal-axis extent of a point cloud."""
    if len(points) == 0:
        return 0.0
    if len(points) < 3:
        axis_aligned_extents = points.max(axis=0) - points.min(axis=0)
        return float(np.max(axis_aligned_extents))

    centered_points = points - points.mean(axis=0)
    covariance_matrix = np.cov(centered_points.T)
    _, eigenvectors = np.linalg.eigh(covariance_matrix)
    principal_coordinates = centered_points @ eigenvectors
    principal_extents = principal_coordinates.max(axis=0) - principal_coordinates.min(
        axis=0
    )
    return float(np.max(principal_extents))


def _is_box_like_extent_profile(
    extents: np.ndarray,
    cross_section_asymmetry_threshold: float,
    cube_axis_similarity_tolerance: float,
) -> bool:
    """Return whether extents represent a box-like profile rather than axisymmetry."""
    sorted_extents = np.sort(np.asarray(extents, dtype=np.float64))
    smallest_extent = float(sorted_extents[0])
    middle_extent = float(sorted_extents[1])
    largest_extent = float(sorted_extents[2])
    if largest_extent <= 1e-9:
        return False

    cross_section_asymmetry = abs(middle_extent - smallest_extent) / max(
        middle_extent, 1e-9
    )
    cube_axis_similarity = (largest_extent - smallest_extent) / largest_extent

    is_cube_like = cube_axis_similarity <= cube_axis_similarity_tolerance
    has_rectangular_cross_section = (
        cross_section_asymmetry >= cross_section_asymmetry_threshold
    )
    return bool(is_cube_like or has_rectangular_cross_section)
