"""Shape estimation for object hypotheses.

This module provides a minimal, pure-Python annotator that fits primitive
shape models to segmented object point clouds.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from py_trees.common import Status
from typing_extensions import List, Optional, Tuple, Dict, Any

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.types.annotation import Cuboid, Cylinder, Shape, Sphere
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.shape_fitting import (
    CuboidFit,
    CylinderFit,
    FittedShape,
    SphereFit,
    compute_fit_score,
    fit_cuboid,
    fit_cylinder,
    fit_sphere,
    point_to_oriented_box_surface_distance,
    select_best_shape,
)
from robokudo.utils.transform import get_quaternion_from_rotation_matrix
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import (
    Box as SemDTBox,
    Cylinder as SemDTCylinder,
    Scale,
    Sphere as SemDTSphere,
)


@dataclass
class CylinderAxisCandidate:
    """Candidate axis direction together with a refitted cylinder."""

    source: str
    axis_direction: np.ndarray
    fit: CylinderFit


class ShapeEstimatorAnnotator(ThreadedAnnotator):
    """Estimate one primitive shape per object hypothesis point cloud.

    Overview:
    1. Validate input cloud (`minimum_point_count`) and optionally run
       statistical outlier removal.
    2. Fit enabled primitives (sphere/cylinder/cuboid) on the same filtered
       points.
    3. Apply post-fit stabilization:
       - Cylinder: optional axis stabilization from principal/preferred axes.
       - Cuboid: optional in-plane orientation stabilization for near-square faces.
    4. Rank accepted candidates with `select_best_shape(...)`.
    5. Convert the winning fit to a RoboKudo annotation (`Sphere`, `Cylinder`,
       or `Cuboid`), keep inlier indices mapped to the original object cloud,
       and publish visualization geometries.

    Parameter groups:
    - Global data quality: `minimum_point_count`, outlier removal settings.
    - Shared fit behavior: `distance_threshold`, `robust_loss`.
    - Shape gates:
      sphere/cylinder/cuboid enable flags plus per-shape size and ratio limits.
    - Candidate selection: `selection_score_tolerance` and cuboid preference
      settings for close scores.
    - Stabilization controls: cylinder axis and cuboid orientation stabilization.

    Practical tuning order:
    1. Set hard physical limits (max radii/heights/extents) for your scene.
    2. Tune inlier thresholds (`distance_threshold`, per-shape min inlier ratio).
    3. Tune selection preference parameters.
    4. Only then tune stabilization triggers/tolerances.

    This annotator intentionally favors robust, bounded fits for household-scale
    objects.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for shape estimation."""

        class Parameters:
            """Parameter container for shape estimation."""

            def __init__(self) -> None:
                self.minimum_point_count: int = 80
                """Minimum number of points required for fitting."""

                self.distance_threshold: float = 0.01
                """Inlier threshold used for all shape residuals."""

                self.robust_loss: str = "soft_l1"
                """Robust loss used for non-linear least squares."""

                self.minimum_inlier_ratio: float = 0.5
                """Minimum inlier ratio required for publishing a shape."""

                self.remove_statistical_outliers: bool = False
                """Whether statistical outlier removal is applied before fitting."""

                self.outlier_neighbor_count: int = 30
                """Neighbor count for statistical outlier removal."""

                self.outlier_standard_deviation_ratio: float = 2.0
                """Standard deviation ratio for statistical outlier removal."""

                self.fit_sphere: bool = False
                """Whether sphere fitting is enabled."""

                self.max_sphere_radius_meters: float = 0.4
                """Maximum allowed sphere radius."""

                self.max_sphere_radius_to_bbox_diagonal_ratio: float = 1.0
                """Maximum allowed ratio between sphere radius and point cloud diagonal."""

                self.max_sphere_radius_to_observed_extent_ratio: float = 0.8
                """Maximum allowed ratio between sphere radius and observed cloud extent."""

                self.max_sphere_center_distance_to_bbox_diagonal_ratio: float = 0.9
                """Maximum allowed ratio between center offset and point cloud diagonal."""

                self.fit_cylinder: bool = True
                """Whether cylinder fitting is enabled."""

                self.cylinder_minimum_inlier_ratio: float = 0.87
                """Minimum inlier ratio required for cylinder candidates."""

                self.max_cylinder_radius_meters: float = 0.4
                """Maximum allowed cylinder radius."""

                self.max_cylinder_height_meters: float = 1.0
                """Maximum allowed cylinder height."""

                self.max_cylinder_radius_to_bbox_diagonal_ratio: float = 1.0
                """Maximum allowed ratio between cylinder radius and cloud diagonal."""

                self.max_cylinder_radius_to_cross_section_extent_ratio: float = 0.95
                """Maximum allowed ratio between cylinder radius and observed cross-section extent."""

                self.max_cylinder_center_distance_to_bbox_diagonal_ratio: float = 0.8
                """Maximum allowed ratio between axis center offset and cloud diagonal."""

                self.cylinder_max_initializations: int = 12
                """Maximum number of cylinder initialization hypotheses for multi-start optimization."""

                self.cylinder_consensus_trials: int = 30
                """Number of random axis hypotheses used to seed cylinder multi-start optimization."""

                self.cylinder_inlier_polishing_iterations: int = 0
                """Number of post-optimization inlier-only polishing iterations for cylinder fitting."""

                self.stabilize_cylinder_axis_with_principal_axis: bool = True
                """Whether cylinder axis should be stabilized using point principal axis."""

                self.cylinder_axis_stabilization_trigger_degrees: float = 8.0
                """Minimum axis deviation angle required to trigger cylinder stabilization."""

                self.cylinder_axis_stabilization_max_score_drop: float = 0.08
                """Maximum tolerated score drop for accepting stabilized cylinder."""

                self.prefer_upright_cylinder_axis: bool = True
                """Whether an upright preferred axis should be favored when scores are close."""

                self.preferred_cylinder_axis_direction: Tuple[float, float, float] = (
                    0.0,
                    0.0,
                    1.0,
                )
                """Preferred global cylinder axis direction used as stabilization prior."""

                self.upright_cylinder_axis_score_tolerance: float = 0.05
                """Maximum score gap for preferring a more upright cylinder axis."""

                self.fit_cuboid: bool = True
                """Whether cuboid fitting is enabled."""

                self.max_cuboid_extent_meters: float = 0.6
                """Maximum allowed cuboid edge length for any axis."""

                self.cuboid_distance_threshold: float = 0.015
                """Inlier threshold used specifically for cuboid fitting."""

                self.cuboid_minimum_inlier_ratio: float = 0.35
                """Minimum inlier ratio required for cuboid candidates."""

                self.selection_score_tolerance: float = 0.05
                """Maximum score gap for considering two fits equivalent."""

                self.prefer_cuboid_when_score_close: bool = True
                """Whether cuboids should be preferred when scores are close."""

                self.cuboid_preference_score_margin: float = 0.06
                """Maximum score gap to still prefer a box-like cuboid candidate."""

                self.cuboid_preference_inlier_ratio_tolerance: float = 0.05
                """Maximum inlier-ratio gap to still prefer a cuboid."""

                self.cuboid_box_like_cross_section_asymmetry_threshold: float = 0.12
                """Minimum cross-section asymmetry to classify a cuboid as box-like."""

                self.cuboid_box_like_cube_axis_similarity_tolerance: float = 0.12
                """Maximum axis spread ratio to classify a cuboid as cube-like."""

                self.stabilize_ambiguous_cuboid_orientation: bool = True
                """Whether ambiguous cuboid in-plane orientation should be stabilized."""

                self.cuboid_ambiguous_in_plane_extent_relative_difference: float = 0.3
                """Maximum relative in-plane extent difference treated as orientation-ambiguous."""

                self.log_candidate_metrics: bool = True
                """Whether candidate metrics are logged on debug level."""

                self.log_rejection_reasons: bool = True
                """Whether shape rejection reasons are logged when metrics logging is enabled."""

        parameters = Parameters()

    def __init__(
        self,
        name: str = "ShapeEstimatorAnnotator",
        descriptor: "ShapeEstimatorAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the shape estimator annotator."""
        super().__init__(name=name, descriptor=descriptor)

    def compute(self) -> Status:
        """Estimate shapes and publish annotations and 3D visualizations."""
        object_hypotheses = self.get_cas().filter_annotations_by_type(ObjectHypothesis)

        total_object_hypotheses = 0
        updated_object_hypotheses = 0
        visual_geometries: List[Dict[str, Any]] = []

        for object_hypothesis in object_hypotheses:
            total_object_hypotheses += 1
            estimation_result = self._estimate_shape_for_object_hypothesis(
                object_hypothesis=object_hypothesis
            )
            if estimation_result is None:
                continue

            shape_annotation, object_visual_geometries = estimation_result
            object_hypothesis.annotations.append(shape_annotation)
            visual_geometries.extend(object_visual_geometries)
            updated_object_hypotheses += 1

        if len(visual_geometries) > 0:
            self.get_annotator_output_struct().set_geometries(visual_geometries)

        self.feedback_message = (
            f"Estimated shapes for {updated_object_hypotheses}/"
            f"{total_object_hypotheses} object hypotheses."
        )
        return Status.SUCCESS

    def _estimate_shape_for_object_hypothesis(
        self,
        object_hypothesis: ObjectHypothesis,
    ) -> Optional[Tuple[Shape, List[Dict[str, Any]]]]:
        """Estimate one shape candidate for a single object hypothesis."""
        point_cloud = object_hypothesis.points
        if point_cloud is None:
            return None
        if not isinstance(point_cloud, o3d.geometry.PointCloud):
            return None
        if len(point_cloud.points) < self.descriptor.parameters.minimum_point_count:
            return None

        filtered_cloud, retained_point_indices = self._prepare_point_cloud(point_cloud)
        if len(filtered_cloud.points) < self.descriptor.parameters.minimum_point_count:
            return None

        points = np.asarray(filtered_cloud.points, dtype=np.float64)
        candidates: List[FittedShape] = []

        if self.descriptor.parameters.fit_sphere:
            sphere_fit = fit_sphere(
                points=points,
                distance_threshold=self.descriptor.parameters.distance_threshold,
                robust_loss=self.descriptor.parameters.robust_loss,
                max_radius=self.descriptor.parameters.max_sphere_radius_meters,
                max_radius_to_bbox_diagonal_ratio=(
                    self.descriptor.parameters.max_sphere_radius_to_bbox_diagonal_ratio
                ),
                max_radius_to_observed_extent_ratio=(
                    self.descriptor.parameters.max_sphere_radius_to_observed_extent_ratio
                ),
                max_center_distance_to_bbox_diagonal_ratio=(
                    self.descriptor.parameters.max_sphere_center_distance_to_bbox_diagonal_ratio
                ),
                min_inlier_ratio=self.descriptor.parameters.minimum_inlier_ratio,
            )
            if sphere_fit is None:
                self._log_rejected_candidate(
                    object_hypothesis=object_hypothesis,
                    shape_name="Sphere",
                )
            else:
                candidates.append(sphere_fit)
                self._log_candidate_metrics(object_hypothesis, sphere_fit)

        if self.descriptor.parameters.fit_cylinder:
            cylinder_fit = fit_cylinder(
                points=points,
                distance_threshold=self.descriptor.parameters.distance_threshold,
                robust_loss=self.descriptor.parameters.robust_loss,
                max_radius=self.descriptor.parameters.max_cylinder_radius_meters,
                max_height=self.descriptor.parameters.max_cylinder_height_meters,
                max_radius_to_bbox_diagonal_ratio=(
                    self.descriptor.parameters.max_cylinder_radius_to_bbox_diagonal_ratio
                ),
                max_radius_to_cross_section_extent_ratio=(
                    self.descriptor.parameters.max_cylinder_radius_to_cross_section_extent_ratio
                ),
                max_axis_center_distance_to_bbox_diagonal_ratio=(
                    self.descriptor.parameters.max_cylinder_center_distance_to_bbox_diagonal_ratio
                ),
                min_inlier_ratio=self.descriptor.parameters.cylinder_minimum_inlier_ratio,
                max_initializations=self.descriptor.parameters.cylinder_max_initializations,
                consensus_trials=self.descriptor.parameters.cylinder_consensus_trials,
                inlier_polishing_iterations=self.descriptor.parameters.cylinder_inlier_polishing_iterations,
            )
            if cylinder_fit is not None:
                cylinder_fit = self._stabilize_cylinder_axis_if_tilted(
                    cylinder_fit=cylinder_fit,
                    points=points,
                )
            if cylinder_fit is None:
                self._log_rejected_candidate(
                    object_hypothesis=object_hypothesis,
                    shape_name="Cylinder",
                )
            else:
                candidates.append(cylinder_fit)
                self._log_candidate_metrics(object_hypothesis, cylinder_fit)

        if self.descriptor.parameters.fit_cuboid:
            cuboid_fit = fit_cuboid(
                points=points,
                distance_threshold=self.descriptor.parameters.cuboid_distance_threshold,
                max_extent=self.descriptor.parameters.max_cuboid_extent_meters,
                min_inlier_ratio=self.descriptor.parameters.cuboid_minimum_inlier_ratio,
            )
            if cuboid_fit is not None:
                cuboid_fit = self._stabilize_cuboid_orientation_if_ambiguous(
                    cuboid_fit=cuboid_fit,
                    points=points,
                )
            if cuboid_fit is None:
                self._log_rejected_candidate(
                    object_hypothesis=object_hypothesis,
                    shape_name="Cuboid",
                )
            else:
                candidates.append(cuboid_fit)
                self._log_candidate_metrics(object_hypothesis, cuboid_fit)

        best_fit = select_best_shape(
            candidates=candidates,
            score_tolerance=self.descriptor.parameters.selection_score_tolerance,
            prefer_cuboid_when_close=self.descriptor.parameters.prefer_cuboid_when_score_close,
            cuboid_preference_score_margin=(
                self.descriptor.parameters.cuboid_preference_score_margin
            ),
            cuboid_preference_inlier_ratio_tolerance=(
                self.descriptor.parameters.cuboid_preference_inlier_ratio_tolerance
            ),
            cuboid_box_like_cross_section_asymmetry_threshold=(
                self.descriptor.parameters.cuboid_box_like_cross_section_asymmetry_threshold
            ),
            cuboid_box_like_cube_axis_similarity_tolerance=(
                self.descriptor.parameters.cuboid_box_like_cube_axis_similarity_tolerance
            ),
        )
        if best_fit is None:
            return None
        minimum_inlier_ratio = self._minimum_inlier_ratio_for_fit(best_fit)
        if best_fit.inlier_ratio < minimum_inlier_ratio:
            return None
        self._log_selected_candidate(object_hypothesis, best_fit)

        inlier_indices_in_filtered_cloud = retained_point_indices[
            best_fit.inlier_indices
        ]
        inlier_indices_in_original_object_cloud = self._map_to_object_indices(
            object_hypothesis=object_hypothesis,
            local_indices=inlier_indices_in_filtered_cloud,
        )

        shape_annotation = self._fit_to_annotation(best_fit)
        shape_annotation.source = self.name
        shape_annotation.inliers = inlier_indices_in_original_object_cloud
        visual_geometries = self._create_visualization_geometries(
            object_hypothesis=object_hypothesis,
            filtered_cloud=filtered_cloud,
            best_fit=best_fit,
        )
        return shape_annotation, visual_geometries

    def _prepare_point_cloud(
        self, point_cloud: o3d.geometry.PointCloud
    ) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
        """Return a filtered point cloud and retained index mapping."""
        filtered_cloud = copy.deepcopy(point_cloud)
        retained_point_indices = np.arange(len(filtered_cloud.points), dtype=np.int64)

        if self.descriptor.parameters.remove_statistical_outliers:
            _, inlier_indices = filtered_cloud.remove_statistical_outlier(
                nb_neighbors=self.descriptor.parameters.outlier_neighbor_count,
                std_ratio=self.descriptor.parameters.outlier_standard_deviation_ratio,
            )
            inlier_indices_as_array = np.asarray(inlier_indices, dtype=np.int64)
            filtered_cloud = filtered_cloud.select_by_index(
                inlier_indices_as_array.tolist()
            )
            retained_point_indices = retained_point_indices[inlier_indices_as_array]

        return filtered_cloud, retained_point_indices

    def _stabilize_cylinder_axis_if_tilted(
        self, cylinder_fit: CylinderFit, points: np.ndarray
    ) -> CylinderFit:
        """Stabilize cylinder axis using principal axes and optional upright prior."""
        if not self.descriptor.parameters.stabilize_cylinder_axis_with_principal_axis:
            return cylinder_fit
        if len(points) < 8:
            return cylinder_fit

        axis_candidates = self._collect_cylinder_axis_candidates(points)
        candidate_fits: List[CylinderAxisCandidate] = [
            CylinderAxisCandidate(
                source="original",
                axis_direction=self._normalized_vector(cylinder_fit.axis_direction),
                fit=cylinder_fit,
            )
        ]

        for source, axis_direction in axis_candidates:
            aligned_axis_direction = np.asarray(axis_direction, dtype=np.float64).copy()
            if float(np.dot(aligned_axis_direction, cylinder_fit.axis_direction)) < 0.0:
                aligned_axis_direction *= -1.0

            axis_deviation_degrees = self._axis_angle_degrees(
                cylinder_fit.axis_direction, aligned_axis_direction
            )
            if axis_deviation_degrees < 1.0:
                continue

            refitted_cylinder = self._refit_cylinder_with_fixed_axis(
                points=points,
                fixed_axis_direction=aligned_axis_direction,
            )
            if refitted_cylinder is None:
                continue

            candidate_fits.append(
                CylinderAxisCandidate(
                    source=source,
                    axis_direction=aligned_axis_direction,
                    fit=refitted_cylinder,
                )
            )

        selected_candidate = self._select_cylinder_axis_candidate(candidate_fits)
        if selected_candidate.source == "original":
            return cylinder_fit

        axis_deviation_degrees = self._axis_angle_degrees(
            cylinder_fit.axis_direction, selected_candidate.axis_direction
        )
        if (
            axis_deviation_degrees
            <= self.descriptor.parameters.cylinder_axis_stabilization_trigger_degrees
            and selected_candidate.fit.score <= cylinder_fit.score
        ):
            return cylinder_fit

        minimum_accepted_score = (
            cylinder_fit.score
            - self.descriptor.parameters.cylinder_axis_stabilization_max_score_drop
        )
        if selected_candidate.fit.score < minimum_accepted_score:
            return cylinder_fit

        if self.descriptor.parameters.log_candidate_metrics:
            self.rk_logger.info(
                f"{self.name} stabilized cylinder axis: "
                f"source={selected_candidate.source}, "
                f"deviation={axis_deviation_degrees:.2f}deg, "
                f"score_before={cylinder_fit.score:.3f}, "
                f"score_after={selected_candidate.fit.score:.3f}"
            )
        return selected_candidate.fit

    def _collect_cylinder_axis_candidates(
        self, points: np.ndarray
    ) -> List[Tuple[str, np.ndarray]]:
        """Return candidate cylinder axis directions for stabilization."""
        candidate_axes: List[Tuple[str, np.ndarray]] = []
        principal_axes = self._point_cloud_principal_axes(points)
        for principal_axis_index, principal_axis in enumerate(principal_axes):
            candidate_axes.append(
                (f"principal_axis_{principal_axis_index + 1}", principal_axis)
            )

        preferred_axis_direction = self._preferred_cylinder_axis_direction()
        if preferred_axis_direction is not None:
            candidate_axes.append(("preferred_axis", preferred_axis_direction))

        distinct_axes: List[Tuple[str, np.ndarray]] = []
        for source, axis_direction in candidate_axes:
            if self._contains_similar_axis_direction(distinct_axes, axis_direction):
                continue
            distinct_axes.append((source, axis_direction))
        return distinct_axes

    def _contains_similar_axis_direction(
        self,
        existing_axis_candidates: List[Tuple[str, np.ndarray]],
        candidate_axis_direction: np.ndarray,
        similarity_threshold_degrees: float = 8.0,
    ) -> bool:
        """Return whether candidate axis is similar to one axis already in the list."""
        for _, existing_axis_direction in existing_axis_candidates:
            angle = self._axis_angle_degrees(
                existing_axis_direction, candidate_axis_direction
            )
            if angle <= similarity_threshold_degrees:
                return True
        return False

    def _preferred_cylinder_axis_direction(self) -> Optional[np.ndarray]:
        """Return configured preferred cylinder axis direction if valid."""
        if not self.descriptor.parameters.prefer_upright_cylinder_axis:
            return None

        preferred_axis_direction = np.asarray(
            self.descriptor.parameters.preferred_cylinder_axis_direction,
            dtype=np.float64,
        )
        if np.linalg.norm(preferred_axis_direction) < 1e-9:
            return None
        return self._normalized_vector(preferred_axis_direction)

    def _select_cylinder_axis_candidate(
        self, candidate_fits: List[CylinderAxisCandidate]
    ) -> CylinderAxisCandidate:
        """Select one cylinder axis candidate by score and optional upright preference."""
        best_candidate_by_score = max(
            candidate_fits, key=lambda candidate: candidate.fit.score
        )
        if len(candidate_fits) == 1:
            return best_candidate_by_score

        preferred_axis_direction = self._preferred_cylinder_axis_direction()
        if preferred_axis_direction is None:
            return best_candidate_by_score

        score_tolerance = (
            self.descriptor.parameters.upright_cylinder_axis_score_tolerance
        )
        close_candidates = [
            candidate
            for candidate in candidate_fits
            if (best_candidate_by_score.fit.score - candidate.fit.score)
            <= score_tolerance
        ]
        if len(close_candidates) == 0:
            return best_candidate_by_score

        return min(
            close_candidates,
            key=lambda candidate: (
                self._axis_angle_degrees(
                    candidate.axis_direction, preferred_axis_direction
                ),
                -candidate.fit.score,
            ),
        )

    def _refit_cylinder_with_fixed_axis(
        self, points: np.ndarray, fixed_axis_direction: np.ndarray
    ) -> Optional[CylinderFit]:
        """Refit cylinder radius and finite height while keeping axis direction fixed."""
        if len(points) < 8:
            return None

        axis_direction = self._normalized_vector(fixed_axis_direction)
        axis_point = points.mean(axis=0)

        point_offsets = points - axis_point
        projected_offsets = np.outer(point_offsets @ axis_direction, axis_direction)
        radial_offsets = point_offsets - projected_offsets
        radial_distances = np.linalg.norm(radial_offsets, axis=1)
        radius = float(np.median(radial_distances))
        if radius <= 1e-9:
            return None
        if radius > self.descriptor.parameters.max_cylinder_radius_meters:
            return None

        absolute_residuals = np.abs(radial_distances - radius)
        inlier_indices = np.where(
            absolute_residuals <= self.descriptor.parameters.distance_threshold
        )[0]
        if len(inlier_indices) < 8:
            return None

        refined_radius = float(np.median(radial_distances[inlier_indices]))
        if refined_radius <= 1e-9:
            return None
        radius = refined_radius
        absolute_residuals = np.abs(radial_distances - radius)
        inlier_indices = np.where(
            absolute_residuals <= self.descriptor.parameters.distance_threshold
        )[0]
        if len(inlier_indices) < 8:
            return None

        axis_coordinates = (points - axis_point) @ axis_direction
        minimum_axis_coordinate = float(axis_coordinates.min())
        maximum_axis_coordinate = float(axis_coordinates.max())
        height = max(maximum_axis_coordinate - minimum_axis_coordinate, 1e-6)
        if height > self.descriptor.parameters.max_cylinder_height_meters:
            return None
        axis_center = axis_point + axis_direction * (
            0.5 * (maximum_axis_coordinate + minimum_axis_coordinate)
        )

        bbox_diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if bbox_diagonal > 1e-9:
            if (
                radius / bbox_diagonal
                > self.descriptor.parameters.max_cylinder_radius_to_bbox_diagonal_ratio
            ):
                return None

            point_centroid = points.mean(axis=0)
            center_distance = float(np.linalg.norm(axis_center - point_centroid))
            if (
                center_distance / bbox_diagonal
                > self.descriptor.parameters.max_cylinder_center_distance_to_bbox_diagonal_ratio
            ):
                return None

        cross_section_max_extent = self._cross_section_max_extent_for_axis(
            points=points[inlier_indices],
            axis_center=axis_center,
            axis_direction=axis_direction,
        )
        if cross_section_max_extent > 1e-9:
            if (
                radius / cross_section_max_extent
                > self.descriptor.parameters.max_cylinder_radius_to_cross_section_extent_ratio
            ):
                return None

        inlier_ratio = float(len(inlier_indices) / len(points))
        if inlier_ratio < self.descriptor.parameters.cylinder_minimum_inlier_ratio:
            return None

        root_mean_square_error = float(
            np.sqrt(np.mean(np.square(absolute_residuals[inlier_indices])))
        )
        score = compute_fit_score(
            inlier_ratio=inlier_ratio,
            root_mean_square_error=root_mean_square_error,
            distance_threshold=self.descriptor.parameters.distance_threshold,
            complexity_penalty=0.02,
        )

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

    def _point_cloud_principal_axes(self, points: np.ndarray) -> List[np.ndarray]:
        """Return principal point-cloud axes sorted by descending variance."""
        centered_points = points - points.mean(axis=0)
        covariance_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        return [
            self._normalized_vector(eigenvectors[:, axis_index])
            for axis_index in sorted_indices
        ]

    def _axis_angle_degrees(
        self, first_axis: np.ndarray, second_axis: np.ndarray
    ) -> float:
        """Return unsigned angle in degrees between two axis directions."""
        normalized_first_axis = self._normalized_vector(first_axis)
        normalized_second_axis = self._normalized_vector(second_axis)
        cosine = float(
            np.clip(
                np.abs(np.dot(normalized_first_axis, normalized_second_axis)),
                0.0,
                1.0,
            )
        )
        return float(np.degrees(np.arccos(cosine)))

    def _normalized_vector(self, vector: np.ndarray) -> np.ndarray:
        """Return normalized vector with safe fallback for near-zero norm."""
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        return np.asarray(vector, dtype=np.float64) / norm

    def _cross_section_max_extent_for_axis(
        self, points: np.ndarray, axis_center: np.ndarray, axis_direction: np.ndarray
    ) -> float:
        """Return largest observed cross-section extent orthogonal to the cylinder axis."""
        if len(points) == 0:
            return 0.0

        first_basis_axis, second_basis_axis = self._orthogonal_axes_for_plane_normal(
            axis_direction
        )
        centered_points = points - axis_center
        first_coordinates = centered_points @ first_basis_axis
        second_coordinates = centered_points @ second_basis_axis
        first_extent = float(first_coordinates.max() - first_coordinates.min())
        second_extent = float(second_coordinates.max() - second_coordinates.min())
        return max(first_extent, second_extent)

    def _orthogonal_axes_for_plane_normal(
        self, plane_normal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return two orthonormal basis axes orthogonal to the given plane normal."""
        normalized_normal = self._normalized_vector(plane_normal)
        helper_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper_axis, normalized_normal))) > 0.9:
            helper_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

        first_basis_axis = np.cross(normalized_normal, helper_axis)
        first_basis_axis = first_basis_axis / max(
            np.linalg.norm(first_basis_axis), 1e-9
        )
        second_basis_axis = np.cross(normalized_normal, first_basis_axis)
        second_basis_axis = second_basis_axis / max(
            np.linalg.norm(second_basis_axis), 1e-9
        )
        return (
            first_basis_axis.astype(np.float64),
            second_basis_axis.astype(np.float64),
        )

    def _stabilize_cuboid_orientation_if_ambiguous(
        self, cuboid_fit: CuboidFit, points: np.ndarray
    ) -> CuboidFit:
        """Stabilize in-plane cuboid orientation for near-square visible faces."""
        if not self.descriptor.parameters.stabilize_ambiguous_cuboid_orientation:
            return cuboid_fit

        extents = np.asarray(cuboid_fit.extents, dtype=np.float64)
        smallest_axis_index = int(np.argmin(extents))
        in_plane_axis_indices = [
            axis_index for axis_index in range(3) if axis_index != smallest_axis_index
        ]
        first_in_plane_axis_index = in_plane_axis_indices[0]
        second_in_plane_axis_index = in_plane_axis_indices[1]

        first_in_plane_extent = float(extents[first_in_plane_axis_index])
        second_in_plane_extent = float(extents[second_in_plane_axis_index])
        in_plane_extent_relative_difference = abs(
            first_in_plane_extent - second_in_plane_extent
        ) / max(max(first_in_plane_extent, second_in_plane_extent), 1e-9)

        if (
            in_plane_extent_relative_difference
            > self.descriptor.parameters.cuboid_ambiguous_in_plane_extent_relative_difference
        ):
            return cuboid_fit

        rotation_matrix = np.asarray(cuboid_fit.rotation_matrix, dtype=np.float64)
        normal_axis = rotation_matrix[:, smallest_axis_index]
        normal_axis = normal_axis / max(np.linalg.norm(normal_axis), 1e-9)

        reference_axes = [
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        ]
        projected_reference_axis: Optional[np.ndarray] = None
        for reference_axis in reference_axes:
            candidate_axis = self._project_axis_onto_plane(reference_axis, normal_axis)
            if np.linalg.norm(candidate_axis) > 1e-6:
                projected_reference_axis = candidate_axis
                break

        if projected_reference_axis is None:
            return cuboid_fit

        canonical_first_axis = projected_reference_axis / max(
            np.linalg.norm(projected_reference_axis), 1e-9
        )
        original_first_axis = rotation_matrix[:, first_in_plane_axis_index]
        original_second_axis = rotation_matrix[:, second_in_plane_axis_index]

        if float(np.dot(canonical_first_axis, original_first_axis)) < 0.0:
            canonical_first_axis *= -1.0

        canonical_second_axis = np.cross(normal_axis, canonical_first_axis)
        canonical_second_axis = canonical_second_axis / max(
            np.linalg.norm(canonical_second_axis), 1e-9
        )

        if float(np.dot(canonical_second_axis, original_second_axis)) < 0.0:
            canonical_first_axis *= -1.0
            canonical_second_axis *= -1.0

        stabilized_rotation_matrix = np.asarray(
            rotation_matrix, dtype=np.float64
        ).copy()
        stabilized_rotation_matrix[:, smallest_axis_index] = normal_axis
        stabilized_rotation_matrix[:, first_in_plane_axis_index] = canonical_first_axis
        stabilized_rotation_matrix[:, second_in_plane_axis_index] = (
            canonical_second_axis
        )

        if np.linalg.det(stabilized_rotation_matrix) < 0.0:
            stabilized_rotation_matrix[:, second_in_plane_axis_index] *= -1.0

        stabilized_fit = self._refit_cuboid_with_fixed_orientation(
            points=points,
            fixed_rotation_matrix=stabilized_rotation_matrix,
            extent_support_indices=np.asarray(
                cuboid_fit.inlier_indices, dtype=np.int64
            ),
        )
        if stabilized_fit is None:
            return cuboid_fit
        return stabilized_fit

    def _project_axis_onto_plane(
        self, axis: np.ndarray, plane_normal: np.ndarray
    ) -> np.ndarray:
        """Project axis onto the plane orthogonal to the given plane normal."""
        return np.asarray(axis, dtype=np.float64) - float(
            np.dot(axis, plane_normal)
        ) * np.asarray(plane_normal, dtype=np.float64)

    def _refit_cuboid_with_fixed_orientation(
        self,
        points: np.ndarray,
        fixed_rotation_matrix: np.ndarray,
        extent_support_indices: np.ndarray,
    ) -> Optional[CuboidFit]:
        """Refit cuboid center and extents while keeping rotation fixed."""
        if len(points) < 8:
            return None

        if len(extent_support_indices) >= 8:
            extent_support_points = points[extent_support_indices]
        else:
            extent_support_points = points

        projected_support_points = extent_support_points @ fixed_rotation_matrix
        minimum_coordinates = projected_support_points.min(axis=0)
        maximum_coordinates = projected_support_points.max(axis=0)
        extents = np.maximum(maximum_coordinates - minimum_coordinates, 1e-6)
        if np.any(extents > self.descriptor.parameters.max_cuboid_extent_meters):
            return None

        center_local_coordinates = 0.5 * (minimum_coordinates + maximum_coordinates)
        center = center_local_coordinates @ fixed_rotation_matrix.T

        surface_distances = point_to_oriented_box_surface_distance(
            points=points,
            center=center.astype(np.float64),
            rotation_matrix=fixed_rotation_matrix.astype(np.float64),
            extents=extents.astype(np.float64),
        )
        inlier_indices = np.where(
            surface_distances <= self.descriptor.parameters.cuboid_distance_threshold
        )[0]
        if len(inlier_indices) < 8:
            return None

        inlier_ratio = float(len(inlier_indices) / len(points))
        if inlier_ratio < self.descriptor.parameters.cuboid_minimum_inlier_ratio:
            return None

        root_mean_square_error = float(
            np.sqrt(np.mean(np.square(surface_distances[inlier_indices])))
        )
        score = compute_fit_score(
            inlier_ratio=inlier_ratio,
            root_mean_square_error=root_mean_square_error,
            distance_threshold=self.descriptor.parameters.cuboid_distance_threshold,
            complexity_penalty=0.02,
        )

        return CuboidFit(
            center=center.astype(np.float64),
            rotation_matrix=fixed_rotation_matrix.astype(np.float64),
            extents=extents.astype(np.float64),
            inlier_indices=inlier_indices.astype(np.int64),
            inlier_ratio=inlier_ratio,
            root_mean_square_error=root_mean_square_error,
            score=score,
        )

    def _map_to_object_indices(
        self,
        object_hypothesis: ObjectHypothesis,
        local_indices: np.ndarray,
    ) -> List[int]:
        """Map local point indices back to object hypothesis index space."""
        if len(object_hypothesis.point_indices) == 0:
            return local_indices.astype(np.int64).tolist()

        if len(object_hypothesis.point_indices) == len(
            object_hypothesis.points.points
        ) and isinstance(object_hypothesis.point_indices[0], (int, np.integer)):
            object_indices = np.asarray(object_hypothesis.point_indices, dtype=np.int64)
            return object_indices[local_indices].astype(np.int64).tolist()

        if len(object_hypothesis.point_indices) == 1:
            nested_indices = np.asarray(
                object_hypothesis.point_indices[0], dtype=np.int64
            ).reshape(-1)
            if len(nested_indices) == len(object_hypothesis.points.points):
                return nested_indices[local_indices].astype(np.int64).tolist()

        return local_indices.astype(np.int64).tolist()

    def _fit_to_annotation(self, fit_result: FittedShape) -> Shape:
        """Convert a fit result into a RoboKudo shape annotation."""
        if isinstance(fit_result, SphereFit):
            return self._sphere_fit_to_annotation(fit_result)
        if isinstance(fit_result, CylinderFit):
            return self._cylinder_fit_to_annotation(fit_result)
        return self._cuboid_fit_to_annotation(fit_result)

    def _sphere_fit_to_annotation(self, fit_result: SphereFit) -> Sphere:
        """Create a sphere annotation from a sphere fit."""
        origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=float(fit_result.center[0]),
            pos_y=float(fit_result.center[1]),
            pos_z=float(fit_result.center[2]),
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
        )
        return Sphere(
            geometry=SemDTSphere(
                origin=origin,
                radius=float(fit_result.radius),
            )
        )

    def _cylinder_fit_to_annotation(self, fit_result: CylinderFit) -> Cylinder:
        """Create a cylinder annotation from a cylinder fit."""
        rotation_matrix = self._rotation_matrix_from_axis(fit_result.axis_direction)
        quaternion = get_quaternion_from_rotation_matrix(rotation_matrix)
        origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=float(fit_result.axis_center[0]),
            pos_y=float(fit_result.axis_center[1]),
            pos_z=float(fit_result.axis_center[2]),
            quat_x=float(quaternion[0]),
            quat_y=float(quaternion[1]),
            quat_z=float(quaternion[2]),
            quat_w=float(quaternion[3]),
        )
        return Cylinder(
            geometry=SemDTCylinder(
                origin=origin,
                width=float(2.0 * fit_result.radius),
                height=float(fit_result.height),
            )
        )

    def _cuboid_fit_to_annotation(self, fit_result: CuboidFit) -> Cuboid:
        """Create a cuboid annotation from a cuboid fit."""
        quaternion = get_quaternion_from_rotation_matrix(fit_result.rotation_matrix)
        origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=float(fit_result.center[0]),
            pos_y=float(fit_result.center[1]),
            pos_z=float(fit_result.center[2]),
            quat_x=float(quaternion[0]),
            quat_y=float(quaternion[1]),
            quat_z=float(quaternion[2]),
            quat_w=float(quaternion[3]),
        )
        return Cuboid(
            geometry=SemDTBox(
                origin=origin,
                scale=Scale(
                    x=float(fit_result.extents[0]),
                    y=float(fit_result.extents[1]),
                    z=float(fit_result.extents[2]),
                ),
            )
        )

    def _rotation_matrix_from_axis(self, axis_direction: np.ndarray) -> np.ndarray:
        """Create an orthonormal rotation matrix with z aligned to the axis."""
        normalized_axis = np.asarray(axis_direction, dtype=np.float64)
        axis_norm = np.linalg.norm(normalized_axis)
        if axis_norm < 1e-9:
            normalized_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            normalized_axis = normalized_axis / axis_norm

        helper_vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(helper_vector, normalized_axis))) > 0.9:
            helper_vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)

        x_axis = np.cross(helper_vector, normalized_axis)
        x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)
        y_axis = np.cross(normalized_axis, x_axis)
        y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)

        rotation_matrix = np.zeros((3, 3), dtype=np.float64)
        rotation_matrix[:, 0] = x_axis
        rotation_matrix[:, 1] = y_axis
        rotation_matrix[:, 2] = normalized_axis
        return rotation_matrix

    def _create_visualization_geometries(
        self,
        object_hypothesis: ObjectHypothesis,
        filtered_cloud: o3d.geometry.PointCloud,
        best_fit: FittedShape,
    ) -> List[Dict[str, Any]]:
        """Create visualization geometries for one fitted object."""
        object_name = object_hypothesis.id if object_hypothesis.id != "" else "object"
        shape_name = type(best_fit).__name__.replace("Fit", "")

        colored_cloud = self._create_inlier_colored_cloud(
            cloud=filtered_cloud,
            inlier_indices=best_fit.inlier_indices,
        )
        fit_geometry = self._fit_to_o3d_geometry(best_fit)
        frame_geometry = self._fit_to_coordinate_frame(best_fit)

        return [
            {
                "name": f"{object_name}_{shape_name}_points",
                "geometry": colored_cloud,
            },
            {
                "name": f"{object_name}_{shape_name}_fit",
                "geometry": fit_geometry,
            },
            {
                "name": f"{object_name}_{shape_name}_frame",
                "geometry": frame_geometry,
            },
        ]

    def _create_inlier_colored_cloud(
        self, cloud: o3d.geometry.PointCloud, inlier_indices: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """Return a copy of the cloud with inliers and outliers color coded."""
        colored_cloud = copy.deepcopy(cloud)
        point_count = len(colored_cloud.points)
        colors = np.tile(
            np.asarray([[0.85, 0.15, 0.15]], dtype=np.float64), (point_count, 1)
        )
        colors[inlier_indices] = np.asarray([0.1, 0.8, 0.1], dtype=np.float64)
        colored_cloud.colors = o3d.utility.Vector3dVector(colors)
        return colored_cloud

    def _fit_to_o3d_geometry(self, fit_result: FittedShape) -> o3d.geometry.Geometry:
        """Convert a fitted primitive into an Open3D geometry."""
        if isinstance(fit_result, SphereFit):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=fit_result.radius)
            sphere.translate(fit_result.center)
            sphere.paint_uniform_color([0.2, 0.4, 1.0])
            return sphere

        if isinstance(fit_result, CylinderFit):
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=fit_result.radius, height=fit_result.height
            )
            rotation_matrix = self._rotation_matrix_from_axis(fit_result.axis_direction)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = fit_result.axis_center
            cylinder.transform(transform)
            cylinder.paint_uniform_color([0.95, 0.6, 0.2])
            return cylinder

        cuboid_fit = fit_result
        oriented_box = o3d.geometry.OrientedBoundingBox(
            center=cuboid_fit.center,
            R=cuboid_fit.rotation_matrix,
            extent=cuboid_fit.extents,
        )
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(oriented_box)
        line_set.paint_uniform_color([0.2, 0.4, 1.0])
        return line_set

    def _fit_to_coordinate_frame(
        self, fit_result: FittedShape
    ) -> o3d.geometry.TriangleMesh:
        """Create a coordinate frame located at the fitted primitive center."""
        if isinstance(fit_result, SphereFit):
            frame_size = max(float(fit_result.radius * 1.5), 0.01)
            center = fit_result.center
            rotation_matrix = np.eye(3, dtype=np.float64)
        elif isinstance(fit_result, CylinderFit):
            frame_size = max(float(fit_result.radius * 1.8), 0.01)
            center = fit_result.axis_center
            rotation_matrix = self._rotation_matrix_from_axis(fit_result.axis_direction)
        else:
            frame_size = max(float(np.max(fit_result.extents) * 0.35), 0.01)
            center = fit_result.center
            rotation_matrix = fit_result.rotation_matrix

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = center
        frame.transform(transform)
        return frame

    def _minimum_inlier_ratio_for_fit(self, fit_result: FittedShape) -> float:
        """Return the configured minimum inlier ratio for one fitted shape type."""
        if isinstance(fit_result, CuboidFit):
            return self.descriptor.parameters.cuboid_minimum_inlier_ratio
        if isinstance(fit_result, CylinderFit):
            return self.descriptor.parameters.cylinder_minimum_inlier_ratio
        return self.descriptor.parameters.minimum_inlier_ratio

    def _log_rejected_candidate(
        self,
        object_hypothesis: ObjectHypothesis,
        shape_name: str,
    ) -> None:
        """Log one rejected shape candidate."""
        if not self.descriptor.parameters.log_candidate_metrics:
            return
        if not self.descriptor.parameters.log_rejection_reasons:
            return
        object_name = object_hypothesis.id if object_hypothesis.id != "" else "object"
        self.rk_logger.info(f"{self.name} rejected {shape_name} for {object_name}")

    def _log_candidate_metrics(
        self, object_hypothesis: ObjectHypothesis, fit_result: FittedShape
    ) -> None:
        """Log shape candidate metrics for debugging."""
        if not self.descriptor.parameters.log_candidate_metrics:
            return
        object_name = object_hypothesis.id if object_hypothesis.id != "" else "object"
        self.rk_logger.info(
            f"{self.name} candidate {object_name}: {self._fit_summary(fit_result)}"
        )

    def _log_selected_candidate(
        self, object_hypothesis: ObjectHypothesis, fit_result: FittedShape
    ) -> None:
        """Log selected candidate metrics for debugging."""
        if not self.descriptor.parameters.log_candidate_metrics:
            return
        object_name = object_hypothesis.id if object_hypothesis.id != "" else "object"
        self.rk_logger.info(
            f"{self.name} selected {object_name}: {self._fit_summary(fit_result)}"
        )

    def _fit_summary(self, fit_result: FittedShape) -> str:
        """Return a compact summary string for one fitted candidate."""
        if isinstance(fit_result, SphereFit):
            return (
                f"Sphere(score={fit_result.score:.3f}, "
                f"inlier_ratio={fit_result.inlier_ratio:.3f}, "
                f"rmse={fit_result.root_mean_square_error:.4f}, "
                f"radius={fit_result.radius:.3f})"
            )
        if isinstance(fit_result, CylinderFit):
            return (
                f"Cylinder(score={fit_result.score:.3f}, "
                f"inlier_ratio={fit_result.inlier_ratio:.3f}, "
                f"rmse={fit_result.root_mean_square_error:.4f}, "
                f"radius={fit_result.radius:.3f}, "
                f"height={fit_result.height:.3f})"
            )
        return (
            f"Cuboid(score={fit_result.score:.3f}, "
            f"inlier_ratio={fit_result.inlier_ratio:.3f}, "
            f"rmse={fit_result.root_mean_square_error:.4f}, "
            f"extents={np.round(fit_result.extents, 3).tolist()})"
        )
