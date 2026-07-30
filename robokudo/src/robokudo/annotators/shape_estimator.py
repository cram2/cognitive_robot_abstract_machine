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
from typing_extensions import Any, Dict, List, Optional, Tuple

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.types.annotation import Shape
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.shape_estimator_adapters import (
    CUBOID_FIT_ADAPTER,
    CYLINDER_FIT_ADAPTER,
    SPHERE_FIT_ADAPTER,
    CuboidFitParameters,
    CylinderFitParameters,
    ShapeFitAdapter,
    ShapeFitParameters,
    SphereFitParameters,
    adapter_for_fit,
)
from robokudo.utils.shape_fitting import (
    CuboidFit,
    CylinderFit,
    CylinderFitConstraints,
    FittedShape,
    refit_cuboid_with_fixed_orientation,
    refit_cylinder_with_fixed_axis,
    select_best_shape,
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
        descriptor: ShapeEstimatorAnnotator.Descriptor | None = None,
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
        if not self._valid_point_cloud(point_cloud):
            return None

        filtered_cloud, retained_point_indices = self._prepare_point_cloud(point_cloud)
        if len(filtered_cloud.points) < self.descriptor.parameters.minimum_point_count:
            return None

        points = np.asarray(filtered_cloud.points, dtype=np.float64)
        candidates = self._collect_shape_candidates(
            object_hypothesis=object_hypothesis,
            points=points,
        )

        best_fit = self._select_best_shape_candidate(candidates)
        if best_fit is None:
            return None
        best_fit_adapter = adapter_for_fit(best_fit)
        self._log_selected_candidate(object_hypothesis, best_fit)

        inlier_indices_in_filtered_cloud = retained_point_indices[
            best_fit.inlier_indices
        ]
        inlier_indices_in_original_object_cloud = self._map_to_object_indices(
            object_hypothesis=object_hypothesis,
            local_indices=inlier_indices_in_filtered_cloud,
        )

        shape_annotation = best_fit_adapter.to_annotation(best_fit)
        shape_annotation.source = self.name
        shape_annotation.inliers = inlier_indices_in_original_object_cloud
        visual_geometries = self._create_visualization_geometries(
            object_hypothesis=object_hypothesis,
            filtered_cloud=filtered_cloud,
            best_fit=best_fit,
        )
        return shape_annotation, visual_geometries

    def _valid_point_cloud(
        self, point_cloud: Optional[o3d.geometry.PointCloud]
    ) -> bool:
        """Return whether a point cloud has enough points for shape estimation."""
        if point_cloud is None:
            return False
        if not isinstance(point_cloud, o3d.geometry.PointCloud):
            return False
        return len(point_cloud.points) >= self.descriptor.parameters.minimum_point_count

    def _collect_shape_candidates(
        self,
        object_hypothesis: ObjectHypothesis,
        points: np.ndarray,
    ) -> List[FittedShape]:
        """Fit all enabled primitives and return accepted candidates."""
        candidates: List[FittedShape] = []

        for adapter, fit_parameters in self._enabled_shape_fit_requests():
            self._record_shape_candidate(
                candidates=candidates,
                object_hypothesis=object_hypothesis,
                shape_name=adapter.shape_name,
                fit_result=self._postprocess_shape_candidate(
                    fit_result=adapter.fit(points, fit_parameters),
                    points=points,
                ),
            )

        return candidates

    def _enabled_shape_fit_requests(
        self,
    ) -> List[Tuple[ShapeFitAdapter, ShapeFitParameters]]:
        """Return enabled shape fitters with explicit fit parameters."""
        parameters = self.descriptor.parameters
        fit_requests: List[Tuple[ShapeFitAdapter, ShapeFitParameters]] = []

        if parameters.fit_sphere:
            fit_requests.append(
                (
                    SPHERE_FIT_ADAPTER,
                    SphereFitParameters(
                        distance_threshold=parameters.distance_threshold,
                        robust_loss=parameters.robust_loss,
                        max_radius=parameters.max_sphere_radius_meters,
                        max_radius_to_bbox_diagonal_ratio=(
                            parameters.max_sphere_radius_to_bbox_diagonal_ratio
                        ),
                        max_radius_to_observed_extent_ratio=(
                            parameters.max_sphere_radius_to_observed_extent_ratio
                        ),
                        max_center_distance_to_bbox_diagonal_ratio=(
                            parameters.max_sphere_center_distance_to_bbox_diagonal_ratio
                        ),
                        min_inlier_ratio=parameters.minimum_inlier_ratio,
                    ),
                )
            )

        if parameters.fit_cylinder:
            fit_requests.append(
                (
                    CYLINDER_FIT_ADAPTER,
                    CylinderFitParameters(
                        distance_threshold=parameters.distance_threshold,
                        robust_loss=parameters.robust_loss,
                        max_radius=parameters.max_cylinder_radius_meters,
                        max_height=parameters.max_cylinder_height_meters,
                        max_radius_to_bbox_diagonal_ratio=(
                            parameters.max_cylinder_radius_to_bbox_diagonal_ratio
                        ),
                        max_radius_to_cross_section_extent_ratio=(
                            parameters.max_cylinder_radius_to_cross_section_extent_ratio
                        ),
                        max_axis_center_distance_to_bbox_diagonal_ratio=(
                            parameters.max_cylinder_center_distance_to_bbox_diagonal_ratio
                        ),
                        min_inlier_ratio=parameters.cylinder_minimum_inlier_ratio,
                        max_initializations=parameters.cylinder_max_initializations,
                        consensus_trials=parameters.cylinder_consensus_trials,
                        inlier_polishing_iterations=(
                            parameters.cylinder_inlier_polishing_iterations
                        ),
                    ),
                )
            )

        if parameters.fit_cuboid:
            fit_requests.append(
                (
                    CUBOID_FIT_ADAPTER,
                    CuboidFitParameters(
                        distance_threshold=parameters.cuboid_distance_threshold,
                        max_extent=parameters.max_cuboid_extent_meters,
                        min_inlier_ratio=parameters.cuboid_minimum_inlier_ratio,
                    ),
                )
            )

        return fit_requests

    def _record_shape_candidate(
        self,
        candidates: List[FittedShape],
        object_hypothesis: ObjectHypothesis,
        shape_name: str,
        fit_result: Optional[FittedShape],
    ) -> None:
        """Append accepted candidate or log its rejection."""
        if fit_result is None:
            self._log_rejected_candidate(
                object_hypothesis=object_hypothesis,
                shape_name=shape_name,
            )
            return

        candidates.append(fit_result)
        self._log_candidate_metrics(object_hypothesis, fit_result)

    def _postprocess_shape_candidate(
        self, fit_result: Optional[FittedShape], points: np.ndarray
    ) -> Optional[FittedShape]:
        """Apply annotator-level stabilization policies to fitted shape candidates."""
        if isinstance(fit_result, CylinderFit):
            return self._stabilize_cylinder_axis_if_tilted(
                cylinder_fit=fit_result,
                points=points,
            )
        if isinstance(fit_result, CuboidFit):
            return self._stabilize_cuboid_orientation_if_ambiguous(
                cuboid_fit=fit_result,
                points=points,
            )
        return fit_result

    def _cylinder_fit_constraints(self) -> CylinderFitConstraints:
        """Return cylinder constraints from descriptor parameters."""
        return CylinderFitConstraints(
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
        )

    def _select_best_shape_candidate(
        self, candidates: List[FittedShape]
    ) -> Optional[FittedShape]:
        """Select the best shape candidate using descriptor preferences."""
        return select_best_shape(
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

        candidate_fits = self._refit_cylinder_axis_candidates(
            cylinder_fit=cylinder_fit,
            points=points,
        )
        selected_candidate = self._select_cylinder_axis_candidate(candidate_fits)
        if selected_candidate.source == "original":
            return cylinder_fit

        if not self._should_accept_stabilized_cylinder_axis(
            original_fit=cylinder_fit,
            selected_candidate=selected_candidate,
        ):
            return cylinder_fit

        self._log_stabilized_cylinder_axis(
            original_fit=cylinder_fit,
            selected_candidate=selected_candidate,
        )
        return selected_candidate.fit

    def _refit_cylinder_axis_candidates(
        self, cylinder_fit: CylinderFit, points: np.ndarray
    ) -> List[CylinderAxisCandidate]:
        """Refit configured cylinder axis candidates and include the original fit."""
        candidate_fits: List[CylinderAxisCandidate] = [
            CylinderAxisCandidate(
                source="original",
                axis_direction=self._normalized_vector(cylinder_fit.axis_direction),
                fit=cylinder_fit,
            )
        ]

        for source, axis_direction in self._collect_cylinder_axis_candidates(points):
            aligned_axis_direction = self._align_axis_with_reference(
                axis_direction=axis_direction,
                reference_axis_direction=cylinder_fit.axis_direction,
            )
            if (
                self._axis_angle_degrees(
                    cylinder_fit.axis_direction, aligned_axis_direction
                )
                < 1.0
            ):
                continue

            refitted_cylinder = refit_cylinder_with_fixed_axis(
                points=points,
                fixed_axis_direction=aligned_axis_direction,
                constraints=self._cylinder_fit_constraints(),
                min_inlier_ratio=(
                    self.descriptor.parameters.cylinder_minimum_inlier_ratio
                ),
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

        return candidate_fits

    def _align_axis_with_reference(
        self, axis_direction: np.ndarray, reference_axis_direction: np.ndarray
    ) -> np.ndarray:
        """Return an axis direction flipped to agree with a reference direction."""
        aligned_axis_direction = np.asarray(axis_direction, dtype=np.float64).copy()
        if float(np.dot(aligned_axis_direction, reference_axis_direction)) < 0.0:
            aligned_axis_direction *= -1.0
        return aligned_axis_direction

    def _should_accept_stabilized_cylinder_axis(
        self,
        original_fit: CylinderFit,
        selected_candidate: CylinderAxisCandidate,
    ) -> bool:
        """Return whether a stabilized cylinder candidate should replace the original."""
        axis_deviation_degrees = self._axis_angle_degrees(
            original_fit.axis_direction, selected_candidate.axis_direction
        )
        if (
            axis_deviation_degrees
            <= self.descriptor.parameters.cylinder_axis_stabilization_trigger_degrees
            and selected_candidate.fit.score <= original_fit.score
        ):
            return False

        minimum_accepted_score = (
            original_fit.score
            - self.descriptor.parameters.cylinder_axis_stabilization_max_score_drop
        )
        if selected_candidate.fit.score < minimum_accepted_score:
            return False

        return True

    def _log_stabilized_cylinder_axis(
        self,
        original_fit: CylinderFit,
        selected_candidate: CylinderAxisCandidate,
    ) -> None:
        """Log an accepted cylinder-axis stabilization."""
        if not self.descriptor.parameters.log_candidate_metrics:
            return
        axis_deviation_degrees = self._axis_angle_degrees(
            original_fit.axis_direction, selected_candidate.axis_direction
        )
        self.rk_logger.info(
            f"{self.name} stabilized cylinder axis: "
            f"source={selected_candidate.source}, "
            f"deviation={axis_deviation_degrees:.2f}deg, "
            f"score_before={original_fit.score:.3f}, "
            f"score_after={selected_candidate.fit.score:.3f}"
        )

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

    def _stabilize_cuboid_orientation_if_ambiguous(
        self, cuboid_fit: CuboidFit, points: np.ndarray
    ) -> CuboidFit:
        """Stabilize in-plane cuboid orientation for near-square visible faces."""
        if not self.descriptor.parameters.stabilize_ambiguous_cuboid_orientation:
            return cuboid_fit

        extents = np.asarray(cuboid_fit.extents, dtype=np.float64)
        (
            smallest_axis_index,
            first_in_plane_axis_index,
            second_in_plane_axis_index,
        ) = self._cuboid_orientation_axis_indices(extents)

        if not self._cuboid_in_plane_orientation_is_ambiguous(
            extents=extents,
            first_in_plane_axis_index=first_in_plane_axis_index,
            second_in_plane_axis_index=second_in_plane_axis_index,
        ):
            return cuboid_fit

        rotation_matrix = np.asarray(cuboid_fit.rotation_matrix, dtype=np.float64)
        stabilized_rotation_matrix = self._stabilized_cuboid_rotation_matrix(
            rotation_matrix=rotation_matrix,
            smallest_axis_index=smallest_axis_index,
            first_in_plane_axis_index=first_in_plane_axis_index,
            second_in_plane_axis_index=second_in_plane_axis_index,
        )
        if stabilized_rotation_matrix is None:
            return cuboid_fit

        return self._refit_cuboid_or_keep_original(
            cuboid_fit=cuboid_fit,
            points=points,
            stabilized_rotation_matrix=stabilized_rotation_matrix,
        )

    def _cuboid_orientation_axis_indices(
        self, extents: np.ndarray
    ) -> Tuple[int, int, int]:
        """Return normal-axis index followed by the two in-plane axis indices."""
        smallest_axis_index = int(np.argmin(extents))
        in_plane_axis_indices = [
            axis_index for axis_index in range(3) if axis_index != smallest_axis_index
        ]
        return (
            smallest_axis_index,
            in_plane_axis_indices[0],
            in_plane_axis_indices[1],
        )

    def _cuboid_in_plane_orientation_is_ambiguous(
        self,
        extents: np.ndarray,
        first_in_plane_axis_index: int,
        second_in_plane_axis_index: int,
    ) -> bool:
        """Return whether in-plane extents are close enough to stabilize."""
        first_in_plane_extent = float(extents[first_in_plane_axis_index])
        second_in_plane_extent = float(extents[second_in_plane_axis_index])
        in_plane_extent_relative_difference = abs(
            first_in_plane_extent - second_in_plane_extent
        ) / max(max(first_in_plane_extent, second_in_plane_extent), 1e-9)
        return (
            in_plane_extent_relative_difference
            <= self.descriptor.parameters.cuboid_ambiguous_in_plane_extent_relative_difference
        )

    def _stabilized_cuboid_rotation_matrix(
        self,
        rotation_matrix: np.ndarray,
        smallest_axis_index: int,
        first_in_plane_axis_index: int,
        second_in_plane_axis_index: int,
    ) -> Optional[np.ndarray]:
        """Return a cuboid rotation matrix with canonical in-plane axes."""
        normal_axis = rotation_matrix[:, smallest_axis_index]
        normal_axis = normal_axis / max(np.linalg.norm(normal_axis), 1e-9)
        projected_reference_axis = self._first_projected_reference_axis(normal_axis)
        if projected_reference_axis is None:
            return None

        canonical_first_axis, canonical_second_axis = self._canonical_cuboid_plane_axes(
            projected_reference_axis=projected_reference_axis,
            normal_axis=normal_axis,
            original_first_axis=rotation_matrix[:, first_in_plane_axis_index],
            original_second_axis=rotation_matrix[:, second_in_plane_axis_index],
        )

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

        return stabilized_rotation_matrix

    def _first_projected_reference_axis(
        self, normal_axis: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return the first global reference axis that projects onto the cuboid plane."""
        reference_axes = [
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        ]
        for reference_axis in reference_axes:
            candidate_axis = self._project_axis_onto_plane(reference_axis, normal_axis)
            if np.linalg.norm(candidate_axis) > 1e-6:
                return candidate_axis
        return None

    def _canonical_cuboid_plane_axes(
        self,
        projected_reference_axis: np.ndarray,
        normal_axis: np.ndarray,
        original_first_axis: np.ndarray,
        original_second_axis: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return canonical in-plane axes aligned with the original axis signs."""
        canonical_first_axis = projected_reference_axis / max(
            np.linalg.norm(projected_reference_axis), 1e-9
        )
        if float(np.dot(canonical_first_axis, original_first_axis)) < 0.0:
            canonical_first_axis *= -1.0

        canonical_second_axis = np.cross(normal_axis, canonical_first_axis)
        canonical_second_axis = canonical_second_axis / max(
            np.linalg.norm(canonical_second_axis), 1e-9
        )

        if float(np.dot(canonical_second_axis, original_second_axis)) < 0.0:
            canonical_first_axis *= -1.0
            canonical_second_axis *= -1.0

        return canonical_first_axis, canonical_second_axis

    def _refit_cuboid_or_keep_original(
        self,
        cuboid_fit: CuboidFit,
        points: np.ndarray,
        stabilized_rotation_matrix: np.ndarray,
    ) -> CuboidFit:
        """Refit a stabilized cuboid and keep the original fit on failure."""
        stabilized_fit = refit_cuboid_with_fixed_orientation(
            points=points,
            fixed_rotation_matrix=stabilized_rotation_matrix,
            extent_support_indices=np.asarray(
                cuboid_fit.inlier_indices, dtype=np.int64
            ),
            distance_threshold=self.descriptor.parameters.cuboid_distance_threshold,
            max_extent=self.descriptor.parameters.max_cuboid_extent_meters,
            min_inlier_ratio=self.descriptor.parameters.cuboid_minimum_inlier_ratio,
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

    def _create_visualization_geometries(
        self,
        object_hypothesis: ObjectHypothesis,
        filtered_cloud: o3d.geometry.PointCloud,
        best_fit: FittedShape,
    ) -> List[Dict[str, Any]]:
        """Create visualization geometries for one fitted object."""
        object_name = object_hypothesis.id if object_hypothesis.id != "" else "object"
        fit_adapter = adapter_for_fit(best_fit)
        shape_name = fit_adapter.shape_name

        colored_cloud = self._create_inlier_colored_cloud(
            cloud=filtered_cloud,
            inlier_indices=best_fit.inlier_indices,
        )
        fit_geometry = fit_adapter.to_o3d_geometry(best_fit)
        frame_geometry = fit_adapter.to_coordinate_frame(best_fit)

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
            f"{self.name} candidate {object_name}: "
            f"{adapter_for_fit(fit_result).summary(fit_result)}"
        )

    def _log_selected_candidate(
        self, object_hypothesis: ObjectHypothesis, fit_result: FittedShape
    ) -> None:
        """Log selected candidate metrics for debugging."""
        if not self.descriptor.parameters.log_candidate_metrics:
            return
        object_name = object_hypothesis.id if object_hypothesis.id != "" else "object"
        self.rk_logger.info(
            f"{self.name} selected {object_name}: "
            f"{adapter_for_fit(fit_result).summary(fit_result)}"
        )
