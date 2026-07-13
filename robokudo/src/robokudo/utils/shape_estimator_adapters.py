"""Shape-specific adapters for fitted primitive results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import (
    Box as SemDTBox,
)
from semantic_digital_twin.world_description.geometry import (
    Cylinder as SemDTCylinder,
)
from semantic_digital_twin.world_description.geometry import (
    Scale,
)
from semantic_digital_twin.world_description.geometry import (
    Sphere as SemDTSphere,
)
from typing_extensions import Optional, Union

from robokudo.types.annotation import Cuboid, Cylinder, Shape, Sphere
from robokudo.utils.shape_fitting import (
    CuboidFit,
    CylinderFit,
    FittedShape,
    SphereFit,
    fit_cuboid,
    fit_cylinder,
    fit_sphere,
)
from robokudo.utils.transform import get_quaternion_from_rotation_matrix


@dataclass(frozen=True)
class SphereFitParameters:
    """Parameters for fitting one sphere candidate."""

    distance_threshold: float
    robust_loss: str
    max_radius: float
    max_radius_to_bbox_diagonal_ratio: float
    max_radius_to_observed_extent_ratio: float
    max_center_distance_to_bbox_diagonal_ratio: float
    min_inlier_ratio: float


@dataclass(frozen=True)
class CylinderFitParameters:
    """Parameters for fitting one cylinder candidate."""

    distance_threshold: float
    robust_loss: str
    max_radius: float
    max_height: float
    max_radius_to_bbox_diagonal_ratio: float
    max_radius_to_cross_section_extent_ratio: float
    max_axis_center_distance_to_bbox_diagonal_ratio: float
    min_inlier_ratio: float
    max_initializations: int
    consensus_trials: int
    inlier_polishing_iterations: int


@dataclass(frozen=True)
class CuboidFitParameters:
    """Parameters for fitting one cuboid candidate."""

    distance_threshold: float
    max_extent: float
    min_inlier_ratio: float


ShapeFitParameters = Union[
    SphereFitParameters,
    CylinderFitParameters,
    CuboidFitParameters,
]


@dataclass(frozen=True)
class ShapeFitSummary(ABC):
    """Shared compact metrics for one fitted shape candidate."""

    score: float
    """Fit quality score."""

    inlier_ratio: float
    """Ratio of input points explained by the fitted shape."""

    root_mean_square_error: float
    """Root mean square distance error of inlier points."""

    @property
    @abstractmethod
    def shape_name(self) -> str:
        """Name of the fitted shape."""

    @abstractmethod
    def shape_specific_fields(self) -> tuple[tuple[str, object, str | None], ...]:
        """Return fields that only apply to this fitted shape."""

    def __str__(self) -> str:
        """Return the compact log representation."""
        common_fields = (
            ("score", self.score, ".3f"),
            ("inlier_ratio", self.inlier_ratio, ".3f"),
            ("rmse", self.root_mean_square_error, ".4f"),
        )
        fields = common_fields + self.shape_specific_fields()
        formatted_fields = [
            f"{name}={_format_summary_value(value, format_spec)}"
            for name, value, format_spec in fields
        ]
        return f"{self.shape_name}({', '.join(formatted_fields)})"


@dataclass(frozen=True)
class SphereFitSummary(ShapeFitSummary):
    """Compact metrics for one fitted sphere candidate."""

    radius: float
    """Fitted sphere radius."""

    @property
    def shape_name(self) -> str:
        return "Sphere"

    def shape_specific_fields(self) -> tuple[tuple[str, object, str | None], ...]:
        return (("radius", self.radius, ".3f"),)


@dataclass(frozen=True)
class CylinderFitSummary(ShapeFitSummary):
    """Compact metrics for one fitted cylinder candidate."""

    radius: float
    """Fitted cylinder radius."""

    height: float
    """Fitted cylinder height."""

    @property
    def shape_name(self) -> str:
        return "Cylinder"

    def shape_specific_fields(self) -> tuple[tuple[str, object, str | None], ...]:
        return (
            ("radius", self.radius, ".3f"),
            ("height", self.height, ".3f"),
        )


@dataclass(frozen=True)
class CuboidFitSummary(ShapeFitSummary):
    """Compact metrics for one fitted cuboid candidate."""

    extents: list[float]
    """Fitted cuboid side lengths."""

    @property
    def shape_name(self) -> str:
        return "Cuboid"

    def shape_specific_fields(self) -> tuple[tuple[str, object, str | None], ...]:
        return (("extents", self.extents, None),)


class ShapeFitAdapter(ABC):
    """Shape-specific behavior for one fitted primitive type."""

    shape_name: str

    @abstractmethod
    def matches(self, fit_result: FittedShape) -> bool:
        """Return whether this adapter handles the fit result."""

    @abstractmethod
    def fit(
        self, points: np.ndarray, parameters: ShapeFitParameters
    ) -> Optional[FittedShape]:
        """Fit this primitive using shape-specific fit parameters."""

    @abstractmethod
    def to_annotation(self, fit_result: FittedShape) -> Shape:
        """Convert a fit result into a RoboKudo shape annotation."""

    @abstractmethod
    def to_o3d_geometry(self, fit_result: FittedShape) -> o3d.geometry.Geometry:
        """Convert a fitted primitive into an Open3D geometry."""

    @abstractmethod
    def to_coordinate_frame(self, fit_result: FittedShape) -> o3d.geometry.TriangleMesh:
        """Create a coordinate frame located at the fitted primitive center."""

    @abstractmethod
    def summary(self, fit_result: FittedShape) -> ShapeFitSummary:
        """Return compact metrics for one fitted candidate."""


class SphereFitAdapter(ShapeFitAdapter):
    """Adapter for sphere fit results."""

    shape_name = "Sphere"

    def matches(self, fit_result: FittedShape) -> bool:
        """Return whether this adapter handles the fit result."""
        return isinstance(fit_result, SphereFit)

    def fit(
        self, points: np.ndarray, parameters: ShapeFitParameters
    ) -> Optional[SphereFit]:
        """Fit one sphere candidate."""
        fit_parameters = _as_sphere_fit_parameters(parameters)
        return fit_sphere(
            points=points,
            distance_threshold=fit_parameters.distance_threshold,
            robust_loss=fit_parameters.robust_loss,
            max_radius=fit_parameters.max_radius,
            max_radius_to_bbox_diagonal_ratio=(
                fit_parameters.max_radius_to_bbox_diagonal_ratio
            ),
            max_radius_to_observed_extent_ratio=(
                fit_parameters.max_radius_to_observed_extent_ratio
            ),
            max_center_distance_to_bbox_diagonal_ratio=(
                fit_parameters.max_center_distance_to_bbox_diagonal_ratio
            ),
            min_inlier_ratio=fit_parameters.min_inlier_ratio,
        )

    def to_annotation(self, fit_result: FittedShape) -> Sphere:
        """Create a sphere annotation from a sphere fit."""
        sphere_fit = _as_sphere_fit(fit_result)
        origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=float(sphere_fit.center[0]),
            pos_y=float(sphere_fit.center[1]),
            pos_z=float(sphere_fit.center[2]),
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
        )
        return Sphere(
            geometry=SemDTSphere(
                origin=origin,
                radius=float(sphere_fit.radius),
            )
        )

    def to_o3d_geometry(self, fit_result: FittedShape) -> o3d.geometry.Geometry:
        """Convert a fitted sphere into an Open3D geometry."""
        sphere_fit = _as_sphere_fit(fit_result)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_fit.radius)
        sphere.translate(sphere_fit.center)
        sphere.paint_uniform_color([0.2, 0.4, 1.0])
        return sphere

    def to_coordinate_frame(self, fit_result: FittedShape) -> o3d.geometry.TriangleMesh:
        """Create a coordinate frame at the fitted sphere center."""
        sphere_fit = _as_sphere_fit(fit_result)
        frame_size = max(float(sphere_fit.radius * 1.5), 0.01)
        return _coordinate_frame(
            center=sphere_fit.center,
            rotation_matrix=np.eye(3, dtype=np.float64),
            frame_size=frame_size,
        )

    def summary(self, fit_result: FittedShape) -> SphereFitSummary:
        """Return compact metrics for one sphere fit."""
        sphere_fit = _as_sphere_fit(fit_result)
        return SphereFitSummary(
            score=float(sphere_fit.score),
            inlier_ratio=float(sphere_fit.inlier_ratio),
            root_mean_square_error=float(sphere_fit.root_mean_square_error),
            radius=float(sphere_fit.radius),
        )


class CylinderFitAdapter(ShapeFitAdapter):
    """Adapter for cylinder fit results."""

    shape_name = "Cylinder"

    def matches(self, fit_result: FittedShape) -> bool:
        """Return whether this adapter handles the fit result."""
        return isinstance(fit_result, CylinderFit)

    def fit(
        self, points: np.ndarray, parameters: ShapeFitParameters
    ) -> Optional[CylinderFit]:
        """Fit one cylinder candidate."""
        fit_parameters = _as_cylinder_fit_parameters(parameters)
        return fit_cylinder(
            points=points,
            distance_threshold=fit_parameters.distance_threshold,
            robust_loss=fit_parameters.robust_loss,
            max_radius=fit_parameters.max_radius,
            max_height=fit_parameters.max_height,
            max_radius_to_bbox_diagonal_ratio=(
                fit_parameters.max_radius_to_bbox_diagonal_ratio
            ),
            max_radius_to_cross_section_extent_ratio=(
                fit_parameters.max_radius_to_cross_section_extent_ratio
            ),
            max_axis_center_distance_to_bbox_diagonal_ratio=(
                fit_parameters.max_axis_center_distance_to_bbox_diagonal_ratio
            ),
            min_inlier_ratio=fit_parameters.min_inlier_ratio,
            max_initializations=fit_parameters.max_initializations,
            consensus_trials=fit_parameters.consensus_trials,
            inlier_polishing_iterations=(fit_parameters.inlier_polishing_iterations),
        )

    def to_annotation(self, fit_result: FittedShape) -> Cylinder:
        """Create a cylinder annotation from a cylinder fit."""
        cylinder_fit = _as_cylinder_fit(fit_result)
        rotation_matrix = rotation_matrix_from_axis(cylinder_fit.axis_direction)
        quaternion = get_quaternion_from_rotation_matrix(rotation_matrix)
        origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=float(cylinder_fit.axis_center[0]),
            pos_y=float(cylinder_fit.axis_center[1]),
            pos_z=float(cylinder_fit.axis_center[2]),
            quat_x=float(quaternion[0]),
            quat_y=float(quaternion[1]),
            quat_z=float(quaternion[2]),
            quat_w=float(quaternion[3]),
        )
        return Cylinder(
            geometry=SemDTCylinder(
                origin=origin,
                width=float(2.0 * cylinder_fit.radius),
                height=float(cylinder_fit.height),
            )
        )

    def to_o3d_geometry(self, fit_result: FittedShape) -> o3d.geometry.Geometry:
        """Convert a fitted cylinder into an Open3D geometry."""
        cylinder_fit = _as_cylinder_fit(fit_result)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=cylinder_fit.radius, height=cylinder_fit.height
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_matrix_from_axis(cylinder_fit.axis_direction)
        transform[:3, 3] = cylinder_fit.axis_center
        cylinder.transform(transform)
        cylinder.paint_uniform_color([0.95, 0.6, 0.2])
        return cylinder

    def to_coordinate_frame(self, fit_result: FittedShape) -> o3d.geometry.TriangleMesh:
        """Create a coordinate frame at the fitted cylinder center."""
        cylinder_fit = _as_cylinder_fit(fit_result)
        frame_size = max(float(cylinder_fit.radius * 1.8), 0.01)
        return _coordinate_frame(
            center=cylinder_fit.axis_center,
            rotation_matrix=rotation_matrix_from_axis(cylinder_fit.axis_direction),
            frame_size=frame_size,
        )

    def summary(self, fit_result: FittedShape) -> CylinderFitSummary:
        """Return compact metrics for one cylinder fit."""
        cylinder_fit = _as_cylinder_fit(fit_result)
        return CylinderFitSummary(
            score=float(cylinder_fit.score),
            inlier_ratio=float(cylinder_fit.inlier_ratio),
            root_mean_square_error=float(cylinder_fit.root_mean_square_error),
            radius=float(cylinder_fit.radius),
            height=float(cylinder_fit.height),
        )


class CuboidFitAdapter(ShapeFitAdapter):
    """Adapter for cuboid fit results."""

    shape_name = "Cuboid"

    def matches(self, fit_result: FittedShape) -> bool:
        """Return whether this adapter handles the fit result."""
        return isinstance(fit_result, CuboidFit)

    def fit(
        self, points: np.ndarray, parameters: ShapeFitParameters
    ) -> Optional[CuboidFit]:
        """Fit one cuboid candidate."""
        fit_parameters = _as_cuboid_fit_parameters(parameters)
        return fit_cuboid(
            points=points,
            distance_threshold=fit_parameters.distance_threshold,
            max_extent=fit_parameters.max_extent,
            min_inlier_ratio=fit_parameters.min_inlier_ratio,
        )

    def to_annotation(self, fit_result: FittedShape) -> Cuboid:
        """Create a cuboid annotation from a cuboid fit."""
        cuboid_fit = _as_cuboid_fit(fit_result)
        quaternion = get_quaternion_from_rotation_matrix(cuboid_fit.rotation_matrix)
        origin = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=float(cuboid_fit.center[0]),
            pos_y=float(cuboid_fit.center[1]),
            pos_z=float(cuboid_fit.center[2]),
            quat_x=float(quaternion[0]),
            quat_y=float(quaternion[1]),
            quat_z=float(quaternion[2]),
            quat_w=float(quaternion[3]),
        )
        return Cuboid(
            geometry=SemDTBox(
                origin=origin,
                scale=Scale(
                    x=float(cuboid_fit.extents[0]),
                    y=float(cuboid_fit.extents[1]),
                    z=float(cuboid_fit.extents[2]),
                ),
            )
        )

    def to_o3d_geometry(self, fit_result: FittedShape) -> o3d.geometry.Geometry:
        """Convert a fitted cuboid into an Open3D geometry."""
        cuboid_fit = _as_cuboid_fit(fit_result)
        oriented_box = o3d.geometry.OrientedBoundingBox(
            center=cuboid_fit.center,
            R=cuboid_fit.rotation_matrix,
            extent=cuboid_fit.extents,
        )
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(oriented_box)
        line_set.paint_uniform_color([0.2, 0.4, 1.0])
        return line_set

    def to_coordinate_frame(self, fit_result: FittedShape) -> o3d.geometry.TriangleMesh:
        """Create a coordinate frame at the fitted cuboid center."""
        cuboid_fit = _as_cuboid_fit(fit_result)
        frame_size = max(float(np.max(cuboid_fit.extents) * 0.35), 0.01)
        return _coordinate_frame(
            center=cuboid_fit.center,
            rotation_matrix=cuboid_fit.rotation_matrix,
            frame_size=frame_size,
        )

    def summary(self, fit_result: FittedShape) -> CuboidFitSummary:
        """Return compact metrics for one cuboid fit."""
        cuboid_fit = _as_cuboid_fit(fit_result)
        return CuboidFitSummary(
            score=float(cuboid_fit.score),
            inlier_ratio=float(cuboid_fit.inlier_ratio),
            root_mean_square_error=float(cuboid_fit.root_mean_square_error),
            extents=np.round(cuboid_fit.extents, 3).tolist(),
        )


SPHERE_FIT_ADAPTER = SphereFitAdapter()
CYLINDER_FIT_ADAPTER = CylinderFitAdapter()
CUBOID_FIT_ADAPTER = CuboidFitAdapter()

SHAPE_FIT_ADAPTERS = (
    SPHERE_FIT_ADAPTER,
    CYLINDER_FIT_ADAPTER,
    CUBOID_FIT_ADAPTER,
)


def adapter_for_fit(fit_result: FittedShape) -> ShapeFitAdapter:
    """Return the adapter for a fitted primitive."""
    for adapter in SHAPE_FIT_ADAPTERS:
        if adapter.matches(fit_result):
            return adapter
    raise TypeError(f"Unsupported fitted shape type: {type(fit_result)!r}")


def rotation_matrix_from_axis(axis_direction: np.ndarray) -> np.ndarray:
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


def _format_summary_value(value: object, format_spec: str | None) -> str:
    """Format one summary value for compact log output."""
    if format_spec is None:
        return str(value)
    return f"{value:{format_spec}}"


def _coordinate_frame(
    center: np.ndarray,
    rotation_matrix: np.ndarray,
    frame_size: float,
) -> o3d.geometry.TriangleMesh:
    """Create a transformed Open3D coordinate frame."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = center
    frame.transform(transform)
    return frame


def _as_sphere_fit(fit_result: FittedShape) -> SphereFit:
    if not isinstance(fit_result, SphereFit):
        raise TypeError(f"Expected SphereFit, got {type(fit_result)!r}")
    return fit_result


def _as_cylinder_fit(fit_result: FittedShape) -> CylinderFit:
    if not isinstance(fit_result, CylinderFit):
        raise TypeError(f"Expected CylinderFit, got {type(fit_result)!r}")
    return fit_result


def _as_cuboid_fit(fit_result: FittedShape) -> CuboidFit:
    if not isinstance(fit_result, CuboidFit):
        raise TypeError(f"Expected CuboidFit, got {type(fit_result)!r}")
    return fit_result


def _as_sphere_fit_parameters(
    parameters: ShapeFitParameters,
) -> SphereFitParameters:
    if not isinstance(parameters, SphereFitParameters):
        raise TypeError(f"Expected SphereFitParameters, got {type(parameters)!r}")
    return parameters


def _as_cylinder_fit_parameters(
    parameters: ShapeFitParameters,
) -> CylinderFitParameters:
    if not isinstance(parameters, CylinderFitParameters):
        raise TypeError(f"Expected CylinderFitParameters, got {type(parameters)!r}")
    return parameters


def _as_cuboid_fit_parameters(
    parameters: ShapeFitParameters,
) -> CuboidFitParameters:
    if not isinstance(parameters, CuboidFitParameters):
        raise TypeError(f"Expected CuboidFitParameters, got {type(parameters)!r}")
    return parameters
