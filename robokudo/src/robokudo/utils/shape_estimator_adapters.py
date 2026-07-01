"""Shape-specific adapters for fitted primitive results."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
from typing_extensions import Any

from robokudo.types.annotation import Cuboid, Cylinder, Shape, Sphere
from robokudo.utils.shape_fitting import CuboidFit, CylinderFit, FittedShape, SphereFit
from robokudo.utils.transform import get_quaternion_from_rotation_matrix


class ShapeFitAdapter(ABC):
    """Shape-specific behavior for one fitted primitive type."""

    shape_name: str

    @abstractmethod
    def matches(self, fit_result: FittedShape) -> bool:
        """Return whether this adapter handles the fit result."""

    @abstractmethod
    def minimum_inlier_ratio(self, parameters: Any) -> float:
        """Return configured minimum inlier ratio for this shape."""

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
    def summary(self, fit_result: FittedShape) -> str:
        """Return a compact summary string for one fitted candidate."""


class SphereFitAdapter(ShapeFitAdapter):
    """Adapter for sphere fit results."""

    shape_name = "Sphere"

    def matches(self, fit_result: FittedShape) -> bool:
        """Return whether this adapter handles the fit result."""
        return isinstance(fit_result, SphereFit)

    def minimum_inlier_ratio(self, parameters: Any) -> float:
        """Return configured minimum inlier ratio for spheres."""
        return parameters.minimum_inlier_ratio

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

    def summary(self, fit_result: FittedShape) -> str:
        """Return a compact summary string for one sphere fit."""
        sphere_fit = _as_sphere_fit(fit_result)
        return (
            f"Sphere(score={sphere_fit.score:.3f}, "
            f"inlier_ratio={sphere_fit.inlier_ratio:.3f}, "
            f"rmse={sphere_fit.root_mean_square_error:.4f}, "
            f"radius={sphere_fit.radius:.3f})"
        )


class CylinderFitAdapter(ShapeFitAdapter):
    """Adapter for cylinder fit results."""

    shape_name = "Cylinder"

    def matches(self, fit_result: FittedShape) -> bool:
        """Return whether this adapter handles the fit result."""
        return isinstance(fit_result, CylinderFit)

    def minimum_inlier_ratio(self, parameters: Any) -> float:
        """Return configured minimum inlier ratio for cylinders."""
        return parameters.cylinder_minimum_inlier_ratio

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

    def summary(self, fit_result: FittedShape) -> str:
        """Return a compact summary string for one cylinder fit."""
        cylinder_fit = _as_cylinder_fit(fit_result)
        return (
            f"Cylinder(score={cylinder_fit.score:.3f}, "
            f"inlier_ratio={cylinder_fit.inlier_ratio:.3f}, "
            f"rmse={cylinder_fit.root_mean_square_error:.4f}, "
            f"radius={cylinder_fit.radius:.3f}, "
            f"height={cylinder_fit.height:.3f})"
        )


class CuboidFitAdapter(ShapeFitAdapter):
    """Adapter for cuboid fit results."""

    shape_name = "Cuboid"

    def matches(self, fit_result: FittedShape) -> bool:
        """Return whether this adapter handles the fit result."""
        return isinstance(fit_result, CuboidFit)

    def minimum_inlier_ratio(self, parameters: Any) -> float:
        """Return configured minimum inlier ratio for cuboids."""
        return parameters.cuboid_minimum_inlier_ratio

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

    def summary(self, fit_result: FittedShape) -> str:
        """Return a compact summary string for one cuboid fit."""
        cuboid_fit = _as_cuboid_fit(fit_result)
        return (
            f"Cuboid(score={cuboid_fit.score:.3f}, "
            f"inlier_ratio={cuboid_fit.inlier_ratio:.3f}, "
            f"rmse={cuboid_fit.root_mean_square_error:.4f}, "
            f"extents={np.round(cuboid_fit.extents, 3).tolist()})"
        )


SHAPE_FIT_ADAPTERS = (
    SphereFitAdapter(),
    CylinderFitAdapter(),
    CuboidFitAdapter(),
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
