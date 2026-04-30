"""Computer vision types for Robokudo.

This module provides types for computer vision operations including:

* 2D and 3D point representations
* Rectangle and region of interest definitions
* 3D bounding box specifications

The types support integration with:
* OpenCV for image processing
* Open3D for point cloud handling
* Transform system for poses
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from typing_extensions import TYPE_CHECKING, Tuple

from robokudo.types.core import Type, Annotation
from robokudo.types.tf import Pose

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class Point2D(Type):
    """2D point representation.

    Represents a point in 2D image coordinates.
    """

    x: int = 0
    """
    X coordinate
    """

    y: int = 0
    """
    Y coordinate
    """


@dataclass
class Points3D(Type):
    """3D point cloud container.

    Wraps an Open3D point cloud for 3D point operations.
    """

    points: o3d.geometry.PointCloud = None
    """
    The actual Open3D point cloud object
    """


@dataclass
class Rect(Type):
    """2D rectangle representation.

    Defines a rectangle by its top-left corner position and dimensions.
    """

    pos: Point2D = field(default_factory=Point2D)
    """
    Top-left corner position
    """

    width: int = 0
    """
    Rectangle width in pixels
    """

    height: int = 0
    """
    Rectangle height in pixels
    """

    def get_corner_points(self) -> Tuple[int, int, int, int]:
        """Get the rectangle as a tuple of (x1, y1, x2, y2)."""
        return self.pos.x, self.pos.y, self.pos.x + self.width, self.pos.y + self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Get the rectangle as a tuple of (x, y, w, h)."""
        return self.pos.x, self.pos.y, self.width, self.height


@dataclass
class ImageROI(Type):
    """Image region of interest.

    Defines a region of interest in an image using:
    * Binary mask for arbitrary shapes
    * Rectangle for bounding region
    """

    mask: npt.NDArray = None
    """
    Binary opencv mask image
    """

    roi: Rect = field(default_factory=Rect)
    """
    Rectangular region of interest
    """


@dataclass
class BoundingBox3D(Type):
    """3D oriented bounding box.

    Represents a 3D box with:
    * Dimensions along each axis
    * 6-DOF pose defining orientation and position
    """

    x_length: float = 0.0
    """
    Box length along x-axis
    """

    y_length: float = 0.0
    """
    Box length along y-axis
    """

    z_length: float = 0.0
    """
    Box length along z-axis
    """

    pose: Pose = field(default_factory=Pose)
    """
    Box pose in 3D space
    """


class TSDFAnnotation(Annotation):
    """A TSDF Volume annotation."""

    volume: o3d.pipelines.integration.ScalableTSDFVolume
    """The Open3D TSDF Volume object."""

    transform: npt.NDArray[np.float64]
    """The transform from the reference frame to the object frame."""

    def get_coordinate_frame(self) -> o3d.geometry.TriangleMesh:
        """Get the coordinate frame of the TSDF volume."""
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        frame.transform(self.transform)
        return frame

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        """Get the mesh representation of the TSDF volume."""
        mesh = self.volume.extract_triangle_mesh()
        mesh.transform(self.transform)
        return mesh

    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        """Get the point cloud representation of the TSDF volume."""
        pcd = self.volume.extract_point_cloud()
        pcd.transform(self.transform)
        return pcd

    def get_voxel_point_cloud(self) -> o3d.geometry.PointCloud:
        """Get the voxel point cloud representation of the TSDF volume."""
        pcd = self.volume.extract_voxel_point_cloud()
        pcd.transform(self.transform)
        return pcd
