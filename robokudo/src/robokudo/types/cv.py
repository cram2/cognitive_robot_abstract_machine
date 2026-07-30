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

import cv2
import open3d as o3d
from typing_extensions import TYPE_CHECKING, Tuple

from robokudo.types.core import Type
from robokudo.types.tf import Pose

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class Points3D(Type):
    """3D point cloud container.

    Wraps an Open3D point cloud for 3D point operations.
    """

    points: o3d.geometry.PointCloud = None
    """
    The actual Open3D point cloud object
    """


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class KeyPoint(Point2D):
    """A key point in an image."""

    size: float
    """Diameter of the keypoint."""

    angle: float
    """Orientation of the keypoint."""

    response: float
    """Strength of the keypoint."""

    octave: int
    """Pyramid octave the keypoint was detected in."""

    class_id: int
    """Object class ID."""

    @classmethod
    def from_cv(cls, keypoint: cv2.KeyPoint) -> KeyPoint:
        """Create a keypoint from an OpenCV keypoint.

        :return: The created keypoint.
        """
        return cls(
            x=keypoint.pt[0],
            y=keypoint.pt[1],
            size=keypoint.size,
            angle=keypoint.angle,
            response=keypoint.response,
            octave=keypoint.octave,
            class_id=keypoint.class_id,
        )

    def to_cv(self) -> cv2.KeyPoint:
        """Convert the RoboKudo keypoint to an OpenCV keypoint.

        :return: The cv2 equivalent of the keypoint.
        """
        return cv2.KeyPoint(
            x=self.x,
            y=self.y,
            size=self.size,
            angle=self.angle,
            response=self.response,
            octave=self.octave,
            class_id=self.class_id,
        )
