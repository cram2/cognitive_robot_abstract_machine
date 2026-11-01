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
from typing_extensions import TYPE_CHECKING

from robokudo.types.core import Type
from robokudo.types.tf import Pose

if TYPE_CHECKING:
    import numpy as np
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

    @classmethod
    def from_corner_points(cls, x1: float, y1: float, x2: float, y2: float) -> Rect:
        """
        :param x1: Horizontal coordinate of the top-left corner.
        :param y1: Vertical coordinate of the top-left corner.
        :param x2: Horizontal coordinate of the bottom-right corner.
        :param y2: Vertical coordinate of the bottom-right corner.
        :return: The rectangle between the pixels containing the two corners.
        """
        left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)
        return cls(pos=Point2D(x=left, y=top), width=right - left, height=bottom - top)

    def get_corner_points(self) -> tuple[int, int, int, int]:
        """Get the rectangle as a tuple of (x1, y1, x2, y2)."""
        return self.pos.x, self.pos.y, self.pos.x + self.width, self.pos.y + self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Get the rectangle as a tuple of (x, y, w, h)."""
        return self.pos.x, self.pos.y, self.width, self.height


@dataclass(kw_only=True)
class ColorToDepthRatio:
    """
    Size of the depth image relative to the color image of the same camera.
    """

    horizontal: float = 1.0
    """
    Depth image width divided by color image width.
    """

    vertical: float = 1.0
    """
    Depth image height divided by color image height.
    """

    @property
    def is_identity(self) -> bool:
        """
        Whether the depth image has the resolution of the color image.
        """
        return self.horizontal == 1.0 and self.vertical == 1.0

    def depth_pixel(self, x: float, y: float) -> Point2D:
        """
        :param x: Horizontal position in color image pixels.
        :param y: Vertical position in color image pixels.
        :return: The depth image pixel at that position.
        """
        return Point2D(x=int(x * self.horizontal), y=int(y * self.vertical))

    def depth_region(self, region: Rect) -> Rect:
        """
        :param region: A region in color image pixels.
        :return: The depth image pixels covering the same region.
        """
        left, top, right, bottom = region.get_corner_points()
        return Rect.from_corner_points(
            x1=left * self.horizontal,
            y1=top * self.vertical,
            x2=right * self.horizontal,
            y2=bottom * self.vertical,
        )

    def color_pixels(self, depth_pixels: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """
        :param depth_pixels: The (x, y) of depth image pixels, one pixel per row.
        :return: The (x, y) of the color image pixel at each of those pixels, one pixel
            per row.
        """
        return (depth_pixels * (1 / self.horizontal, 1 / self.vertical)).astype(int)


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
