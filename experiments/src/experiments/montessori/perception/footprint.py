"""
Measure a rectified outline and decide which Montessori shape it is.

Because the outlines come from a metric top-down rectification (see
:mod:`~experiments.montessori.perception.orthophoto`), every measurement here is already
in metres and every angle is already about the world frame's z-axis.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from typing_extensions import Optional, Self

from experiments.montessori.semantics import MontessoriShapeCategory

# %% measuring an outline

_POLYGON_TOLERANCE_RATIO = 0.02
"""
How far, as a fraction of an outline's own perimeter, a corner may be moved when
simplifying it to a polygon.

Two percent is enough to absorb the stair-stepping a rectified edge picks up while
keeping the corners that tell a triangle from a square.
"""


@dataclass(frozen=True)
class Footprint:
    """
    The shape of one rectified outline, measured in metres about the world frame.
    """

    area: float
    """
    Area enclosed by the outline, in square metres.
    """

    width: float
    """
    Shorter side of the smallest rectangle enclosing the outline, in metres.
    """

    length: float
    """
    Longer side of the smallest rectangle enclosing the outline, in metres.
    """

    fill_ratio: float
    """
    Area divided by the area of that smallest enclosing rectangle.

    This is what separates the shape families, because it depends only on the outline's
    proportions and not on its size: a triangle fills half of its rectangle, a circle
    fills a quarter of pi, and the square, rectangular and slot outlines fill all of it.
    """

    corner_count: int
    """
    Number of corners left after simplifying the outline to a polygon.
    """

    yaw: float
    """
    Rotation of the enclosing rectangle's longer side about the world frame's z-axis, in
    radians, wrapped into ``[-pi/2, pi/2)``.

    Only meaningful up to the outline's own symmetry: a square is unchanged by a quarter
    turn and a circle by any rotation at all.
    """

    @property
    def aspect_ratio(self) -> float:
        """
        Longer side of the enclosing rectangle divided by its shorter side.
        """
        return self.length / self.width

    @classmethod
    def from_contour(cls, contour: np.ndarray, resolution: float) -> Self:
        """
        Measure an OpenCV contour taken from a rectified image.

        :param contour: The contour, in rectified pixels.
        :param resolution: Edge length of one rectified pixel, in metres.
        :return: The measured footprint.
        """
        (_, _), (first_side, second_side), angle_in_degrees = cv2.minAreaRect(contour)
        width, length = sorted(
            (max(first_side, 1.0) * resolution, max(second_side, 1.0) * resolution)
        )
        rectangle_area = first_side * second_side * resolution * resolution
        area = cv2.contourArea(contour) * resolution * resolution
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, _POLYGON_TOLERANCE_RATIO * perimeter, True)
        yaw = math.radians(angle_in_degrees)
        if second_side > first_side:
            yaw += math.pi / 2
        return cls(
            area=area,
            width=width,
            length=length,
            fill_ratio=area / rectangle_area if rectangle_area > 0.0 else 0.0,
            corner_count=len(corners),
            yaw=(yaw + math.pi / 2) % math.pi - math.pi / 2,
        )


# %% deciding what it is


class FootprintClassifier(ABC):
    """
    Decides which Montessori shape a measured outline is.
    """

    @abstractmethod
    def classify(self, footprint: Footprint) -> Optional[MontessoriShapeCategory]:
        """
        Name the shape an outline belongs to.

        :param footprint: The measured outline.
        :return: The category, or None if the outline matches none of them.
        """


@dataclass(frozen=True)
class CrossSectionClassifier(FootprintClassifier):
    """
    Tells the Montessori shapes apart by the proportions of their cross-sections.

    The board's own mesh is classified the same way (see
    :func:`~experiments.montessori.hole_geometry._classify_hole_shape`), but from counts
    of mesh vertices, which a rectified camera outline has no counterpart for; the
    proportions below stand in for them and hold for any outline of the same shape at
    any size.
    """

    triangle_fill_ratio: float = 0.65
    """
    Fill ratio below which an outline is a triangle.

    Sits between a triangle's own half and a circle's quarter of pi.
    """

    circle_fill_ratio: float = 0.87
    """
    Fill ratio below which a non-triangular outline is a circle.

    Sits between a circle's quarter of pi and the one a four-cornered outline fills.
    """

    slot_aspect_ratio: float = 5.0
    """
    Aspect ratio above which a four-cornered outline is the disk's narrow slot rather
    than a rectangle, matching
    :data:`~experiments.montessori.hole_geometry._DISK_ASPECT_RATIO_THRESHOLD`.
    """

    rectangle_aspect_ratio: float = 1.3
    """
    Aspect ratio above which a four-cornered outline is a rectangle rather than a
    square, matching
    :data:`~experiments.montessori.hole_geometry._RECTANGLE_ASPECT_RATIO_THRESHOLD`.
    """

    def classify(self, footprint: Footprint) -> Optional[MontessoriShapeCategory]:
        if footprint.fill_ratio <= 0.0:
            return None
        if footprint.fill_ratio < self.triangle_fill_ratio:
            return MontessoriShapeCategory.TRIANGULAR_PRISM
        if footprint.fill_ratio < self.circle_fill_ratio:
            return MontessoriShapeCategory.CYLINDER
        if footprint.aspect_ratio > self.slot_aspect_ratio:
            return MontessoriShapeCategory.DISK
        if footprint.aspect_ratio > self.rectangle_aspect_ratio:
            return MontessoriShapeCategory.RECTANGULAR_PRISM
        return MontessoriShapeCategory.CUBE
