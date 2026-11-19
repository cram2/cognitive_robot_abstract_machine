"""
Points of an affine space, given by their coordinates in an affine frame.
"""

from __future__ import annotations

from abc import abstractmethod

from typing_extensions import Generic, Self, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.mixin import Mixin
from krrood_spatial_math.coordinate_system import (
    GenericAffineFrame,
    GenericCartesianFrame,
    CoordinateRepresentation,
    GenericCoordinate,
)
from krrood_spatial_math.vector import GenericVector

# %% points given in an affine frame


class Point(
    CoordinateRepresentation[GenericAffineFrame],
    Generic[GenericAffineFrame, GenericCoordinate, GenericVector],
    Interface,
):
    """
    A point of an affine space, given by its amounts along an affine frame's axes from
    the frame's origin. The difference of two points is a vector, and a point moved by a
    vector is a point; two points never add up to a point.

    .. note::
        Each implementation binds every parameter once, so one kind of frame has one
        point type. The frame's space fixes how many coordinates a point has.
    """

    coordinates: tuple[GenericCoordinate, ...]
    """
    The amounts along the axes of the frame from its origin, one per axis, in the order
    of the axes.
    """

    @abstractmethod
    def __sub__(self, other: Self) -> GenericVector:
        """
        :param other: A point given in the same frame.
        :return: The vector that moves ``other`` onto this point.
        """

    @abstractmethod
    def __add__(self, vector: GenericVector) -> Self:
        """
        :param vector: A vector given in the same frame.
        :return: This point moved by ``vector``.
        """


GenericPoint = TypeVar("GenericPoint", bound=Point)
"""
The kind of point of an affine space.
"""


# %% points whose frame kind measures the distances between them


class MetricPoint(
    Point[GenericCartesianFrame, GenericCoordinate, GenericVector], Mixin
):
    """
    A point given in a Cartesian frame, whose distances to other points the kind of that
    frame measures.

    .. note::
        :meth:`euclidean_distance` gives the metric of
        :class:`~krrood_spatial_math.coordinate_system.CartesianFrame` on the value
        itself, so it reads like the point's own.
    """

    def euclidean_distance(self, other: Self) -> GenericCoordinate:
        """
        :param other: A point given in the same frame.
        :return: The straight-line distance between this point and ``other``.
        """
        frame_kind = self.get_type_of_generic_parameter(parameter=GenericCartesianFrame)
        return frame_kind.distance(first=self, second=other)
