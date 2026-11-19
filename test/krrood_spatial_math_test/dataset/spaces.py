"""
Mimic spaces, frames, points, vectors, lines and changes that the krrood_spatial_math
interface tests run against.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Self

from krrood.patterns.interface import Interface
from krrood_spatial_math.line import Line, LineSegment
from krrood_spatial_math.point import Point, GenericPoint
from krrood_spatial_math.vector import Vector, GenericVector
from krrood_spatial_math.cartesian_frame_3_dimensional import CartesianFrame3Dimensional
from krrood_spatial_math.coordinate_system import (
    AffineCoordinateChange,
    AffineFrame,
    GenericAffineFrame,
    GenericCoordinate,
    RigidCoordinateChange,
)
from krrood_spatial_math.space import Space

# %% a plane whose frames differ by scaling each axis


class ScalablePlane(Space, Interface):
    """
    A plane whose frames differ by scaling and shifting each axis, so its coordinate
    changes are affine but not rigid.
    """


class ScalablePlaneFrame(AffineFrame[ScalablePlane, str], Interface):
    """
    An affine frame on a :class:`ScalablePlane`, called by a plain name.
    """


# %% points and vectors that store their coordinates


@dataclass(kw_only=True)
class StoredVector(Vector[GenericAffineFrame, GenericCoordinate]):
    """
    A vector that stores its two coordinates and frame as given.
    """

    x: GenericCoordinate
    """
    The amount along the first axis.
    """

    y: GenericCoordinate
    """
    The amount along the second axis.
    """

    coordinate_system: GenericAffineFrame | None
    """
    The frame the coordinates are given in.
    """

    @property
    def coordinates(self) -> tuple[GenericCoordinate, GenericCoordinate]:
        return self.x, self.y

    def __add__(self, other: Self) -> Self:
        return type(self)(
            x=self.x + other.x,
            y=self.y + other.y,
            coordinate_system=self.coordinate_system,
        )

    def __sub__(self, other: Self) -> Self:
        return type(self)(
            x=self.x - other.x,
            y=self.y - other.y,
            coordinate_system=self.coordinate_system,
        )

    def __mul__(self, factor: GenericCoordinate) -> Self:
        return type(self)(
            x=self.x * factor,
            y=self.y * factor,
            coordinate_system=self.coordinate_system,
        )


@dataclass(kw_only=True)
class StoredScalablePlaneVector(StoredVector[ScalablePlaneFrame, float]):
    """
    A vector on a scalable plane that stores its coordinates as given.
    """


@dataclass(kw_only=True)
class StoredPoint(Point[GenericAffineFrame, GenericCoordinate, GenericVector]):
    """
    A point that stores its two coordinates and frame as given.
    """

    x: GenericCoordinate
    """
    The first coordinate.
    """

    y: GenericCoordinate
    """
    The second coordinate.
    """

    coordinate_system: GenericAffineFrame | None
    """
    The frame the coordinates are given in.
    """

    @property
    def coordinates(self) -> tuple[GenericCoordinate, GenericCoordinate]:
        return self.x, self.y

    def __sub__(self, other: Self) -> GenericVector:
        vector_type = self.get_type_of_generic_parameter(parameter=GenericVector)
        return vector_type(
            x=self.x - other.x,
            y=self.y - other.y,
            coordinate_system=self.coordinate_system,
        )

    def __add__(self, vector: GenericVector) -> Self:
        return type(self)(
            x=self.x + vector.x,
            y=self.y + vector.y,
            coordinate_system=self.coordinate_system,
        )


@dataclass(kw_only=True)
class StoredScalablePlanePoint(
    StoredPoint[ScalablePlaneFrame, float, StoredScalablePlaneVector]
):
    """
    A point on a scalable plane that stores its coordinates as given.
    """


# %% lines that store their point and direction


@dataclass(kw_only=True)
class StoredLine(Line[GenericPoint, GenericVector]):
    """
    A line that stores its point and direction as given.
    """

    point: GenericPoint
    """
    The point the line passes through.
    """

    direction: GenericVector
    """
    The vector the line runs along.
    """


@dataclass(kw_only=True)
class StoredScalablePlaneLine(
    StoredLine[StoredScalablePlanePoint, StoredScalablePlaneVector]
):
    """
    A line on a scalable plane that stores its point and direction as given.
    """


# %% segments that store their ends


@dataclass(kw_only=True)
class StoredScalablePlaneLineSegment(
    LineSegment[StoredScalablePlanePoint, StoredScalablePlaneLine]
):
    """
    A segment on a scalable plane that stores its ends as given.
    """

    start: StoredScalablePlanePoint
    """
    The point the segment runs from.
    """

    end: StoredScalablePlanePoint
    """
    The point the segment runs to.
    """


# %% coordinate changes of one kind each


@dataclass(kw_only=True)
class ScalingChange(AffineCoordinateChange[ScalablePlaneFrame]):
    """
    An affine change between two frames of one scalable plane.
    """


@dataclass(kw_only=True)
class CartesianFrame3DimensionalChange(
    RigidCoordinateChange[CartesianFrame3Dimensional]
):
    """
    A rigid change between two Cartesian frames, such as from a gripper to the table.
    """
