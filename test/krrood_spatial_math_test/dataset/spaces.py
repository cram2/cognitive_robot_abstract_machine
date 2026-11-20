"""
Mimic spaces, frames, points, vectors, lines and changes that the krrood_spatial_math
interface tests run against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from typing_extensions import Self

from krrood.patterns.classproperty import classproperty
from krrood.patterns.interface import Interface
from krrood_spatial_math.line import Line, LineSegment
from krrood_spatial_math.point import GenericPoint, Point
from krrood_spatial_math.vector import GenericVector, Vector
from krrood_spatial_math.cartesian_frame_3_dimensional import CartesianFrame3Dimensional
from krrood_spatial_math.coordinate_system import (
    AffineCoordinateChange,
    AffineFrame,
    FrameAxis,
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

    @classproperty
    def axes(cls) -> tuple[FrameAxis, ...]:
        return (FrameAxis.X, FrameAxis.Y)


# %% a frame of a scalable plane


@dataclass(eq=False, kw_only=True)
class NamedScalablePlaneFrame(ScalablePlaneFrame):
    """
    A frame on a scalable plane that only has a name, on no known plane.
    """

    name: str
    """
    What the frame is called.
    """

    space: ScalablePlane | None = None
    """
    The plane the frame charts.
    """

    id: UUID = field(default_factory=uuid4)
    """
    The identifier that tells the frame apart from every other frame.
    """


# %% points and vectors of a scalable plane


@dataclass(kw_only=True)
class ScalablePlaneVector(Vector[ScalablePlaneFrame]):
    """
    A vector on a scalable plane whose coordinates are its fields named after the axes.
    """

    x: float
    """
    The amount along the first axis.
    """

    y: float
    """
    The amount along the second axis.
    """


@dataclass(kw_only=True)
class ScalablePlanePoint(Point[ScalablePlaneVector]):
    """
    A point on a scalable plane whose coordinates are its fields named after the axes.
    """

    x: float
    """
    The first coordinate.
    """

    y: float
    """
    The second coordinate.
    """


# %% lines that store their point and direction


@dataclass(kw_only=True)
class StoredLine(Line[GenericPoint]):
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
class StoredScalablePlaneLine(StoredLine[ScalablePlanePoint]):
    """
    A line on a scalable plane that stores its point and direction as given.
    """


# %% segments that store their ends


@dataclass(kw_only=True)
class StoredScalablePlaneLineSegment(LineSegment[StoredScalablePlaneLine]):
    """
    A segment on a scalable plane that stores its ends as given.
    """

    start: ScalablePlanePoint
    """
    The point the segment runs from.
    """

    end: ScalablePlanePoint
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


# %% a change that scales each axis and then shifts the origin


@dataclass(kw_only=True)
class AxisScalingChange(ScalingChange):
    """
    An affine change between two frames of one scalable plane: each axis is scaled by
    its own factor, and then the origin is shifted.
    """

    source_coordinate_system: ScalablePlaneFrame | None = None
    """
    The frame the coordinates are given in.
    """

    target_coordinate_system: ScalablePlaneFrame | None = None
    """
    The frame the coordinates are changed into.
    """

    x_scale: float = 1.0
    """
    The factor the first coordinate is multiplied by.
    """

    y_scale: float = 1.0
    """
    The factor the second coordinate is multiplied by.
    """

    x_shift: float = 0.0
    """
    The first coordinate of the source origin in the target frame.
    """

    y_shift: float = 0.0
    """
    The second coordinate of the source origin in the target frame.
    """

    def apply_to_homogeneous_coordinates(
        self, *, homogeneous_coordinates: tuple[float, ...]
    ) -> tuple[float, ...]:
        x, y, weight = homogeneous_coordinates
        return (
            self.x_scale * x + self.x_shift * weight,
            self.y_scale * y + self.y_shift * weight,
            weight,
        )

    def inverse(self) -> Self:
        return AxisScalingChange(
            source_coordinate_system=self.target_coordinate_system,
            target_coordinate_system=self.source_coordinate_system,
            x_scale=1.0 / self.x_scale,
            y_scale=1.0 / self.y_scale,
            x_shift=-self.x_shift / self.x_scale,
            y_shift=-self.y_shift / self.y_scale,
        )

    def __matmul__(self, other: Self) -> Self:
        return AxisScalingChange(
            source_coordinate_system=other.source_coordinate_system,
            target_coordinate_system=self.target_coordinate_system,
            x_scale=self.x_scale * other.x_scale,
            y_scale=self.y_scale * other.y_scale,
            x_shift=self.x_scale * other.x_shift + self.x_shift,
            y_shift=self.y_scale * other.y_shift + self.y_shift,
        )
