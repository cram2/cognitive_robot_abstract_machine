"""
Mimic spaces, frames, points, vectors, lines and changes that the krrood_spatial_math
interface tests run against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from krrood.patterns.interface import Interface
from krrood_spatial_math.line import Line, LineSegment
from krrood_spatial_math.point import GenericPoint, StoredPoint
from krrood_spatial_math.vector import GenericVector, StoredVector
from krrood_spatial_math.cartesian_frame_3_dimensional import CartesianFrame3Dimensional
from krrood_spatial_math.coordinate_system import (
    AffineCoordinateChange,
    AffineFrame,
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


# %% points and vectors that store their coordinates


@dataclass(kw_only=True)
class StoredScalablePlaneVector(StoredVector[ScalablePlaneFrame]):
    """
    A vector on a scalable plane that stores its coordinates as given.
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
class StoredScalablePlanePoint(
    StoredPoint[ScalablePlaneFrame, StoredScalablePlaneVector]
):
    """
    A point on a scalable plane that stores its coordinates as given.
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
