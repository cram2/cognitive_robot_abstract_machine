from __future__ import annotations

from abc import abstractmethod

from typing_extensions import TYPE_CHECKING, Generic, Self, TypeVar

from krrood.patterns.identifier_and_name import HasID, HasName, NameT
from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_spatial_math.space import EuclideanSpace, Space

if TYPE_CHECKING:
    from krrood_spatial_math.point import Point
    from krrood_spatial_math.vector import Vector

# %% what coordinates are given in

GenericSpace = TypeVar("GenericSpace", bound=Space)
"""
The kind of space a coordinate system charts.
"""


class CoordinateSystem(
    HasName[NameT], Generic[GenericSpace, NameT], SubClassSafeGeneric, HasID, Interface
):
    """
    What the coordinates of a point are given in: an origin, axes and a unit on one
    space.

    Coordinates given in one coordinate system mean nothing in another, so values of two
    coordinate systems are combined only after changing one into the other.
    """

    space: GenericSpace | None
    """
    The space this coordinate system charts; ``None`` where it is not known.
    """


# %% frames: coordinate systems with straight axes


class AffineFrame(CoordinateSystem[GenericSpace, NameT], Interface):
    """
    A coordinate system given by an origin and linearly independent, straight axes; the
    coordinates of a point are its amounts along the axes from the origin.

    The axes may differ in length and need not be perpendicular, as the axes of an
    image's pixels may.
    """


GenericEuclideanSpace = TypeVar("GenericEuclideanSpace", bound=EuclideanSpace)
"""
The kind of Euclidean space a Cartesian frame charts.
"""


class CartesianFrame(AffineFrame[GenericEuclideanSpace, NameT], Interface):
    """
    An affine frame whose axes are perpendicular and one unit of its Euclidean space
    long.

    Every Cartesian frame measures lengths and angles the same way, so the metric belongs
    to the kind of frame and needs no frame instance.
    """

    @classmethod
    def inner_product(cls, *, first: Vector, second: Vector) -> GenericCoordinate:
        """
        :param first: A vector given in a frame of this kind.
        :param second: A vector given in the same frame.
        :return: The product of both lengths and the cosine of the angle between them.
        """
        return sum(
            first_coordinate * second_coordinate
            for first_coordinate, second_coordinate in zip(
                first.coordinates, second.coordinates, strict=True
            )
        )

    @classmethod
    def norm(cls, *, vector: Vector) -> GenericCoordinate:
        """
        :param vector: A vector given in a frame of this kind.
        :return: The length of ``vector``.
        """
        return cls.inner_product(first=vector, second=vector) ** 0.5

    @classmethod
    def distance(cls, *, first: Point, second: Point) -> GenericCoordinate:
        """
        :param first: A point given in a frame of this kind.
        :param second: A point given in the same frame.
        :return: The straight-line distance between both points.
        """
        return cls.norm(vector=first - second)


# %% values given in a coordinate system

GenericCoordinate = TypeVar("GenericCoordinate")
"""
The type a coordinate's real value is stored as, such as a float, or a symbolic
expression standing for a real number.
"""

GenericCoordinateSystem = TypeVar("GenericCoordinateSystem", bound=CoordinateSystem)
"""
The kind of coordinate system coordinates are given in.
"""


class CoordinateRepresentation(
    Generic[GenericCoordinateSystem], SubClassSafeGeneric, Interface
):
    """
    A geometric value, such as a point, a direction or an orientation, given by its
    coordinates in a coordinate system.

    .. seealso::
        T. De Laet, S. Bellens and H. Bruyninckx, "Geometric Relations Between Rigid
        Bodies (Part 1): Semantics for Standardization", IEEE Robotics & Automation
        Magazine, 2013, which separates a geometric value from its coordinates.
    """

    coordinate_system: GenericCoordinateSystem | None
    """
    The coordinate system the coordinates are given in; ``None`` where it is not known.
    """


GenericCoordinateRepresentation = TypeVar(
    "GenericCoordinateRepresentation", bound=CoordinateRepresentation
)
"""
The kind of value given in a coordinate system.
"""


# %% changing coordinates within one space


class CoordinateChange(
    Generic[GenericCoordinateSystem], SubClassSafeGeneric, Interface
):
    """
    Turns coordinates given in one coordinate system into those of the same points in
    another coordinate system of the same space.

    .. note::
        Each implementation binds the kind of coordinate system once, so the kind of
        change a space allows is part of its type. A map between two spaces is a
        :class:`~krrood_spatial_math.space_map.SpaceMap`, not a coordinate change.
    """

    source_coordinate_system: GenericCoordinateSystem | None
    """
    The coordinate system the coordinates are given in; ``None`` where it is not known.
    """

    target_coordinate_system: GenericCoordinateSystem | None
    """
    The coordinate system the coordinates are changed into; ``None`` where it is not
    known.
    """

    @abstractmethod
    def inverse(self) -> Self:
        """
        :return: The change from the target coordinate system back to the source one.
        """

    @abstractmethod
    def __matmul__(self, other: Self) -> Self:
        """
        :param other: The change applied first, whose target coordinate system is the
            source one of this change.
        :return: The change that applies the other change and then this one.
        """


GenericAffineFrame = TypeVar("GenericAffineFrame", bound=AffineFrame)
"""
The kind of affine frame coordinates are given in.
"""


class AffineCoordinateChange(CoordinateChange[GenericAffineFrame], Interface):
    """
    A coordinate change between two affine frames, which scales, shears, rotates or
    reflects the axes and shifts the origin.

    It keeps straight lines, parallel lines and ratios of lengths along one line.
    """


GenericCartesianFrame = TypeVar("GenericCartesianFrame", bound=CartesianFrame)
"""
The kind of Cartesian frame coordinates are given in.
"""


class RigidCoordinateChange(AffineCoordinateChange[GenericCartesianFrame], Interface):
    """
    An affine coordinate change between two Cartesian frames, which only rotates the
    axes and shifts the origin.

    It also keeps distances and angles.
    """
