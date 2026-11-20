"""
Coordinate systems on a space, the values given in them, and the changes between
coordinate systems of one space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from typing_extensions import TYPE_CHECKING, Any, Generic, Self, TypeVar

from krrood.patterns.field_metadata import FieldMetadata
from krrood.patterns.identifier_and_name import HasID, HasName, NameT
from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_spatial_math.exceptions import (
    FrameMeasuresNoLengthsError,
    ReferenceFrameMismatchError,
)
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
    image's pixels may, so an affine frame measures no lengths or angles.
    """

    @classmethod
    def inner_product(cls, *, first: Vector, second: Vector) -> GenericCoordinate:
        """
        :param first: A vector given in a frame of this kind.
        :param second: A vector given in the same frame.
        :return: The product of both lengths and the cosine of the angle between them.
        :raises FrameMeasuresNoLengthsError: If this kind of frame measures no lengths.
        """
        raise FrameMeasuresNoLengthsError(frame_kind=cls)

    @classmethod
    def norm(cls, *, vector: Vector) -> GenericCoordinate:
        """
        :param vector: A vector given in a frame of this kind.
        :return: The length of ``vector``.
        :raises FrameMeasuresNoLengthsError: If this kind of frame measures no lengths.
        """
        return cls.inner_product(first=vector, second=vector) ** 0.5

    @classmethod
    def distance(cls, *, first: Point, second: Point) -> GenericCoordinate:
        """
        :param first: A point given in a frame of this kind.
        :param second: A point given in the same frame.
        :return: The straight-line distance between both points.
        :raises FrameMeasuresNoLengthsError: If this kind of frame measures no lengths.
        """
        return cls.norm(vector=first - second)


GenericEuclideanSpace = TypeVar("GenericEuclideanSpace", bound=EuclideanSpace)
"""
The kind of Euclidean space a Cartesian frame charts.
"""


class CartesianFrame(AffineFrame[GenericEuclideanSpace, NameT], Interface):
    """
    An affine frame whose axes are perpendicular and one unit of its Euclidean space
    long, so it measures lengths and angles.

    Every Cartesian frame measures them the same way, so the metric belongs to the kind
    of frame and needs no frame instance.
    """

    @classmethod
    def inner_product(cls, *, first: Vector, second: Vector) -> GenericCoordinate:
        """
        The sum of the products of the coordinates, unlike :meth:`AffineFrame.inner_product`.
        """
        return sum(
            first_coordinate * second_coordinate
            for first_coordinate, second_coordinate in zip(
                first.coordinates, second.coordinates, strict=True
            )
        )


# %% values given in a coordinate system

GenericCoordinate = TypeVar("GenericCoordinate", default=float)
"""
The type a coordinate's real value is stored as, such as a float, or a symbolic
expression standing for a real number; a float unless a class binds another type.
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

    def ensure_same_coordinate_system(
        self, *, other: CoordinateRepresentation, context: str
    ) -> None:
        """
        :param other: A value combined with this one, whose coordinates must be given in
            the same coordinate system.
        :param context: What ``other`` is to the operation, for the error message.
        :raises ReferenceFrameMismatchError: If both values are given in a coordinate
            system and the two differ.
        """
        if (
            self.coordinate_system is not None
            and other.coordinate_system is not None
            and other.coordinate_system.id != self.coordinate_system.id
        ):
            raise ReferenceFrameMismatchError(
                expected_frame=self.coordinate_system,
                actual_frame=other.coordinate_system,
                context=context,
            )


GenericCoordinateRepresentation = TypeVar(
    "GenericCoordinateRepresentation", bound=CoordinateRepresentation
)
"""
The kind of value given in a coordinate system.
"""


# %% values given by one coordinate per axis of a frame


@dataclass(kw_only=True)
class AxisCoordinate(FieldMetadata):
    """
    Marks a field of an :class:`AxisCoordinates` value as its coordinate along one axis.
    The marked fields are the axes in the order they are declared.
    """


@dataclass(kw_only=True, eq=False)
class AxisCoordinates(CoordinateRepresentation[GenericCoordinateSystem], ABC):
    """
    A value given by one coordinate per axis of its coordinate system, such as a point or
    a vector, whose coordinates are the fields its class marks with :class:`AxisCoordinate`.
    """

    coordinate_system: GenericCoordinateSystem | None
    """
    The coordinate system the coordinates are given in; ``None`` where it is not known.
    """

    @property
    def coordinates(self) -> tuple[Any, ...]:
        """
        The amounts along the axes of the coordinate system, one per axis, in the order
        of the axes.
        """
        return tuple(getattr(self, name) for name in self._coordinate_field_names())

    @classmethod
    def from_coordinates(
        cls,
        *,
        coordinates: tuple[Any, ...],
        coordinate_system: GenericCoordinateSystem | None,
    ) -> Self:
        """
        :param coordinates: One amount per axis, in the order of the axes.
        :param coordinate_system: The coordinate system the coordinates are given in.
        :return: The value with these coordinates.
        """
        return cls(
            **dict(zip(cls._coordinate_field_names(), coordinates, strict=True)),
            coordinate_system=coordinate_system,
        )

    @classmethod
    def _coordinate_field_names(cls) -> tuple[str, ...]:
        """
        :return: The names of the fields marked with :class:`AxisCoordinate`, in order.
        """
        return tuple(
            coordinate_field.name
            for coordinate_field in fields(cls)
            if AxisCoordinate in coordinate_field.metadata
        )


# %% changing coordinates within one space


class CoordinateChange(
    Generic[GenericCoordinateSystem, GenericCoordinate], SubClassSafeGeneric, Interface
):
    """
    Turns coordinates given in one coordinate system into those of the same points in
    another coordinate system of the same space.

    .. note::
        Each implementation binds the kind of coordinate system once, so the kind of
        change a space allows is part of its type, and the coordinate type, which is
        :class:`float` unless given. A map between two spaces is a
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


class AffineCoordinateChange(
    CoordinateChange[GenericAffineFrame, GenericCoordinate], Interface
):
    """
    A coordinate change between two affine frames, which scales, shears, rotates or
    reflects the axes and shifts the origin.

    It keeps straight lines, parallel lines and ratios of lengths along one line.
    """


GenericCartesianFrame = TypeVar("GenericCartesianFrame", bound=CartesianFrame)
"""
The kind of Cartesian frame coordinates are given in.
"""


class RigidCoordinateChange(
    AffineCoordinateChange[GenericCartesianFrame, GenericCoordinate], Interface
):
    """
    An affine coordinate change between two Cartesian frames, which only rotates the
    axes and shifts the origin.

    It also keeps distances and angles.
    """
