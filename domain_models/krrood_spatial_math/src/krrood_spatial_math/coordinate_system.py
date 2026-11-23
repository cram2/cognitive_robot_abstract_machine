"""
Coordinate systems on a space, the values given in them, and the changes between
coordinate systems of one space.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, Any, Generic, Self, TypeVar

from krrood.patterns.classproperty import classproperty
from krrood.patterns.identifier_and_name import HasID, HasName, NameT
from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood_spatial_math.exceptions import (
    FrameHasNoCrossProductError,
    FrameMeasuresNoLengthsError,
    MissingAxisCoordinatesError,
    ReferenceFrameMismatchError,
)
from krrood_spatial_math.space import EuclideanSpace, Space

if TYPE_CHECKING:
    from krrood_spatial_math.pose import GenericPose
    from krrood_spatial_math.vector import GenericVector, Vector

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


class FrameAxis(StrEnum):
    """
    An axis of a frame with straight axes, named as the coordinates along it are; the
    first, second and third axes of most frames.
    """

    X = "x"
    """
    The first axis.
    """

    Y = "y"
    """
    The second axis.
    """

    Z = "z"
    """
    The third axis.
    """


class AffineFrame(CoordinateSystem[GenericSpace, NameT], Interface):
    """
    A coordinate system given by an origin and linearly independent, straight axes; the
    coordinates of a point are its amounts along the axes from the origin.

    The axes may differ in length and need not be perpendicular, as an image's pixels
    may, so its coordinates alone give no lengths or angles.
    """

    @classproperty
    @abstractmethod
    def axes(cls) -> tuple[StrEnum, ...]:
        """
        The axes of a frame of this kind, in order; a value given in such a frame holds
        its coordinate along each axis in the attribute named after it.
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
    def cross_product(
        cls, *, first: GenericVector, second: GenericVector
    ) -> GenericVector:
        """
        :param first: A vector given in a frame of this kind.
        :param second: A vector given in the same frame.
        :return: The vector perpendicular to both, as long as the area of the
            parallelogram they span, such that ``first``, ``second`` and the result are
            right-handed.
        :raises FrameHasNoCrossProductError: If this kind of frame is not an oriented
            three-dimensional Cartesian frame.
        """
        raise FrameHasNoCrossProductError(frame_kind=cls)


GenericAffineFrame = TypeVar("GenericAffineFrame", bound=AffineFrame)
"""
The kind of affine frame coordinates are given in.
"""


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

    @classmethod
    def from_constructor_arguments(
        cls,
        *,
        coordinate_system: GenericCoordinateSystem | None,
        **constructor_arguments: Any,
    ) -> Self:
        """
        :param coordinate_system: The coordinate system the new value is given in.
        :param constructor_arguments: This type's other constructor arguments.
        :return: The value these arguments give, in ``coordinate_system``.
        """
        return cls(coordinate_system=coordinate_system, **constructor_arguments)

    @abstractmethod
    def apply_transformation(self, *, transformation: CoordinateChange) -> Self:
        """
        :param transformation: A change whose source coordinate system is the one this
            value is given in.
        :return: The same value, given in the target coordinate system of
            ``transformation``.
        """


GenericCoordinateRepresentation = TypeVar(
    "GenericCoordinateRepresentation", bound=CoordinateRepresentation
)
"""
The kind of value given in a coordinate system.
"""


# %% values given by one coordinate per axis of a frame


@dataclass(kw_only=True, eq=False)
class AxisCoordinates(
    CoordinateRepresentation[GenericAffineFrame],
    Generic[GenericAffineFrame, GenericCoordinate],
    ABC,
):
    """
    A value given by one coordinate per axis of its kind of frame, such as a point or a
    vector, held in the attribute named after that axis.
    """

    coordinate_system: GenericAffineFrame | None
    """
    The coordinate system the coordinates are given in; ``None`` where it is not known.
    """

    @property
    def coordinates(self) -> tuple[GenericCoordinate, ...]:
        """
        The amounts along the axes of the coordinate system, one per axis, in the order
        of the axes.
        """
        return tuple(getattr(self, axis) for axis in self.frame_kind.axes)

    @classproperty
    def frame_kind(cls) -> type[GenericAffineFrame]:
        """
        The kind of frame values of this type are given in, which names their axes and
        measures their lengths and angles.
        """
        return cls.get_type_of_generic_parameter(parameter=GenericAffineFrame)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        :raises MissingAxisCoordinatesError: If a concrete value type lacks a field or
            property for one of the axes of its kind of frame.
        """
        super().__init_subclass__(**kwargs)
        if Interface.is_abstract(class_=cls) or not cls.binds_generic_parameter(
            parameter=GenericAffineFrame
        ):
            return
        missing_axes = tuple(
            axis
            for axis in cls.frame_kind.axes
            if not any(
                axis in vars(base) or axis in inspect.get_annotations(base)
                for base in cls.__mro__
            )
        )
        if missing_axes:
            raise MissingAxisCoordinatesError(value_type=cls, axes=missing_axes)

    @classmethod
    def from_coordinates(
        cls,
        *,
        coordinates: tuple[GenericCoordinate, ...],
        coordinate_system: GenericAffineFrame | None,
    ) -> Self:
        """
        :param coordinates: One amount per axis, in the order of the axes.
        :param coordinate_system: The coordinate system the coordinates are given in.
        :return: The value with these coordinates.
        """
        return cls.from_constructor_arguments(
            coordinate_system=coordinate_system,
            **dict(zip(cls.frame_kind.axes, coordinates, strict=True)),
        )

    @classproperty
    @abstractmethod
    def has_position(cls) -> bool:
        """
        Whether a value of this type is a position, which an affine change shifts along
        with the origin, rather than an offset, which it only turns or scales; the last
        of its homogeneous coordinates, 1 or 0.
        """

    def apply_transformation(
        self, *, transformation: AffineCoordinateChange[GenericAffineFrame, Any]
    ) -> Self:
        return transformation.express_axis_coordinates(value=self)


GenericAxisCoordinates = TypeVar("GenericAxisCoordinates", bound=AxisCoordinates)
"""
The kind of value given by one coordinate per axis of its frame.
"""


# %% changing coordinates within one space


class CoordinateChange(
    Generic[GenericCoordinateSystem, GenericCoordinate], SubClassSafeGeneric, Interface
):
    """
    Turns coordinates given in one coordinate system into those of the same points in
    another coordinate system of the same space.

    .. note::
        Each implementation binds the kind of coordinate system once, so the kind of
        change a space allows is part of its type, and the coordinate type. A map
        between two spaces is a
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


GenericCoordinateChange = TypeVar("GenericCoordinateChange", bound=CoordinateChange)
"""
The kind of change that re-expresses a value in another coordinate system of its kind.
"""


class AffineCoordinateChange(
    CoordinateChange[GenericAffineFrame, GenericCoordinate], Interface
):
    """
    A coordinate change between two affine frames, which scales, shears, rotates or
    reflects the axes and shifts the origin.

    It keeps straight lines, parallel lines and ratios of lengths along one line.
    """

    @abstractmethod
    def apply_to_homogeneous_coordinates(
        self, *, homogeneous_coordinates: tuple[GenericCoordinate, ...]
    ) -> tuple[GenericCoordinate, ...]:
        """
        :param homogeneous_coordinates: The coordinates of a value in the source
            coordinate system, followed by 1 if the value
            :attr:`~AxisCoordinates.has_position` and 0 if not.
        :return: The same coordinates in the target coordinate system, followed by the
            same last entry: the axes are mapped, and the shift between the origins is
            added only to a value that has a position.
        """

    def express_axis_coordinates(
        self, *, value: GenericAxisCoordinates
    ) -> GenericAxisCoordinates:
        """
        :param value: A point or vector given in the source coordinate system.
        :return: The same point or vector, given in the target coordinate system.
        """
        moved = self.apply_to_homogeneous_coordinates(
            homogeneous_coordinates=(*value.coordinates, int(value.has_position))
        )
        return value.from_coordinates(
            coordinates=moved[:-1], coordinate_system=self.target_coordinate_system
        )


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

    @property
    @abstractmethod
    def rotation_matrix(self) -> RigidCoordinateChange:
        """
        This change without the shift between the origins: it turns the axes only.
        """

    @property
    @abstractmethod
    def translation(self) -> Vector:
        """
        Where the source origin lies, along the axes of the target coordinate system.
        """

    def express_pose(self, *, value: GenericPose) -> GenericPose:
        """
        :param value: A pose given in the source coordinate system.
        :return: The same pose, given in the target coordinate system.
        """
        return value.from_coordinate_change(change=self @ value.to_coordinate_change())
