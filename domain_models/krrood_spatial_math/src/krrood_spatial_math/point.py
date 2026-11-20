"""
Points of an affine space, given by their coordinates in an affine frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Generic, Self, TypeVar

from krrood.patterns.classproperty import classproperty
from krrood.patterns.subclass_safe_generic import DerivedGenericParameter
from krrood_spatial_math.coordinate_system import (
    AxisCoordinates,
    GenericCoordinateChange,
    GenericCoordinate,
    GenericCoordinateSystem,
)
from krrood_spatial_math.vector import GenericVector

# %% points given in an affine frame


@dataclass(kw_only=True, eq=False)
class Point(AxisCoordinates, Generic[GenericVector]):
    """
    A point of the affine space its vector type translates: the difference of two points
    is such a vector, and adding one moves the point. It is given by its amounts along
    the axes of an affine frame, one per axis its kind of frame names.

    .. note::
        The type parameter is the vector type, which fixes the kind of frame, the
        coordinate type and, where the vector type binds one, the change that
        re-expresses it, so a point and its vector always share them.
    """

    @classproperty
    def derived_generic_parameters(cls) -> tuple[DerivedGenericParameter, ...]:
        """
        The kind of frame, the coordinate type and the change type, read from the vector
        type.
        """
        return (
            DerivedGenericParameter(
                parameter=GenericCoordinateSystem, source=GenericVector
            ),
            DerivedGenericParameter(parameter=GenericCoordinate, source=GenericVector),
            DerivedGenericParameter(
                parameter=GenericCoordinateChange, source=GenericVector
            ),
        )

    @classproperty
    def has_position(cls) -> bool:
        return True

    def __sub__(self, other: Self) -> GenericVector:
        """
        :param other: A point given in the same frame.
        :return: The vector that moves ``other`` onto this point.
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        self.ensure_same_coordinate_system(other=other, context="the subtracted point")
        vector_type = self.get_type_of_generic_parameter(parameter=GenericVector)
        return vector_type.from_coordinates(
            coordinates=tuple(
                own - others
                for own, others in zip(self.coordinates, other.coordinates, strict=True)
            ),
            coordinate_system=self.coordinate_system,
        )

    def __add__(self, vector: GenericVector) -> Self:
        """
        :param vector: A vector given in the same frame.
        :return: This point moved by ``vector``.
        :raises ReferenceFrameMismatchError: If ``vector`` is given in another frame.
        """
        self.ensure_same_coordinate_system(other=vector, context="the added vector")
        return self.from_coordinates(
            coordinates=tuple(
                own + moved
                for own, moved in zip(self.coordinates, vector.coordinates, strict=True)
            ),
            coordinate_system=self.coordinate_system,
        )

    def euclidean_distance(self, other: Self) -> GenericCoordinate:
        """
        :param other: A point given in the same frame.
        :return: The straight-line distance between this point and ``other``.
        :raises FrameMeasuresNoLengthsError: If its kind of frame is not Cartesian.
        """
        frame_kind = self.get_type_of_generic_parameter(
            parameter=GenericCoordinateSystem
        )
        return frame_kind.distance(first=self, second=other)


GenericPoint = TypeVar("GenericPoint", bound=Point)
"""
The kind of point of an affine space.
"""
