"""
Vectors of an affine space: the translations that move its points.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Generic, Self, TypeVar

from krrood.symbolic_math.exceptions import UnsupportedOperationError
from krrood_spatial_math.coordinate_system import (
    AxisCoordinates,
    GenericAffineFrame,
    GenericCoordinate,
    GenericCoordinateSystem,
)

# %% vectors given in an affine frame


@dataclass(kw_only=True, eq=False)
class Vector(
    AxisCoordinates[GenericAffineFrame],
    Generic[GenericAffineFrame, GenericCoordinate],
):
    """
    A translation of an affine space, given by its amounts along the axes of an affine
    frame; a subclass marks one field per axis, such as ``u`` and ``v``, with
    :class:`~krrood_spatial_math.coordinate_system.AxisCoordinate`.

    A vector has no position: a change of frame turns its axes but never shifts it.

    .. note::
        The type parameters are, in order, the kind of frame and the coordinate type,
        which is :class:`float` unless given.
    """

    def __add__(self, other: Self) -> Self:
        """
        :param other: A vector given in the same frame.
        :return: The translation by this vector followed by ``other``.
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        self.ensure_same_coordinate_system(other=other, context="the added vector")
        return self.from_coordinates(
            coordinates=tuple(
                own + others
                for own, others in zip(self.coordinates, other.coordinates, strict=True)
            ),
            coordinate_system=self.coordinate_system,
        )

    def __sub__(self, other: Self) -> Self:
        """
        :param other: A vector given in the same frame.
        :return: The vector that ``other`` has to be followed by to give this one.
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        self.ensure_same_coordinate_system(other=other, context="the subtracted vector")
        return self.from_coordinates(
            coordinates=tuple(
                own - others
                for own, others in zip(self.coordinates, other.coordinates, strict=True)
            ),
            coordinate_system=self.coordinate_system,
        )

    def __mul__(self, factor: GenericCoordinate) -> Self:
        """
        :param factor: The number every coordinate is multiplied by.
        :return: This vector stretched by ``factor``.
        """
        return self.from_coordinates(
            coordinates=tuple(coordinate * factor for coordinate in self.coordinates),
            coordinate_system=self.coordinate_system,
        )

    def norm(self) -> GenericCoordinate:
        """
        :return: The length of this vector.
        :raises FrameMeasuresNoLengthsError: If its kind of frame is not Cartesian.
        """
        frame_kind = self.get_type_of_generic_parameter(
            parameter=GenericCoordinateSystem
        )
        return frame_kind.norm(vector=self)

    def dot(self, other: Self) -> GenericCoordinate:
        """
        :param other: A vector given in the same frame.
        :return: The product of both lengths and the cosine of the angle between them.
        :raises UnsupportedOperationError: If ``other`` is not a vector of this type.
        :raises ReferenceFrameMismatchError: If both vectors are given in a frame and the
            frames differ.
        :raises FrameMeasuresNoLengthsError: If its kind of frame is not Cartesian.
        """
        if not isinstance(other, type(self)):
            raise UnsupportedOperationError(operation="dot", left=self, right=other)
        self.ensure_same_coordinate_system(other=other, context="the multiplied vector")
        frame_kind = self.get_type_of_generic_parameter(
            parameter=GenericCoordinateSystem
        )
        return frame_kind.inner_product(first=self, second=other)


GenericVector = TypeVar("GenericVector", bound=Vector)
"""
The kind of vector the points of an affine space are moved by.
"""
