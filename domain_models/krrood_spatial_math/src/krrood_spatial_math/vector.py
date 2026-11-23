"""
Vectors of an affine space: the translations that move its points.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Generic, Self, TypeVar

from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.exceptions import (
    MathematicalOperation,
    UnsupportedOperationError,
)
from krrood_spatial_math.coordinate_system import (
    AxisCoordinates,
    GenericAffineFrame,
    GenericCoordinate,
    GenericCoordinateSystem,
)
from krrood_spatial_math.exceptions import SpaceMeasuresNoLengthsError
from krrood_spatial_math.space import EuclideanSpace

# %% vectors given in an affine frame


@dataclass(kw_only=True, eq=False)
class Vector(
    AxisCoordinates[GenericAffineFrame, GenericCoordinate],
    Generic[GenericAffineFrame, GenericCoordinate],
):
    """
    A translation of an affine space, given by its amounts along the axes of an affine
    frame, one per axis its kind of frame names, such as ``u`` and ``v``.

    A vector has no position: a change of frame turns its axes but never shifts it.

    .. note::
        The type parameters are, in order, the kind of frame and the coordinate type.
    """

    @classproperty
    def has_position(cls) -> bool:
        return False

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
        :return: The length of this vector: the square root of its inner product with
            itself.
        :raises SpaceMeasuresNoLengthsError: If its kind of frame charts a space that is
            not Euclidean.
        :raises FrameMeasuresNoLengthsError: If the Gram matrix of its kind of frame is
            not known.
        """
        return self.dot(self) ** 0.5

    def dot(self, other: Self) -> GenericCoordinate:
        """
        :param other: A vector given in the same frame.
        :return: The product of both lengths and the cosine of the angle between them,
            as the Euclidean space of its kind of frame measures it.
        :raises UnsupportedOperationError: If ``other`` is not a vector of this type.
        :raises ReferenceFrameMismatchError: If both vectors are given in a frame and the
            frames differ.
        :raises SpaceMeasuresNoLengthsError: If its kind of frame charts a space that is
            not Euclidean.
        :raises FrameMeasuresNoLengthsError: If the Gram matrix of its kind of frame is
            not known.
        """
        if not isinstance(other, type(self)):
            raise UnsupportedOperationError(
                operation=MathematicalOperation.DOT_PRODUCT, left=self, right=other
            )
        self.ensure_same_coordinate_system(other=other, context="the multiplied vector")
        return self._measure_inner_product(other=other)

    def _measure_inner_product(self, *, other: Self) -> GenericCoordinate:
        """
        :param other: A vector of this type, given in the same frame.
        :return: The inner product of this vector and ``other``, which
            :meth:`dot` checks the operands of.
        :raises SpaceMeasuresNoLengthsError: If its kind of frame charts a space that is
            not Euclidean.
        """
        space_kind = self.frame_kind.space_kind
        if not issubclass(space_kind, EuclideanSpace):
            raise SpaceMeasuresNoLengthsError(space_kind=space_kind)
        return space_kind.inner_product(first=self, second=other)


GenericVector = TypeVar("GenericVector", bound=Vector)
"""
The kind of vector the points of an affine space are moved by.
"""
