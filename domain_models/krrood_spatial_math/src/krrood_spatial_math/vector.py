"""
Vectors of an affine space: the translations that move its points.
"""

from __future__ import annotations

from abc import abstractmethod

from typing_extensions import Generic, Self, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.mixin import Mixin
from krrood.symbolic_math.exceptions import UnsupportedOperationError
from krrood_spatial_math.coordinate_system import (
    GenericAffineFrame,
    GenericCartesianFrame,
    CoordinateRepresentation,
    GenericCoordinate,
)
from krrood_spatial_math.exceptions import ReferenceFrameMismatchError

# %% vectors given in an affine frame


class Vector(
    CoordinateRepresentation[GenericAffineFrame],
    Generic[GenericAffineFrame, GenericCoordinate],
    Interface,
):
    """
    A translation of an affine space, given by its amounts along the axes of an affine
    frame.

    A vector has no position: a change of frame turns its axes but never shifts it.
    """

    coordinates: tuple[GenericCoordinate, ...]
    """
    The amounts along the axes of the frame, one per axis, in the order of the axes.
    """

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        """
        :param other: A vector given in the same frame.
        :return: The translation by this vector followed by ``other``.
        """

    @abstractmethod
    def __sub__(self, other: Self) -> Self:
        """
        :param other: A vector given in the same frame.
        :return: The vector that ``other`` has to be followed by to give this one.
        """

    @abstractmethod
    def __mul__(self, factor: GenericCoordinate) -> Self:
        """
        :param factor: The number every coordinate is multiplied by.
        :return: This vector stretched by ``factor``.
        """


GenericVector = TypeVar("GenericVector", bound=Vector)
"""
The kind of vector the points of an affine space are moved by.
"""


# %% vectors whose frame kind measures them


class MetricVector(Vector[GenericCartesianFrame, GenericCoordinate], Mixin):
    """
    A vector given in a Cartesian frame, whose length and angles the kind of that frame
    measures.

    .. note::
        :meth:`norm` and :meth:`dot` give the metric of
        :class:`~krrood_spatial_math.coordinate_system.CartesianFrame` on the value
        itself, so it reads like the vector's own.
    """

    def norm(self) -> GenericCoordinate:
        """
        :return: The length of this vector.
        """
        frame_kind = self.get_type_of_generic_parameter(parameter=GenericCartesianFrame)
        return frame_kind.norm(vector=self)

    def dot(self, other: Self) -> GenericCoordinate:
        """
        :param other: A vector given in the same frame.
        :return: The product of both lengths and the cosine of the angle between them.
        :raises UnsupportedOperationError: If ``other`` is not a vector of this type.
        :raises ReferenceFrameMismatchError: If both vectors are given in a frame and the
            frames differ.
        """
        if not isinstance(other, type(self)):
            raise UnsupportedOperationError(operation="dot", left=self, right=other)
        if (
            self.coordinate_system is not None
            and other.coordinate_system is not None
            and other.coordinate_system.id != self.coordinate_system.id
        ):
            raise ReferenceFrameMismatchError(
                expected_frame=self.coordinate_system,
                actual_frame=other.coordinate_system,
                context="the multiplied vector",
            )
        frame_kind = self.get_type_of_generic_parameter(parameter=GenericCartesianFrame)
        return frame_kind.inner_product(first=self, second=other)
