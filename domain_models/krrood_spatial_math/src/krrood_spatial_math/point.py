"""
Points of an affine space, given by their coordinates in an affine frame.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from typing_extensions import Any, Generic, Self, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.mixin import Mixin
from krrood_spatial_math.coordinate_system import (
    GenericAffineFrame,
    GenericCartesianFrame,
    CoordinateRepresentation,
    GenericCoordinate,
    StoredCoordinateRepresentation,
)
from krrood_spatial_math.exceptions import PointAndVectorFramesDifferError
from krrood_spatial_math.vector import GenericVector

# %% points given in an affine frame


class Point(
    CoordinateRepresentation[GenericAffineFrame],
    Generic[GenericAffineFrame, GenericVector, GenericCoordinate],
    Interface,
):
    """
    A point of an affine space, given by its amounts along an affine frame's axes from
    the frame's origin. The difference of two points is a vector, and a point moved by a
    vector is a point; two points never add up to a point.

    .. note::
        The type parameters are, in order, the kind of frame, the vector type and the
        coordinate type, which is :class:`float` unless given. Each implementation binds
        them once, so one kind of frame has one point type.
    """

    coordinates: tuple[GenericCoordinate, ...]
    """
    The amounts along the axes of the frame from its origin, one per axis, in the order
    of the axes.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Also refuses a point type bound to a vector type of another kind of frame.

        :raises PointAndVectorFramesDifferError: If the bound vector type is given in
            another kind of frame than the point type.
        """
        super().__init_subclass__(**kwargs)
        if not (
            cls.binds_generic_parameter(parameter=GenericAffineFrame)
            and cls.binds_generic_parameter(parameter=GenericVector)
        ):
            return
        vector_type = cls.get_type_of_generic_parameter(parameter=GenericVector)
        if not vector_type.binds_generic_parameter(parameter=GenericAffineFrame):
            return
        point_frame_type = cls.get_type_of_generic_parameter(
            parameter=GenericAffineFrame
        )
        vector_frame_type = vector_type.get_type_of_generic_parameter(
            parameter=GenericAffineFrame
        )
        if vector_frame_type is not point_frame_type:
            raise PointAndVectorFramesDifferError(
                point_type=cls,
                point_frame_type=point_frame_type,
                vector_type=vector_type,
                vector_frame_type=vector_frame_type,
            )

    @abstractmethod
    def __sub__(self, other: Self) -> GenericVector:
        """
        :param other: A point given in the same frame.
        :return: The vector that moves ``other`` onto this point.
        """

    @abstractmethod
    def __add__(self, vector: GenericVector) -> Self:
        """
        :param vector: A vector given in the same frame.
        :return: This point moved by ``vector``.
        """


GenericPoint = TypeVar("GenericPoint", bound=Point)
"""
The kind of point of an affine space.
"""


# %% points whose frame kind measures the distances between them


class CartesianPoint(
    Point[GenericCartesianFrame, GenericVector, GenericCoordinate], Mixin
):
    """
    A point given in a Cartesian frame, whose distances to other points the kind of that
    frame measures.

    .. note::
        :meth:`euclidean_distance` gives the metric of
        :class:`~krrood_spatial_math.coordinate_system.CartesianFrame` on the value
        itself, so it reads like the point's own.
    """

    def euclidean_distance(self, other: Self) -> GenericCoordinate:
        """
        :param other: A point given in the same frame.
        :return: The straight-line distance between this point and ``other``.
        """
        frame_kind = self.get_type_of_generic_parameter(parameter=GenericCartesianFrame)
        return frame_kind.distance(first=self, second=other)


# %% points whose coordinates are the fields their class declares


@dataclass(kw_only=True)
class StoredPoint(
    StoredCoordinateRepresentation[GenericAffineFrame],
    Point[GenericAffineFrame, GenericVector, GenericCoordinate],
):
    """
    A point whose coordinates are the fields its class declares, such as ``u`` and
    ``v``, moved by and differing by its vector type coordinate by coordinate.
    """

    def __sub__(self, other: Self) -> GenericVector:
        """
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
