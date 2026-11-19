"""
Points of an affine space, given by their coordinates in an affine frame.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Generic, Self, TypeVar

from krrood_spatial_math.coordinate_system import (
    AxisCoordinates,
    GenericAffineFrame,
    GenericCoordinate,
)
from krrood_spatial_math.exceptions import PointAndVectorFramesDifferError
from krrood_spatial_math.vector import GenericVector

# %% points given in an affine frame


@dataclass(kw_only=True, eq=False)
class Point(
    AxisCoordinates[GenericAffineFrame],
    Generic[GenericAffineFrame, GenericVector, GenericCoordinate],
):
    """
    A point of an affine space, given by its amounts along an affine frame's axes from
    the frame's origin; a subclass declares one field per axis, such as ``u`` and ``v``.
    The difference of two points is a vector, and a point moved by a vector is a point.

    .. note::
        The type parameters are, in order, the kind of frame, the vector type and the
        coordinate type, which is :class:`float` unless given.
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
        return self.cartesian_frame_kind().distance(first=self, second=other)


GenericPoint = TypeVar("GenericPoint", bound=Point)
"""
The kind of point of an affine space.
"""
