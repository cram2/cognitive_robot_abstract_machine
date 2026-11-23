"""
Cartesian frames on a three-dimensional Euclidean space, and how serialized values refer
to them.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from uuid import UUID

from typing_extensions import TYPE_CHECKING, Any

from krrood.adapters.json_serializer import from_json
from krrood.adapters.object_reference import ObjectReference
from krrood.patterns.classproperty import classproperty
from krrood.patterns.identifier_and_name import NameT
from krrood.patterns.interface import Interface
from krrood_spatial_math.coordinate_system import (
    FrameAxis,
    CartesianFrame,
    GenericCoordinateRepresentation,
)
from krrood_spatial_math.exceptions import (
    FrameOutsideSpaceError,
    FramesOfDifferentSpacesError,
    MissingCartesianFrame3DimensionalResolver,
    MissingReferenceFrameError,
)
from krrood_spatial_math.space import EuclideanSpace

if TYPE_CHECKING:
    from krrood_spatial_math.embedded_plane import EmbeddedPlane
    from krrood_spatial_math.vector import Vector

# %% frames that positions and directions are expressed in


class CartesianFrame3Dimensional(CartesianFrame[EuclideanSpace, NameT], Interface):
    """
    A right-handed Cartesian frame on a three-dimensional Euclidean space, in metres,
    such as the frame of a robot's body.
    """

    @classproperty
    def axes(cls) -> tuple[FrameAxis, ...]:
        return (FrameAxis.X, FrameAxis.Y, FrameAxis.Z)

    @classmethod
    def from_frame_3_dimensional(
        cls, *, frame: CartesianFrame3Dimensional
    ) -> CartesianFrame3Dimensional:
        """
        :param frame: The reference frame of a value given in a frame of this kind.
        :return: The frame of this kind that ``frame`` fixes, which is ``frame`` itself.
        """
        return frame

    @property
    def frame_3_dimensional(self) -> CartesianFrame3Dimensional:
        """
        The reference frame of the values given in this frame, which is this frame
        itself.
        """
        return self

    def express_in_own_frame(
        self, *, value: GenericCoordinateRepresentation
    ) -> GenericCoordinateRepresentation:
        """
        :param value: A value given in a frame of the same space as this frame.
        :return: The same value, given in this frame.
        :raises MissingReferenceFrameError: If the value is given in no frame.
        :raises FrameOutsideSpaceError: If this frame or the value's frame charts no
            known space.
        :raises FramesOfDifferentSpacesError: If the value's frame charts another space.
        """
        value_frame = value.coordinate_system
        if value_frame is None:
            raise MissingReferenceFrameError(spatial_type=value)
        for frame in (self, value_frame):
            if frame.space is None:
                raise FrameOutsideSpaceError(frame=frame)
        if value_frame.space is not self.space:
            raise FramesOfDifferentSpacesError(frame=self, value_frame=value_frame)
        return self.transform_into_own_frame(value=value)

    @abstractmethod
    def transform_into_own_frame(
        self, *, value: GenericCoordinateRepresentation
    ) -> GenericCoordinateRepresentation:
        """
        :param value: A value given in a frame of the same space as this frame.
        :return: The same value, given in this frame.
        """

    @property
    @abstractmethod
    def z_axis(self) -> Vector:
        """
        The unit vector along this frame's z-axis, given in this frame, which is the
        normal of the plane its x- and y-axes span.
        """

    @property
    @abstractmethod
    def embedded_plane(self) -> EmbeddedPlane:
        """
        The plane this frame's x-y plane lies on, with its z-axis pointing the same way,
        in every state of the space it charts; frames on one plane are related by a
        rigid change on that plane.
        """


class CartesianFrame3DimensionalResolver(Interface):
    """
    Finds the Cartesian frames serialized coordinate representations refer to, while one
    JSON document is deserialized.

    It travels through nested ``from_json`` calls in their keyword arguments.
    """

    @abstractmethod
    def get(self, key: UUID, name: Any = None) -> CartesianFrame3Dimensional:
        """
        :param key: The identifier the document refers to the frame with.
        :param name: The name the reference to the frame went by, which says which frame
            was meant when it cannot be found.
        :return: The frame with that identifier.
        """


# %% referring to a frame in a serialized coordinate representation


@dataclass(kw_only=True)
class CartesianFrame3DimensionalReference(ObjectReference):
    """
    How a serialized coordinate representation points at a
    :class:`CartesianFrame3Dimensional`.
    """

    def resolve(
        self, *, data: dict[str, Any], **kwargs: Any
    ) -> CartesianFrame3Dimensional:
        """
        :param data: The serialized spatial type that refers to the frame.
        :param kwargs: The keyword arguments of the ``from_json`` call, which carry the
            :class:`CartesianFrame3DimensionalResolver` that finds the frame.
        :return: The frame the serialized spatial type refers to.
        :raises MissingCartesianFrame3DimensionalResolver: If the kwargs carry no
            :class:`CartesianFrame3DimensionalResolver`.
        """
        resolvers = [
            value
            for value in kwargs.values()
            if isinstance(value, CartesianFrame3DimensionalResolver)
        ]
        if not resolvers:
            raise MissingCartesianFrame3DimensionalResolver(frame_id_key=self.id_key)
        return resolvers[0].get(
            key=from_json(data[self.id_key]), name=from_json(data.get(self.name_key))
        )
