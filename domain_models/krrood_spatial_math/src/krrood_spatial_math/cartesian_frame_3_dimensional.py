"""Cartesian frames on a three-dimensional Euclidean space, and how serialized
values refer to them."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from uuid import UUID

from typing_extensions import Any

from krrood.adapters.json_serializer import from_json
from krrood.adapters.object_reference import ObjectReference
from krrood.patterns.identifier_and_name import NameT
from krrood.patterns.interface import Interface
from krrood_spatial_math.coordinate_system import (
    CartesianFrame,
    CoordinateRepresentationT,
)
from krrood_spatial_math.exceptions import (
    FrameOutsideSpaceError,
    FramesOfDifferentSpacesError,
    MissingCartesianFrame3DimensionalResolver,
    MissingReferenceFrameError,
)
from krrood_spatial_math.space import EuclideanSpace3Dimensional

# %% how the embedded planes of two frames lie


class EmbeddedPlaneRelation(Enum):
    """How the :class:`~krrood_spatial_math.embedded_plane.EmbeddedPlane` of
    one Cartesian frame, its x-y plane, lies relative to that of another."""

    SAME_PLANE_SAME_NORMAL = "same plane, same normal"
    """Both frames span one plane and their z-axes point to the same side of
    it, so a rigid change on that plane relates them."""

    SAME_PLANE_OPPOSITE_NORMAL = "same plane, opposite normal"
    """Both frames span one plane but their z-axes point to opposite sides of
    it, so the frames on that plane are mirror images."""

    DIFFERENT_PLANES = "different planes"
    """The frames span two planes: their z-axes are not parallel, one origin
    lies off the other's plane, or they chart two spaces."""

    UNKNOWN = "unknown"
    """Where the frames lie relative to each other is not known, such as when
    one charts no known space."""

    @property
    def is_same_plane(self) -> bool:
        """Whether both frames are known to span one plane, whichever side
        their z-axes point to."""
        return self in (
            EmbeddedPlaneRelation.SAME_PLANE_SAME_NORMAL,
            EmbeddedPlaneRelation.SAME_PLANE_OPPOSITE_NORMAL,
        )


# %% frames that positions and directions are expressed in


class CartesianFrame3Dimensional(
    CartesianFrame[EuclideanSpace3Dimensional, NameT], Interface
):
    """A right-handed Cartesian frame on a three-dimensional Euclidean space,
    in metres, such as the frame of a robot's body."""

    def express_in_own_frame(
        self, *, value: CoordinateRepresentationT
    ) -> CoordinateRepresentationT:
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
        self, *, value: CoordinateRepresentationT
    ) -> CoordinateRepresentationT:
        """
        :param value: A value given in a frame of the same space as this frame.
        :return: The same value, given in this frame.
        """

    def compare_embedded_plane(
        self, *, frame: CartesianFrame3Dimensional
    ) -> EmbeddedPlaneRelation:
        """
        :param frame: Any Cartesian frame.
        :return: How the embedded plane of ``frame`` lies relative to that of this frame;
            :attr:`EmbeddedPlaneRelation.UNKNOWN` when either frame charts no known space.
        """
        if frame is self:
            return EmbeddedPlaneRelation.SAME_PLANE_SAME_NORMAL
        if self.space is None or frame.space is None:
            return EmbeddedPlaneRelation.UNKNOWN
        if frame.space is not self.space:
            return EmbeddedPlaneRelation.DIFFERENT_PLANES
        return self.compare_embedded_plane_in_own_space(frame=frame)

    @abstractmethod
    def compare_embedded_plane_in_own_space(
        self, *, frame: CartesianFrame3Dimensional
    ) -> EmbeddedPlaneRelation:
        """
        :param frame: Another Cartesian frame of the space this frame charts.
        :return: How the embedded plane of ``frame`` lies relative to that of this frame.
        """


class CartesianFrame3DimensionalResolver(Interface):
    """Finds the Cartesian frames serialized coordinate representations refer
    to, while one JSON document is deserialized.

    It travels through nested ``from_json`` calls in their keyword
    arguments.
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


@dataclass
class CartesianFrame3DimensionalReference(ObjectReference):
    """
    How a serialized coordinate representation points at a
    :class:`CartesianFrame3Dimensional`.
    """

    def resolve(self, *, data: dict[str, Any], **kwargs) -> CartesianFrame3Dimensional:
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
