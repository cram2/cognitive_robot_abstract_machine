"""
Cartesian frames, and the spaces they chart, that tests build values on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from krrood.adapters.deserialized_object_tracker import DeserializedObjectTracker
from krrood.patterns.identifier_and_name import HasID, HasName
from krrood_spatial_math.cartesian_frame_3_dimensional import (
    CartesianFrame3Dimensional,
    CartesianFrame3DimensionalResolver,
    EmbeddedPlaneRelation,
)
from krrood_spatial_math.coordinate_system import GenericCoordinateRepresentation
from krrood_spatial_math.exceptions import ReferenceFrameMismatchError
from krrood_spatial_math.space import EuclideanSpace3Dimensional

# %% a space frames chart


@dataclass(eq=False)
class Room(EuclideanSpace3Dimensional):
    """
    A three-dimensional Euclidean space, told apart from other rooms only by identity.
    """


# %% frames told apart by name


@dataclass(eq=False)
class NamedFrame(CartesianFrame3Dimensional[str]):
    """
    A Cartesian frame that only has a name, so it knows no change to any other frame and
    expresses only values already given in it.
    """

    name: str
    """
    What the frame is called.
    """

    space: EuclideanSpace3Dimensional | None = None
    """
    The space the frame charts.
    """

    id: UUID = field(default_factory=uuid4)
    """
    The identifier serialized coordinate representations refer to the frame with.
    """

    def transform_into_own_frame(
        self, *, value: GenericCoordinateRepresentation
    ) -> GenericCoordinateRepresentation:
        if value.reference_frame is not self:
            raise ReferenceFrameMismatchError(
                expected_frame=self, actual_frame=value.reference_frame
            )
        return value

    def compare_embedded_plane_in_own_space(
        self, *, frame: CartesianFrame3Dimensional
    ) -> EmbeddedPlaneRelation:
        return EmbeddedPlaneRelation.UNKNOWN


# %% finding frames while deserializing


@dataclass
class KnownFrames(
    DeserializedObjectTracker[UUID, NamedFrame], CartesianFrame3DimensionalResolver
):
    """
    The frames the reader of a document already has.
    """


# %% frames that extend objects named before they are identified


@dataclass(eq=False)
class NamedThing(HasName[str]):
    """
    An object that is named whether or not it has an identifier.
    """

    name: str
    """
    What the object is called.
    """


@dataclass(eq=False)
class IdentifiedNamedThing(NamedThing, HasID):
    """
    A named object that also has an identifier, given after its name.
    """

    id: UUID = field(default_factory=uuid4)
    """
    The identifier given after the name.
    """


@dataclass(eq=False)
class ThingFrame(IdentifiedNamedThing, CartesianFrame3Dimensional[str]):
    """
    A Cartesian frame that extends an object named before it is identified.
    """

    space: EuclideanSpace3Dimensional | None = None
    """
    The space the frame charts.
    """
