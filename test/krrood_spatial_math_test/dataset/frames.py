from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from typing_extensions import Any

from krrood.adapters.deserialized_object_tracker import DeserializedObjectTracker
from krrood_spatial_math.cartesian_frame_3d import (
    CartesianFrame3D,
    CartesianFrame3DResolver,
    ValueT,
)
from krrood_spatial_math.exceptions import ReferenceFrameMismatchError

# %% frames told apart by name


@dataclass(eq=False)
class NamedFrame(CartesianFrame3D):
    """
    A Cartesian frame that only has a name, so it knows no change to any other frame and
    expresses only values already given in it.
    """

    name: str
    """
    What the frame is called.
    """

    id: UUID = field(default_factory=uuid4)
    """
    The identifier serialized spatial types refer to the frame with.
    """

    def express(self, value: ValueT) -> ValueT:
        if value.reference_frame is not self:
            raise ReferenceFrameMismatchError(
                expected_frame=self, actual_frame=value.reference_frame
            )
        return value


# %% finding frames while deserializing


@dataclass
class KnownFrames(
    DeserializedObjectTracker[UUID, NamedFrame], CartesianFrame3DResolver
):
    """
    The frames the reader of a document already has.
    """

    def get(self, key: UUID, name: Any = None) -> NamedFrame:
        return super().get(key)
