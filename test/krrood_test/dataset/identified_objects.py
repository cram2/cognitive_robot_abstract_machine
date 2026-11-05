from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from typing_extensions import Any

from krrood.adapters.deserialized_object_tracker import DeserializedObjectTracker
from krrood.adapters.json_serializer import from_json
from krrood.adapters.object_reference import ObjectReference
from krrood.patterns.identifier_and_name import HasID, HasName

# %% objects referred to by identifier


@dataclass(kw_only=True)
class Landmark(HasID, HasName[str]):
    """
    A named object that documents refer to instead of containing it.
    """

    name: str
    """
    What the landmark is called.
    """

    id: UUID = field(default_factory=uuid4)
    """
    The identifier documents refer to the landmark with.
    """


@dataclass(kw_only=True)
class UnnamedMarker(HasID):
    """
    An object that documents refer to by its identifier alone, having no name.
    """

    id: UUID = field(default_factory=uuid4)
    """
    The identifier documents refer to the marker with.
    """


@dataclass(kw_only=True)
class LandmarkTracker(DeserializedObjectTracker[UUID, Landmark]):
    """
    The landmarks the reader of a document already has.
    """


@dataclass(kw_only=True)
class LandmarkReference(ObjectReference):
    """
    How a document points at a :class:`Landmark` or an :class:`UnnamedMarker`.
    """

    def resolve(self, *, data: dict[str, Any], **kwargs) -> Landmark:
        return LandmarkTracker.from_kwargs(from_json_kwargs=kwargs).get(key=from_json(data[self.id_key]))
