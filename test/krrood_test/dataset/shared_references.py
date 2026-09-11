from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any, Dict, Self

from krrood.adapters.deserialized_object_tracker import DeserializedObjectTracker
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    SubclassJSONSerializer,
)


@dataclass
class KeyedElement(SubclassJSONSerializer):
    """
    An element that every reference within one JSON document resolves to a single
    instance of, by its key.
    """

    key: str
    """
    Identifies the element within a JSON document.
    """

    def to_json(self) -> Dict[str, Any]:
        return DataclassJSONSerializer.to_json(self)

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = KeyedElementTracker.from_kwargs(kwargs)
        element = DataclassJSONSerializer.from_json(data, clazz=cls, **kwargs)
        if tracker.has(element.key):
            return tracker.get(element.key)
        tracker.add(element.key, element)
        return element


@dataclass
class KeyedElementTracker(DeserializedObjectTracker[str, KeyedElement]):
    """
    The keyed elements deserialized from one JSON document.
    """


@dataclass
class OtherKeyedElementTracker(DeserializedObjectTracker[str, KeyedElement]):
    """
    A second tracker type, to tell apart the keyword argument slots of different
    trackers.
    """


@dataclass
class ElementReferencedTwice:
    """
    Refers to one keyed element from two fields.
    """

    first: KeyedElement
    """
    The first reference to the element.
    """

    second: KeyedElement
    """
    The second reference to the element.
    """
