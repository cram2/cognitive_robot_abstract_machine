from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Self

from krrood.adapters.deserialized_object_tracker import DeserializedObjectTracker
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    ReferenceWriter,
    SubclassJSONSerializer,
    from_json,
    to_json,
)

# %% an element every reader already has


@dataclass
class ReaderOwnedElement(SubclassJSONSerializer):
    """
    An element that whoever reads a document already has, so the document may refer to
    it instead of containing it.
    """

    key: str
    """
    Identifies the element.
    """

    def to_json(self, **kwargs) -> Dict[str, Any]:
        return DataclassJSONSerializer.to_json(self, **kwargs)

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return DataclassJSONSerializer.from_json(data, clazz=cls, **kwargs)


@dataclass
class ReaderOwnedElementTracker(DeserializedObjectTracker[str, ReaderOwnedElement]):
    """
    The elements the reader of a document already has.
    """


@dataclass
class ReaderOwnedElementReference(SubclassJSONSerializer):
    """
    Refers to an element by its key.
    """

    key: str
    """
    The key of the element referred to.
    """

    def to_json(self, **kwargs) -> Dict[str, Any]:
        return {**super().to_json(**kwargs), "key": self.key}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> ReaderOwnedElement:
        return ReaderOwnedElementTracker.from_kwargs(kwargs).get(data["key"])


@dataclass
class ReaderOwnedElementReferenceWriter(ReferenceWriter[ReaderOwnedElement]):
    """
    Writes elements as references to their key.
    """

    def write_reference(self, element: ReaderOwnedElement) -> Dict[str, Any]:
        return ReaderOwnedElementReference(key=element.key).to_json()


# %% the places an element can be held in


@dataclass
class ElementHolder:
    """
    Holds an element in a field of a dataclass without a serialization of its own.
    """

    element: ReaderOwnedElement
    """
    The element held.
    """


@dataclass
class ElementHolderWithOwnSerialization(SubclassJSONSerializer):
    """
    Holds an element in a field it serializes itself.
    """

    element: ReaderOwnedElement
    """
    The element held.
    """

    def to_json(self, **kwargs) -> Dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            "element": to_json(self.element, **kwargs),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(element=from_json(data["element"], **kwargs))


@dataclass
class ElementInEveryPosition:
    """
    Holds one element in every kind of position a serialized document can hold it in.
    """

    direct: ReaderOwnedElement
    """
    The element as the value of a field.
    """

    nested: ElementHolder
    """
    The element one dataclass further down.
    """

    in_own_serialization: ElementHolderWithOwnSerialization
    """
    The element below a class that serializes its fields itself.
    """

    in_list: List[ReaderOwnedElement] = field(default_factory=list)
    """
    The element as an item of a list.
    """

    in_dict: Dict[str, ReaderOwnedElement] = field(default_factory=dict)
    """
    The element as a value of a dict.
    """
