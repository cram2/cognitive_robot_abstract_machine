from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, Generic, Self, TypeVar

from krrood.adapters.exceptions import UntrackedObjectError
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_full_class_name

KeyType = TypeVar("KeyType")
TrackedType = TypeVar("TrackedType")


@dataclass
class DeserializedObjectTracker(Generic[KeyType, TrackedType], SubClassSafeGeneric):
    """
    The objects created while deserializing one JSON document, by the key the document
    refers to them with.

    A document holds an object once for every place that refers to it; resolving each of
    these through one tracker turns them back into a single instance. The tracker
    travels through nested ``from_json`` calls in their keyword arguments: the top-level
    caller passes :meth:`create_kwargs`, and every ``_from_json`` that creates or needs
    a tracked object retrieves the tracker with :meth:`from_kwargs`.
    """

    tracked_objects: Dict[KeyType, TrackedType] = field(default_factory=dict)
    """
    The objects deserialized so far, by their key.
    """

    @classmethod
    def from_kwargs(cls, from_json_kwargs: Dict[str, Any]) -> Self:
        """
        Retrieves the tracker from the keyword arguments of a ``from_json`` call, and
        adds a new one to them if there is none, so that calls receiving them share it.

        :param from_json_kwargs: The keyword arguments of a ``from_json`` call.
        :return: The tracker of the document being deserialized.
        """
        keyword = cls._keyword()
        if keyword not in from_json_kwargs:
            from_json_kwargs[keyword] = cls()
        return from_json_kwargs[keyword]

    @classmethod
    def _keyword(cls) -> str:
        """
        :return: The keyword argument this type of tracker is passed as, distinct for every
            tracker type.
        """
        return get_full_class_name(cls)

    def create_kwargs(self) -> Dict[str, Self]:
        """
        :return: Keyword arguments carrying this tracker, to pass to the top-level
            ``from_json`` call.
        """
        return {self._keyword(): self}

    def add(self, key: KeyType, tracked_object: TrackedType) -> None:
        """
        Makes an object available to every later reference to its key.

        :param key: The key the document refers to the object with.
        :param tracked_object: The deserialized object.
        """
        self.tracked_objects[key] = tracked_object

    def has(self, key: KeyType) -> bool:
        """
        :param key: The key the document refers to an object with.
        :return: Whether :meth:`get` finds an object for the key.
        """
        return key in self.tracked_objects or self._has_untracked(key)

    def get(self, key: KeyType) -> TrackedType:
        """
        :param key: The key the document refers to an object with.
        :return: The object added for the key, or else the one :meth:`_get_untracked` finds.
        """
        if key in self.tracked_objects:
            return self.tracked_objects[key]
        return self._get_untracked(key)

    def _has_untracked(self, key: KeyType) -> bool:
        """
        :param key: A key no object was added for.
        :return: Whether :meth:`_get_untracked` finds an object for the key.
        """
        return False

    def _get_untracked(self, key: KeyType) -> TrackedType:
        """
        Finds an object that was not deserialized from the document, for trackers that
        know another source of objects.

        :param key: A key no object was added for.
        :return: The object for the key.
        :raises UntrackedObjectError: If there is no such object.
        """
        raise UntrackedObjectError(key=key)
