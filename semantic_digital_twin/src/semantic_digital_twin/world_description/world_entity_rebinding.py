"""
Rebinding objects built against one world onto the world entities of another.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass, field, fields, is_dataclass

from typing_extensions import Any, Dict, TYPE_CHECKING, TypeVar

from krrood.adapters.json_serializer import list_like_classes
from semantic_digital_twin.exceptions import (
    WorldEntityWithIDBelongsToAnotherWorld,
    WorldEntityWithIDNotFoundError,
)
from semantic_digital_twin.world_description.world_entity import WorldEntityWithID

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

RelocatableType = TypeVar("RelocatableType")


@dataclass
class WorldEntityRebinding:
    """
    One rebinding of objects onto the world entities of :attr:`world`.

    Remembers what each object it reached was rebound into, so every object is rebound
    once however often, and through however many cycles, it is reached.
    """

    world: World
    """
    The world whose own entities the rebound objects refer to.
    """

    rebound: Dict[int, Any] = field(default_factory=dict)
    """
    What each object already reached was rebound into, by the id of that object.

    Doubles as the memo of the deep copies made on the way, so an object first reached
    by one of them is not copied a second time either.
    """

    def rebind(self, obj: RelocatableType) -> RelocatableType:
        """
        :param obj: The object to rebind, or a value containing world entities.
        :return: The equivalent of `obj` referring to :attr:`world`, as described in
            :meth:`World.rebind_world_entities`.
        """
        if isinstance(obj, WorldEntityWithID):
            return self._rebound_entity(obj)
        if id(obj) in self.rebound:
            return self.rebound[id(obj)]
        if self._is_walked_dataclass(obj):
            return self._rebound_dataclass(obj)
        if isinstance(obj, list_like_classes):
            rebound = type(obj)(self.rebind(item) for item in obj)
        elif isinstance(obj, dict):
            rebound = {key: self.rebind(value) for key, value in obj.items()}
        else:
            return deepcopy(obj, self.rebound)
        self.rebound[id(obj)] = rebound
        return rebound

    def _rebound_entity(self, entity: WorldEntityWithID) -> WorldEntityWithID:
        """
        :param entity: The entity to look up.
        :return: :attr:`world`'s own instance of `entity`, or `entity` itself if
            :attr:`world` does not contain it.
        :raises WorldEntityWithIDBelongsToAnotherWorld: If :attr:`world`'s lookup
            answers with an entity that reports belonging elsewhere.
        """
        try:
            found = self.world.get_world_entity_with_id_by_id(entity.id)
        except WorldEntityWithIDNotFoundError:
            return entity
        if found._world is not self.world:
            raise WorldEntityWithIDBelongsToAnotherWorld(
                world=self.world, world_entity=found
            )
        return found

    @staticmethod
    def _is_walked_dataclass(obj: Any) -> bool:
        """
        :param obj: The object to rebind.
        :return: Whether `obj` is a dataclass instance whose fields are rebound one by
            one, rather than one whose type says itself how it is deep-copied.
        """
        return (
            is_dataclass(obj)
            and not isinstance(obj, type)
            and not hasattr(type(obj), "__deepcopy__")
        )

    def _rebound_dataclass(self, obj: Any) -> Any:
        """
        :param obj: The dataclass instance to rebind.
        :return: A copy of `obj` with every field rebound, registered before its fields
            are rebound so a field referring back to it finds it. Attributes that are
            not fields are deep-copied.
        """
        result = copy(obj)
        self.rebound[id(obj)] = result
        field_names = set()
        for dataclass_field in fields(obj):
            field_names.add(dataclass_field.name)
            setattr(
                result,
                dataclass_field.name,
                self.rebind(getattr(obj, dataclass_field.name)),
            )
        if hasattr(obj, "__dict__"):
            for name, value in vars(obj).items():
                if name not in field_names:
                    setattr(result, name, deepcopy(value, self.rebound))
        return result
