from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from typing_extensions import Any, Dict, Optional, TYPE_CHECKING, Self

from krrood.adapters.deserialized_object_tracker import DeserializedObjectTracker
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.exceptions import (
    MissingWorldError,
    WorldEntityWithIDNotInKwargs,
)

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import WorldEntityWithID


@dataclass
class WorldEntityReference:
    """
    How a serialized object points at a world entity: by the id that identifies it, and
    by the name it went by, which says which entity was meant where it cannot be found.

    The name is never looked up with, so two entities sharing one name stay apart.
    """

    subject: str
    """
    What the referring object calls the entity, for example ``parent`` or ``dof``.
    """

    @property
    def id_key(self) -> str:
        """
        Where the id of the entity sits in the serialized object.
        """
        return f"{self.subject}_id"

    @property
    def name_key(self) -> str:
        """
        Where the name of the entity sits in the serialized object.
        """
        return f"{self.subject}_name"

    def write(self, data: Dict[str, Any], entity: WorldEntityWithID) -> None:
        """
        Put the reference to an entity into a serialized object.

        :param data: The json of the object that refers to it.
        :param entity: The entity it refers to.
        """
        data[self.id_key] = to_json(entity.id)
        data[self.name_key] = to_json(entity.name)

    def resolve(self, data: Dict[str, Any], **kwargs) -> WorldEntityWithID:
        """
        The entity a serialized object refers to.

        :param data: The json of the object that refers to it.
        :param kwargs: The kwargs of the ``_from_json`` that is reading it.
        :raises WorldEntityWithIDNotInKwargs: If nothing known carries that id.
        """
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        return tracker.get(
            from_json(data[self.id_key]), name=from_json(data.get(self.name_key))
        )


@dataclass
class WorldEntityWithIDKwargsTracker(
    # A string, because world_entity imports this module and cannot be imported back.
    DeserializedObjectTracker[UUID, "WorldEntityWithID"]
):
    """
    The world entities deserialized from one JSON document, by their id.

    An entity the document does not contain is looked up in the world the tracker was
    created with through :meth:`from_world`.
    """

    _world: Optional[World] = field(init=False, default=None)
    """
    The world to look up the entities in that were not deserialized from the document.
    """

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Create a new tracker from a world.

        :param world: A world instance that will be used as a backup to look for world
            entities.
        """
        if world is None:
            raise MissingWorldError()
        tracker = cls()
        tracker._world = world
        return tracker

    def get(self, key: UUID, name: Optional[PrefixedName] = None) -> WorldEntityWithID:
        """
        :param name: The name the reference to the entity went by, which says which
            entity was meant when it cannot be found.
        """
        if key in self.tracked_objects:
            return self.tracked_objects[key]
        return self._get_untracked(key, name)

    def _has_untracked(self, key: UUID) -> bool:
        if self._world is None:
            return False
        return self._world.find_world_entity_with_id(key) is not None

    def _get_untracked(
        self, key: UUID, name: Optional[PrefixedName] = None
    ) -> WorldEntityWithID:
        """
        :param name: The name the reference to the entity went by, which says which
            entity was meant when it cannot be found.
        :raises MissingWorldError: If the tracker has no world to look the entity up in.
        :raises WorldEntityWithIDNotInKwargs: If the world holds no entity with the id.
        """
        if self._world is None:
            raise MissingWorldError()
        entity = self._world.find_world_entity_with_id(key)
        if entity is None:
            raise WorldEntityWithIDNotInKwargs(key=key, world_entity_name=name)
        return entity
