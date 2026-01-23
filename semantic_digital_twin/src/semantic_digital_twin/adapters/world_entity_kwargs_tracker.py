from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from typing_extensions import Dict, Optional, TYPE_CHECKING, Self, ClassVar, Any

from ..exceptions import (
    WorldEntityWithIDNotInKwargs,
    WorldEntityNotFoundError,
    WorldEntityWithIDNotFoundError,
)

if TYPE_CHECKING:
    from ..world import World
    from ..world_description.world_entity import WorldEntityWithID


@dataclass
class WorldEntityWithIDKwargsTracker:
    """
    Keeps track of the world entities that have been parsed in a from_json call from SubclassJSONSerializer.
    Usage:
        Top-level object must create a new tracker, optionally using a world instance if present, and pass it along:
            tracker = WorldEntityWithIDKwargsTracker.from_world(world)
            SubclassJSONSerializer.from_json(json_data, **tracker.create_from_json_kwargs())

        Objects that create world entities:
            def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
                new_instance = cls(...)
                tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
                tracker.add_world_entity_with_id(new_instance)
                ...

        Objects that need world entities for parsing:
            def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
                tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
                entity = tracker.get_world_entity_with_id(name_of_entity)
                ...
    """

    _world_entities_with_id: Dict[UUID, WorldEntityWithID] = field(default_factory=dict)
    _world: Optional[World] = field(init=False, default=None)
    __world_entity_tracker: ClassVar[str] = "__world_entity_tracker"

    @classmethod
    def from_kwargs(cls, from_json_kwargs) -> Self:
        """
        Retrieve the tracker from the kwargs, or initialize a new one if it doesn't exist.
        Adds itself to the kwargs so that it is available for future from_json calls.
        :param from_json_kwargs: the **kwargs of a from_json call.
        """
        tracker = from_json_kwargs.get(cls.__world_entity_tracker) or cls()
        tracker.add_to_kwargs(from_json_kwargs)
        return tracker

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Create a new tracker from a world.
        :param world: A world instance that will be used as a backup to look for world entities.
        """
        tracker = cls()
        tracker._world = world
        return tracker

    def create_kwargs(self) -> Dict[str, Self]:
        """
        Creates a new kwargs that contains the tracker.
        The top-level object that calls from_json should add this to its kwargs.
        :return: A new kwargs dict with the tracker.
        """
        return {self.__world_entity_tracker: self}

    def add_to_kwargs(self, kwargs: Dict[str, Any]):
        """
        Adds the current instance to the provided keyword arguments dictionary,
        using a specific key internally defined within the instance.

        :param kwargs: A dictionary to which the current instance will be added.
                       The specific key is determined by the internal attribute of
                       the instance.
        :return: None
        """
        kwargs[self.__world_entity_tracker] = self

    def add_world_entity_with_id(self, world_entity_with_id: WorldEntityWithID):
        """
        Add a new world entity with id to the tracker in-place, to make it available for parsing in future from_json calls.
        """
        self._world_entities_with_id[world_entity_with_id.id] = world_entity_with_id

    def has_world_entity_with_id(self, id: UUID) -> bool:
        return id in self._world_entities_with_id

    def get_world_entity_with_id(self, id: UUID) -> WorldEntityWithID:
        """
        Retrieve a world entity by its UUID.

        This method attempts to find a world entity from the internal
        collection. If the entity is not found and a world object is available,
        it will try to retrieve the entity by its UUID from the world object.

        :param id: The UUID of the world entity to retrieve.
        :return: The world entity corresponding to the specified UUID,
                 or None if not found.
        """
        result = self._world_entities_with_id.get(id)
        if result is not None:
            return result
        if self._world is not None:
            try:
                return self._world.get_world_entity_with_id_by_id(id)
            except WorldEntityWithIDNotFoundError:
                pass
        raise WorldEntityWithIDNotInKwargs(id)
