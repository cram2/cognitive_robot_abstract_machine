from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import WorldEntityWithID

# %% unique names


@dataclass
class UniqueEntityNames(ABC):
    """
    Names entities for a format that identifies everything by a name no two things
    share, while entity names need not be unique even within one world.

    Only entities that actually share a name are told apart, by appending their
    identifier; every other entity keeps its :meth:`preferred_name`.

    The name an entity is first given is kept for as long as this object lives, so a
    name never moves to another entity and an entity arriving later never renames the
    ones already named.
    """

    _name_per_entity: dict[UUID, str] = field(init=False, default_factory=dict)
    """
    The name each entity has been given so far.
    """

    _assigned_names: set[str] = field(init=False, default_factory=set)
    """
    Every name handed out so far, kept even after the entity holding it is gone so that
    no later entity can take over a name something may still refer to.
    """

    def assign(self, entity: WorldEntityWithID) -> str:
        """
        :param entity: The entity to name.
        :return: the name of the entity, giving it one if it has none yet.
        """
        if entity.id in self._name_per_entity:
            return self._name_per_entity[entity.id]

        name = self._unused_name_for(entity)
        self._name_per_entity[entity.id] = name
        self._assigned_names.add(name)
        return name

    def _unused_name_for(self, entity: WorldEntityWithID) -> str:
        """
        :param entity: An entity that has not been named yet.
        :return: the name the entity should get.
        """
        name = self.preferred_name(entity)
        if name not in self._assigned_names:
            return name
        return f"{name}_{entity.id.hex}"

    @abstractmethod
    def preferred_name(self, entity: WorldEntityWithID) -> str:
        """
        :param entity: The entity to name.
        :return: the name the entity gets unless another entity already has it.
        """
        raise NotImplementedError
