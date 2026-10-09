"""
The specification base classes every specification derives from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    WorldEntity,
)

# %% specification type parameters
TWorldEntity = TypeVar("TWorldEntity", bound=WorldEntity)


# %% specification base classes


@dataclass
class NamedSpecification(ABC):
    """
    Base for every specification: it carries a name and normalizes it.

    It deliberately declares no materialization contract, so entity-spawn specs and
    connection specs can derive their own (incompatible) verbs from it without one
    masquerading as the other.
    """

    name: str | None
    """
    The name of entities created from this specification, as a plain string.

    ``None`` defers naming to materialization.
    """

    def _resolved_name(self, name: str | None = None) -> PrefixedName | None:
        """
        Normalize the spawn-time name override, or the spec's own name, into a
        :class:`PrefixedName`.

        A bare string is wrapped into a :class:`PrefixedName`. ``None`` is preserved so
        materialization can fall back to default name generation.

        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :return: The normalized name, or None when neither name is set.
        """
        used_name = name or self.name
        if used_name is None:
            return None
        return PrefixedName(name=used_name)


@dataclass
class SpawnSpecification(NamedSpecification, Generic[TWorldEntity], ABC):
    """
    Specification for a world entity that materializes itself together with the
    connection that attaches it to its parent.

    Materialized via :meth:`spawn`.
    """

    @abstractmethod
    def spawn(
        self,
        world: World,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TWorldEntity:
        """
        Instantiate the World Entity and add it to the given world.

        :param world: The world the entity and its parent connection are added to.
        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the specification's stored default pose. If
            None, the stored default is used.
        :return: The materialized world entity.
        """
