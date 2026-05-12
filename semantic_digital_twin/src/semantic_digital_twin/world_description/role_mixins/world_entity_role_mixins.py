from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from krrood.utils import memoize
from semantic_digital_twin.role_mixins.mixin_role_mixins import (
    DelegatorForHasSimulatorProperties,
    RoleForHasSimulatorProperties,
)
from typing import Any, Dict, List, Optional, Self, Set, Type
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from krrood.adapters.json_serializer import JSONAttributeDiff
    from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
        WorldEntityWithIDKwargsTracker,
    )
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
    )
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.shape_collection import (
        BoundingBoxCollection,
    )
    from semantic_digital_twin.world_description.world_entity import (
        Body,
        GenericKinematicStructureEntity,
        KinematicStructureEntity,
        Region,
        SemanticAnnotation,
        WorldEntity,
        WorldEntityWithID,
        WorldEntityWithSimulatorProperties,
    )
    from uuid import UUID


@dataclass(eq=False)
class DelegatorForWorldEntity(ABC):
    @property
    @abstractmethod
    def delegatee(self) -> WorldEntity: ...
    @property
    def _world(self) -> Optional[World]:
        return self.delegatee._world

    @_world.setter
    def _world(self, value: Optional[World]):
        self.delegatee._world = value

    @property
    def name(self) -> PrefixedName:
        return self.delegatee.name

    @name.setter
    def name(self, value: PrefixedName):
        self.delegatee.name = value

    def __eq__(self, other):
        return self.delegatee.__eq__(other)

    def add_to_world(self, world: World):
        return self.delegatee.add_to_world(world)

    def remove_from_world(self):
        return self.delegatee.remove_from_world()


@dataclass(eq=False)
class DelegatorForWorldEntityWithID(DelegatorForWorldEntity, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> WorldEntityWithID: ...
    @property
    def id(self) -> UUID:
        return self.delegatee.id

    @id.setter
    def id(self, value: UUID):
        self.delegatee.id = value

    def _apply_diff(self, diff: JSONAttributeDiff, **kwargs) -> None:
        return self.delegatee._apply_diff(diff, kwargs)

    def _item_from_json(self, data: Dict[str, Any], **kwargs) -> Any:
        return self.delegatee._item_from_json(data, kwargs)

    def _item_to_json(self, item: Any):
        return self.delegatee._item_to_json(item)

    def _track_object_in_from_json(
        self, from_json_kwargs
    ) -> WorldEntityWithIDKwargsTracker:
        return self.delegatee._track_object_in_from_json(from_json_kwargs)

    def copy_for_world(self, world: World) -> Self:
        return self.delegatee.copy_for_world(world)

    def to_json(self) -> Dict[str, Any]:
        return self.delegatee.to_json()

    def update_from_json_diff(self, diffs: List[JSONAttributeDiff], **kwargs) -> None:
        return self.delegatee.update_from_json_diff(diffs, kwargs)


@dataclass(eq=False)
class DelegatorForWorldEntityWithSimulatorProperties(
    DelegatorForWorldEntityWithID, DelegatorForHasSimulatorProperties, ABC
):
    @property
    @abstractmethod
    def delegatee(self) -> WorldEntityWithSimulatorProperties: ...


@dataclass(eq=False)
class DelegatorForSemanticAnnotation(
    DelegatorForWorldEntityWithSimulatorProperties, ABC
):
    @property
    @abstractmethod
    def delegatee(self) -> SemanticAnnotation: ...
    @property
    def bodies(self) -> list[Body]:
        return self.delegatee.bodies

    @property
    def kinematic_structure_entities(self) -> list[KinematicStructureEntity]:
        return self.delegatee.kinematic_structure_entities

    @property
    def regions(self) -> list[Region]:
        return self.delegatee.regions

    def _kinematic_structure_entities(
        self, aggregation_type: Type[GenericKinematicStructureEntity]
    ) -> list[GenericKinematicStructureEntity]:
        return self.delegatee._kinematic_structure_entities(aggregation_type)

    def as_bounding_box_collection_at_origin(
        self, origin: HomogeneousTransformationMatrix
    ) -> BoundingBoxCollection:
        return self.delegatee.as_bounding_box_collection_at_origin(origin)

    def as_bounding_box_collection_in_frame(
        self, reference_frame: KinematicStructureEntity
    ) -> BoundingBoxCollection:
        return self.delegatee.as_bounding_box_collection_in_frame(reference_frame)

    @memoize
    def class_name_tokens(self) -> Set[str]:
        return self.delegatee.class_name_tokens()


@dataclass(eq=False)
class RoleForWorldEntity(DelegatorForWorldEntity, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> WorldEntity: ...


@dataclass(eq=False)
class RoleForWorldEntityWithID(DelegatorForWorldEntityWithID, RoleForWorldEntity, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> WorldEntityWithID: ...
    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        delegatee_type = cls.get_delegatee_type()
        role_taker = delegatee_type._from_json(data, kwargs)
        delegatee_attr = cls.delegatee_attribute_name()
        return cls(**{delegatee_attr: role_taker})

    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        delegatee_type = cls.get_delegatee_type()
        role_taker = delegatee_type.from_json(data, kwargs)
        delegatee_attr = cls.delegatee_attribute_name()
        return cls(**{delegatee_attr: role_taker})


@dataclass(eq=False)
class RoleForWorldEntityWithSimulatorProperties(
    DelegatorForWorldEntityWithSimulatorProperties,
    RoleForWorldEntityWithID,
    RoleForHasSimulatorProperties,
    ABC,
):
    @property
    @abstractmethod
    def delegatee(self) -> WorldEntityWithSimulatorProperties: ...
    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        delegatee_type = cls.get_delegatee_type()
        role_taker = delegatee_type.from_json(data, kwargs)
        delegatee_attr = cls.delegatee_attribute_name()
        return cls(**{delegatee_attr: role_taker})


@dataclass(eq=False)
class RoleForSemanticAnnotation(
    DelegatorForSemanticAnnotation, RoleForWorldEntityWithSimulatorProperties, ABC
):
    @property
    @abstractmethod
    def delegatee(self) -> SemanticAnnotation: ...
    @classmethod
    def from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        delegatee_type = cls.get_delegatee_type()
        role_taker = delegatee_type.from_json(data, kwargs)
        delegatee_attr = cls.delegatee_attribute_name()
        return cls(**{delegatee_attr: role_taker})
