from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)
from typing import Any, Dict, List, Self, Type
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from krrood.adapters.json_serializer import JSONAttributeDiff
    from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
        WorldEntityWithIDKwargsTracker,
    )
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.mixin import (
        HasSimulatorProperties,
        SimulatorAdditionalProperty,
    )
    from semantic_digital_twin.semantic_annotations.mixins import (
        HasRootBody,
        HasRootKinematicStructureEntity,
        HasStorageSpace,
        TBody,
        THasRootBody,
        TKinematicStructureEntity,
    )
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
        Point3,
    )
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.geometry import Scale
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
    )
    from uuid import UUID


@dataclass(eq=False)
class RoleForHasSimulatorProperties(ABC):
    @property
    @abstractmethod
    def role_taker(self) -> HasSimulatorProperties: ...
    @property
    def simulator_additional_properties(self) -> list[SimulatorAdditionalProperty]:
        return self.role_taker.simulator_additional_properties

    @simulator_additional_properties.setter
    def simulator_additional_properties(self, value: list[SimulatorAdditionalProperty]):
        self.role_taker.simulator_additional_properties = value


@dataclass(eq=False)
class RoleForWorldEntity(ABC):
    @property
    @abstractmethod
    def role_taker(self) -> WorldEntity: ...
    @property
    def name(self) -> PrefixedName:
        return self.role_taker.name

    @name.setter
    def name(self, value: PrefixedName):
        self.role_taker.name = value

    def remove_from_world(self):
        return self.role_taker.remove_from_world()


@dataclass(eq=False)
class RoleForWorldEntityWithID(RoleForWorldEntity, ABC):
    @property
    @abstractmethod
    def role_taker(self) -> WorldEntityWithID: ...
    @property
    def id(self) -> UUID:
        return self.role_taker.id

    @id.setter
    def id(self, value: UUID):
        self.role_taker.id = value

    def _track_object_in_from_json(
        self, from_json_kwargs
    ) -> WorldEntityWithIDKwargsTracker:
        return self.role_taker._track_object_in_from_json(from_json_kwargs)

    def add_to_world(self, world: World):
        return self.role_taker.add_to_world(world)

    def copy_for_world(self, world: World) -> Self:
        return self.role_taker.copy_for_world(world)

    def to_json(self) -> Dict[str, Any]:
        return self.role_taker.to_json()


@dataclass(eq=False)
class RoleForSemanticAnnotation(RoleForWorldEntityWithID, ABC):
    @property
    @abstractmethod
    def role_taker(self) -> SemanticAnnotation: ...
    @property
    def kinematic_structure_entities(self) -> list[KinematicStructureEntity]:
        return self.role_taker.kinematic_structure_entities

    @property
    def regions(self) -> list[Region]:
        return self.role_taker.regions

    def __eq__(self, other):
        return self.role_taker.__eq__(other)

    def __hash__(self):
        return self.role_taker.__hash__()

    def _kinematic_structure_entities(
        self, aggregation_type: Type[GenericKinematicStructureEntity]
    ) -> list[GenericKinematicStructureEntity]:
        return self.role_taker._kinematic_structure_entities(aggregation_type)

    def as_bounding_box_collection_at_origin(
        self, origin: HomogeneousTransformationMatrix
    ) -> BoundingBoxCollection:
        return self.role_taker.as_bounding_box_collection_at_origin(origin)

    def as_bounding_box_collection_in_frame(
        self, reference_frame: KinematicStructureEntity
    ) -> BoundingBoxCollection:
        return self.role_taker.as_bounding_box_collection_in_frame(reference_frame)


@dataclass(eq=False)
class RoleForHasRootKinematicStructureEntity(RoleForSemanticAnnotation, ABC):
    @property
    @abstractmethod
    def role_taker(self) -> HasRootKinematicStructureEntity: ...
    @property
    def root(self) -> TKinematicStructureEntity:
        return self.role_taker.root

    @root.setter
    def root(self, value: TKinematicStructureEntity):
        self.role_taker.root = value

    @property
    def global_transform(self) -> HomogeneousTransformationMatrix:
        return self.role_taker.global_transform

    @property
    def min_max_points(self) -> tuple[Point3, Point3]:
        return self.role_taker.min_max_points

    @property
    def scale(self) -> Scale:
        return self.role_taker.scale

    def _attach_child_entity_in_kinematic_structure(
        self,
        child_kinematic_structure_entity: KinematicStructureEntity,
    ):
        return self.role_taker._attach_child_entity_in_kinematic_structure(
            child_kinematic_structure_entity
        )

    def _attach_parent_entity_in_kinematic_structure(
        self,
        new_parent_entity: KinematicStructureEntity,
    ):
        return self.role_taker._attach_parent_entity_in_kinematic_structure(
            new_parent_entity
        )

    def _offline_root_T_entity(
        self, entity: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        return self.role_taker._offline_root_T_entity(entity)

    def get_new_grandparent(
        self,
        parent_kinematic_structure_entity: KinematicStructureEntity,
    ):
        return self.role_taker.get_new_grandparent(parent_kinematic_structure_entity)


@dataclass(eq=False)
class RoleForHasRootBody(RoleForHasRootKinematicStructureEntity, ABC):
    @property
    @abstractmethod
    def role_taker(self) -> HasRootBody: ...
    @property
    def root(self) -> TBody:
        return self.role_taker.root

    @root.setter
    def root(self, value: TBody):
        self.role_taker.root = value

    @property
    def bodies(self) -> list[Body]:
        return self.role_taker.bodies


@dataclass(eq=False)
class RoleForHasStorageSpace(RoleForHasRootBody, ABC):
    @property
    @abstractmethod
    def role_taker(self) -> HasStorageSpace: ...
    @property
    def objects(self) -> List[THasRootBody]:
        return self.role_taker.objects

    @objects.setter
    def objects(self, value: List[THasRootBody]):
        self.role_taker.objects = value

    def _apply_diff(self, diff: JSONAttributeDiff, **kwargs) -> None:
        return self.role_taker._apply_diff(diff, kwargs)

    @synchronized_attribute_modification
    def add_object(self, object: HasRootBody):
        return self.role_taker.add_object(object)

    def get_objects_of_type(
        self, object_type: Type[SemanticAnnotation]
    ) -> List[HasRootBody]:
        return self.role_taker.get_objects_of_type(object_type)

    def update_from_json_diff(self, diffs: List[JSONAttributeDiff], **kwargs) -> None:
        return self.role_taker.update_from_json_diff(diffs, kwargs)
