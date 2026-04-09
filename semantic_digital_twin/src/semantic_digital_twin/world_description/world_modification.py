from __future__ import annotations

from functools import wraps

from typing_extensions import (
    List,
    TYPE_CHECKING,
)

from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    shallow_diff_json,
    JSONAttributeDiff,
    list_like_classes,
)
from semantic_digital_twin.exceptions import (
    MissingWorldModificationContextError,
    MismatchingIDsInWorldModification,
    WorldEntityWithIDNotFoundError,
)
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityWithID,
    SemanticAnnotation,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Dict, Any, Self, TYPE_CHECKING, Optional
from uuid import UUID

from krrood.adapters.json_serializer import to_json, from_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Connection,
    Actuator,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass
class WorldModification(ABC):
    """
    An abstract base class representing a modification to the world which may be synchronized.
    """

    @abstractmethod
    def apply(self, world: World):
        """
        Apply this change to the given world.

        :param world: The world to modify.
        """

    @classmethod
    @abstractmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> Self:
        """
        Factory to construct this change from the kwargs of its corresponding method in World decorated with
        `atomic_world_modification(modification=cls)`.

        :param kwargs: The kwargs of the function call.
        :return: A new instance.
        """
        raise NotImplementedError


@dataclass
class WorldModificationWithLiveReference(WorldModification, ABC):
    """
    An abstract base class representing a modification to the world which may be synchronized, and which contains a live
     reference to an entity in the world, as those cases may sometimes be treated differently (see World.__deepcopy__).
    """

    @abstractmethod
    def update_reference_for_world(self, world: World) -> Self:
        """
        Update the reference to the entity in the world to point to the corresponding entity in the given world. This is used
        when applying modifications to a copied world, to ensure that the references in the modifications point to the entities
        in the copied world rather than the original world.

        :param world: The world to update the reference for.
        """


@dataclass
class AddKinematicStructureEntityModification(WorldModificationWithLiveReference):
    """
    Addition of a body to the world.
    """

    kinematic_structure_entity: KinematicStructureEntity
    """
    The body that was added.
    """

    original_kinematic_structure_entity_id: Optional[UUID] = field(default=None)
    """
    The ID of the body when this block was created. This is used to ensure the KinematicStructureEntity sent is the same
    as the one that was originally added. This should always be the case, but if for some reason its not it will be 
    annoying to debug, and this check should be cheap anyways.
    """

    def __post_init__(self):
        self.original_kinematic_structure_entity_id = self.kinematic_structure_entity.id

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["kinematic_structure_entity"])

    def apply(self, world: World):
        if (
            not self.original_kinematic_structure_entity_id
            == self.kinematic_structure_entity.id
        ):
            raise MismatchingIDsInWorldModification(
                self.__class__,
                [self.original_kinematic_structure_entity_id],
                [self.kinematic_structure_entity.id],
            )
        world.add_kinematic_structure_entity(self.kinematic_structure_entity)

    def update_reference_for_world(self, world: World) -> Self:
        return self.__class__(self.kinematic_structure_entity.copy_for_world(world))


@dataclass
class RemoveKinematicStructureEntityModification(WorldModification):
    """
    Removal of a body from the world.
    """

    kinematic_structure_id: UUID
    """
    The UUID of the body that was removed.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["kinematic_structure_entity"].id)

    def apply(self, world: World):
        world.remove_kinematic_structure_entity(
            world.get_kinematic_structure_entity_by_id(self.kinematic_structure_id)
        )


@dataclass
class AddConnectionModification(WorldModificationWithLiveReference):
    """
    Addition of a connection to the world.
    """

    connection: Connection
    """
    The connection that was added.
    """

    original_child_id: Optional[UUID] = field(default=None)
    original_parent_id: Optional[UUID] = field(default=None)

    def __post_init__(self):
        self.original_child_id = self.connection.child.id
        self.original_parent_id = self.connection.parent.id

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["connection"])

    def apply(self, world: World):
        if (
            self.connection.parent.id != self.original_parent_id
            or self.connection.child.id != self.original_child_id
        ):
            raise MismatchingIDsInWorldModification(
                self.__class__,
                [self.original_child_id, self.original_parent_id],
                [self.connection.child.id, self.connection.parent.id],
            )
        world.add_connection(self.connection.copy_for_world(world))

    def update_reference_for_world(self, world: World) -> Self:
        return self.__class__(self.connection.copy_for_world(world))


@dataclass
class RemoveConnectionModification(WorldModification):
    """
    Removal of a connection from the world.
    """

    parent_id: UUID
    """
    The UUID of the parent body of the removed connection.
    """

    child_id: UUID
    """
    The UUIDs of the entities connected by the removed connection.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["connection"].parent.id, kwargs["connection"].child.id)

    def apply(self, world: World):
        parent = world.get_kinematic_structure_entity_by_id(self.parent_id)
        child = world.get_kinematic_structure_entity_by_id(self.child_id)
        world.remove_connection(world.get_connection(parent, child))


@dataclass
class AddDegreeOfFreedomModification(WorldModificationWithLiveReference):
    """
    Addition of a degree of freedom to the world.
    """

    degree_of_freedom: DegreeOfFreedom
    """
    The degree of freedom that was added.
    """

    original_degree_of_freedom_id: Optional[UUID] = field(default=None)

    def __post_init__(self):
        self.original_degree_of_freedom_id = self.degree_of_freedom.id

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(degree_of_freedom=kwargs["dof"])

    def apply(self, world: World):
        if not self.original_degree_of_freedom_id == self.degree_of_freedom.id:
            raise MismatchingIDsInWorldModification(
                self.__class__,
                [self.original_degree_of_freedom_id],
                [self.degree_of_freedom.id],
            )
        world.add_degree_of_freedom(self.degree_of_freedom)

    def update_reference_for_world(self, world: World) -> Self:
        return self.__class__(self.degree_of_freedom.copy_for_world(world))


@dataclass
class RemoveDegreeOfFreedomModification(WorldModification):

    dof_id: UUID

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(dof_id=kwargs["dof"].id)

    def apply(self, world: World):
        world.remove_degree_of_freedom(world.get_degree_of_freedom_by_id(self.dof_id))


@dataclass
class AddSemanticAnnotationModification(WorldModification, SubclassJSONSerializer):
    semantic_annotation_json: Dict[str, Any]

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(semantic_annotation_json=to_json(kwargs["semantic_annotation"]))

    @classmethod
    def from_domain_object(cls, domain_object: SemanticAnnotation):
        return cls(semantic_annotation_json=to_json(domain_object))

    def apply(self, world: World):
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        world.add_semantic_annotation(
            from_json(self.semantic_annotation_json, **kwargs)
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "semantic_annotation_json": self.semantic_annotation_json,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(semantic_annotation_json=data["semantic_annotation_json"])


@dataclass
class RemoveSemanticAnnotationModification(WorldModification):

    semantic_annotation_id: UUID

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(semantic_annotation_id=kwargs["semantic_annotation"].id)

    def apply(self, world: World):
        world.remove_semantic_annotation(
            world.get_semantic_annotation_by_id(self.semantic_annotation_id)
        )


@dataclass
class AddActuatorModification(WorldModificationWithLiveReference):
    actuator: Actuator

    original_actuator_id: Optional[UUID] = field(default=None)

    def __post_init__(self):
        self.original_actuator_id = self.actuator.id

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(actuator=kwargs["actuator"])

    def apply(self, world: World):
        if not self.original_actuator_id == self.actuator.id:
            raise MismatchingIDsInWorldModification(
                self.__class__,
                [self.original_actuator_id],
                [self.actuator.id],
            )

        world.add_actuator(self.actuator)

    def update_reference_for_world(self, world: World) -> Self:
        return self.__class__(self.actuator.copy_for_world(world))


@dataclass
class RemoveActuatorModification(WorldModification):
    actuator_id: UUID

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(actuator_id=kwargs["actuator"].id)

    def apply(self, world: World):
        world.remove_actuator(world.get_actuator_by_id(self.actuator_id))


@dataclass
class WorldModelModificationBlock:
    """
    A sequence of WorldModelModifications that were applied to the world within one `with world.modify_world()` context.
    """

    modifications: List[WorldModification] = field(default_factory=list)
    """
    The list of modifications to apply to the world.
    """

    def apply(self, world: World):
        for modification in self.modifications:
            try:
                modification.apply(world)
            except WorldEntityWithIDNotFoundError as e:
                ...

    def update_references_for_world_and_apply(self, world: World):
        """
        Update the references in the modifications to point to the corresponding entities in the given world, and
        then apply the modifications to the world. This is used when applying modifications to a copied world, to
        ensure that the references in the modifications point to the entities in the copied world rather than the original world.

        :param world: The world to update the references for.
        """
        for modification in self.modifications:
            if isinstance(modification, WorldModificationWithLiveReference):
                modification = modification.update_reference_for_world(world)

            modification.apply(world)

    @classmethod
    def apply_from_json(cls, world: World, data: Dict[str, Any], **kwargs) -> Self:
        """
        Apply the modifications in the given JSON data to the given world.
        """
        data = data["modifications"]

        for modification in data:
            from_json(modification, **kwargs).apply(world)

    def __iter__(self):
        return iter(self.modifications)

    def __getitem__(self, item):
        return self.modifications[item]

    def __len__(self):
        return len(self.modifications)

    def append(self, modification: WorldModification):
        self.modifications.append(modification)


@dataclass
class SetDofHasHardwareInterface(WorldModification):
    degree_of_freedom_ids: List[UUID]
    value: bool

    def apply(self, world: World):
        for dof_id in self.degree_of_freedom_ids:
            world.get_degree_of_freedom_by_id(dof_id).has_hardware_interface = (
                self.value
            )

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> Self:
        dofs = kwargs["dofs"]
        degree_of_freedom_ids = [dof.id for dof in dofs]
        return cls(degree_of_freedom_ids=degree_of_freedom_ids, value=kwargs["value"])

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "degree_of_freedom_ids": [
                to_json(dof_id) for dof_id in self.degree_of_freedom_ids
            ],
            "value": self.value,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            degree_of_freedom_ids=[
                from_json(_id) for _id in data["degree_of_freedom_ids"]
            ],
            value=data["value"],
        )


@dataclass
class AttributeUpdateModification(WorldModification):
    """
    An update to one or more attributes of an entity in the world.
    This is used when decorating a method with  @synchronized_attribute_modification
    """

    entity_id: UUID
    """
    The UUID of the entity that was updated.
    """

    updated_kwargs: List[JSONAttributeDiff]
    """
    The list of attribute names and their new values.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(
            from_json(kwargs["entity_id"], **kwargs),
            from_json(kwargs["updated_kwargs"], **kwargs),
        )

    def apply(self, world: World):
        entity = world.get_world_entity_with_id_by_id(self.entity_id)
        for diff in self.update_world_references_in_updated_kwargs(world):
            current_value = getattr(entity, diff.attribute_name)
            if isinstance(current_value, list_like_classes):
                self._apply_to_list(world, current_value, diff)
            else:
                obj = self._resolve_item(world, diff.added_values[0])
                setattr(entity, diff.attribute_name, obj)
        world._model_manager.current_model_modification_block.append(self)

    def update_world_references_in_updated_kwargs(self, world: World):
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        return from_json(to_json(self.updated_kwargs), **kwargs)

    def _apply_to_list(
        self, world: World, current_value: List[Any], diff: JSONAttributeDiff
    ):
        for raw in diff.removed_values:
            obj = self._resolve_item(world, raw)
            if obj in current_value:
                current_value.remove(obj)

        for raw in diff.added_values:
            obj = self._resolve_item(world, raw)
            if obj not in current_value:
                current_value.append(obj)

    def _resolve_item(self, world: World, item: Any):
        if isinstance(item, UUID):
            return world.get_world_entity_with_id_by_id(item)
        return item


def synchronized_attribute_modification(func):
    """
    Decorator to synchronize attribute modifications.

    Ensures that any modifications to the attributes of an instance of WorldEntityWithID are properly recorded and any
    resultant changes are appended to the current model modification block in the world model manager. Keeps track of
    the pre- and post-modification states of the object to compute the differences and maintain a log of updates.

    ..warning::
        This only works for WorldEntityWithID which are also completely JSONSerializable without any many-to-many/one objects
        out side of other WorldEntityWithID
    """

    @wraps(func)
    def wrapper(self: WorldEntityWithID, *args: Any, **kwargs: Any) -> Any:

        object_before_change = to_json(self)
        result = func(self, *args, **kwargs)
        object_after_change = to_json(self)

        tracker = WorldEntityWithIDKwargsTracker.from_world(self._world)
        tracker_kwargs = tracker.create_kwargs()

        diff = shallow_diff_json(
            object_before_change, object_after_change, **tracker_kwargs
        )

        current_model_modification_block = (
            self._world.get_world_model_manager().current_model_modification_block
        )
        if (
            not self._world._model_manager._active_world_model_update_context_manager_ids
        ):
            raise MissingWorldModificationContextError(func)

        current_model_modification_block.append(
            AttributeUpdateModification.from_kwargs(
                {
                    "entity_id": object_after_change["id"],
                    "updated_kwargs": to_json(diff),
                    **tracker_kwargs,
                }
            )
        )
        return result

    return wrapper
