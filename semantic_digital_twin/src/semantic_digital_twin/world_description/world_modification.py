from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from functools import wraps
from uuid import UUID

from typing_extensions import (
    List,
    Dict,
    Any,
    Self,
    TYPE_CHECKING,
)

from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    to_json,
    from_json,
    JSONAttributeDiff,
    list_like_classes,
    shallow_diff_json,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    SemanticAnnotation,
    Connection,
    Actuator,
    WorldEntityWithID,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.exceptions import MissingWorldModificationContextError

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass
class WorldModification(SubclassJSONSerializer, ABC):
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
class WorldModelModification(WorldModification, ABC):
    """
    A record of a modification to the model (structure) of the world.
    This includes add/remove body and add/remove connection.

    All modifications are compared via the names of the objects they reference.

    This class is referenced by the `atomic_world_modification` decorator and should be used for a method that
    applies such a modification to the world.
    """

    object_json: Dict[str, Any] = field(default_factory=dict)
    """
    The JSON representation of the object that was modified. This is used to reconstruct the object when applying the modification to a different world.
    This representations was chosen to freeze the object in the point in time when the modification was applied.
    """

    @classmethod
    def from_domain_object(cls, domain_object: WorldEntityWithID) -> Self:
        """
        Creates an instance of the class from a given domain object.

        :param domain_object: The domain object to create an instance from.

        :return: An instance of the class.
        """
        return cls(object_json=domain_object.to_json())

    @abstractmethod
    def to_domain_object(self, world: World) -> WorldEntityWithID:
        """
        Reconstructs the domain object from the JSON representation of the modification.
        """

    def to_json(self):
        return {**super().to_json(), "object_json": self.object_json}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            object_json=data["object_json"],
        )


@dataclass
class WorldModelModificationViaID(WorldModelModification, ABC):
    """
    A record of a modification to the model (structure) of the world that only stores the ID of the modified object.
    """

    @classmethod
    def from_domain_object(cls, domain_object: WorldEntityWithID) -> Self:
        """
        Creates an instance of the class from a given domain object.

        :param domain_object: The domain object to create an instance from.

        :return: An instance of the class.
        """
        return cls(object_json=to_json(domain_object.id))


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
        for diff in self.updated_kwargs:
            current_value = getattr(entity, diff.attribute_name)
            if isinstance(current_value, list_like_classes):
                self._apply_to_list(world, current_value, diff)
            else:
                obj = self._resolve_item(world, diff.added_values[0])
                setattr(entity, diff.attribute_name, obj)

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

    def to_json(self):
        return {
            **super().to_json(),
            "entity_id": to_json(self.entity_id),
            "updated_kwargs": to_json(self.updated_kwargs),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            entity_id=from_json(data["entity_id"], **kwargs),
            updated_kwargs=from_json(data["updated_kwargs"], **kwargs),
        )


@dataclass
class AddKinematicStructureEntityModification(WorldModelModification):
    """
    Addition of a body to the world.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=kwargs["kinematic_structure_entity"].to_json())

    def to_domain_object(self, world: World) -> KinematicStructureEntity:
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        return KinematicStructureEntity.from_json(self.object_json, **kwargs)

    def apply(self, world: World):
        world.add_kinematic_structure_entity(self.to_domain_object(world))


@dataclass
class RemoveBodyModification(WorldModelModificationViaID):
    """
    Removal of a body from the world.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=to_json(kwargs["kinematic_structure_entity"].id))

    def to_domain_object(self, world: World) -> KinematicStructureEntity:
        return world.get_kinematic_structure_entity_by_id(from_json(self.object_json))

    def apply(self, world: World):
        world.remove_kinematic_structure_entity(self.to_domain_object(world))


@dataclass
class AddConnectionModification(WorldModelModification):
    """
    Addition of a connection to the world.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=kwargs["connection"].to_json())

    def to_domain_object(self, world: World) -> Connection:
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        return Connection.from_json(self.object_json, **kwargs)

    def apply(self, world: World):
        world.add_connection(self.to_domain_object(world))


@dataclass
class RemoveConnectionModification(WorldModelModificationViaID):
    """
    Removal of a connection from the world.
    """

    @classmethod
    def from_domain_object(cls, domain_object: Connection) -> Self:
        parent_id = domain_object.parent.id
        child_id = domain_object.child.id
        object_json = {"parent": to_json(parent_id), "child": to_json(child_id)}
        return cls(object_json=object_json)

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        parent_json = to_json(kwargs["connection"].parent.id)
        child_json = to_json(kwargs["connection"].child.id)
        object_json = {"parent": parent_json, "child": child_json}
        return cls(object_json=object_json)

    def to_domain_object(self, world: World) -> Connection:
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        parent_id = from_json(self.object_json["parent"], **kwargs)
        child_id = from_json(self.object_json["child"], **kwargs)
        parent = world.get_kinematic_structure_entity_by_id(parent_id)
        child = world.get_kinematic_structure_entity_by_id(child_id)
        return world.get_connection(parent, child)

    def apply(self, world: World):
        world.remove_connection(self.to_domain_object(world))

    def to_json(self):
        return {
            **super().to_json(),
            "parent_id": self.object_json["parent"],
            "child_id": self.object_json["child"],
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        parent_id = from_json(data["parent_id"])
        child_id = from_json(data["child_id"])
        return cls(
            object_json={"parent": to_json(parent_id), "child": to_json(child_id)},
        )


@dataclass
class AddDegreeOfFreedomModification(WorldModelModification):
    """
    Addition of a degree of freedom to the world.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=kwargs["dof"].to_json())

    def to_domain_object(self, world: World) -> DegreeOfFreedom:
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        return DegreeOfFreedom.from_json(self.object_json, **kwargs)

    def apply(self, world: World):
        world.add_degree_of_freedom(self.to_domain_object(world))


@dataclass
class RemoveDegreeOfFreedomModification(WorldModelModificationViaID):

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=to_json(kwargs["dof"].id))

    def to_domain_object(self, world: World) -> DegreeOfFreedom:
        return world.get_degree_of_freedom_by_id(from_json(self.object_json))

    def apply(self, world: World):
        world.remove_degree_of_freedom(self.to_domain_object(world))


@dataclass
class AddSemanticAnnotationModification(WorldModelModification):

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=kwargs["semantic_annotation"].to_json())

    def to_domain_object(self, world: World) -> SemanticAnnotation:
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        return SemanticAnnotation.from_json(self.object_json, **kwargs)

    def apply(self, world: World):
        world.add_semantic_annotation(self.to_domain_object(world))


@dataclass
class RemoveSemanticAnnotationModification(WorldModelModificationViaID):

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=to_json(kwargs["semantic_annotation"].id))

    def to_domain_object(self, world: World) -> SemanticAnnotation:
        return world.get_semantic_annotation_by_id(from_json(self.object_json))

    def apply(self, world: World):
        world.remove_semantic_annotation(self.to_domain_object(world))


@dataclass
class AddActuatorModification(WorldModelModification):

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=kwargs["actuator"].to_json())

    def to_domain_object(self, world: World) -> Actuator:
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        return Actuator.from_json(self.object_json, **kwargs)

    def apply(self, world: World):
        world.add_actuator(self.to_domain_object(world))


@dataclass
class RemoveActuatorModification(WorldModelModificationViaID):

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(object_json=to_json(kwargs["actuator"].id))

    def to_domain_object(self, world: World) -> Actuator:
        return world.get_actuator_by_id(from_json(self.object_json))

    def apply(self, world: World):
        world.remove_actuator(self.to_domain_object(world))


@dataclass
class WorldModelModificationBlock(SubclassJSONSerializer):
    """
    A sequence of WorldModelModifications that were applied to the world within one `with world.modify_world()` context.
    """

    modifications: List[WorldModification] = field(default_factory=list)
    """
    The list of modifications to apply to the world.
    """

    def apply(self, world: World):
        for modification in self.modifications:
            modification.apply(world)

    @classmethod
    def apply_from_json(cls, world: World, data: Dict[str, Any], **kwargs) -> Self:
        """
        Apply the modifications in the given JSON data to the given world.
        """
        data = data["modifications"]

        for modification in data:
            WorldModification.from_json(modification, **kwargs).apply(world)

    def to_json(self):
        return {
            **super().to_json(),
            "modifications": to_json(self.modifications),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            modifications=[
                WorldModification.from_json(d, **kwargs) for d in data["modifications"]
            ],
        )

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
