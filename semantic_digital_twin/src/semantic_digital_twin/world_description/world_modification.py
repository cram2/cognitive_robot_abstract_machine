from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from functools import wraps
from typing import Any
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
from .degree_of_freedom import DegreeOfFreedom
from .world_entity import (
    KinematicStructureEntity,
    SemanticAnnotation,
    Connection,
    Actuator,
    WorldEntityWithID,
)
from ..exceptions import MissingWorldModificationContextError

if TYPE_CHECKING:
    from ..world import World


@dataclass
class WorldModelModification(SubclassJSONSerializer, ABC):
    """
    A record of a modification to the model (structure) of the world.
    This includes add/remove body and add/remove connection.

    All modifications are compared via the names of the objects they reference.

    This class is referenced by the `atomic_world_modification` decorator and should be used for a method that
    applies such a modification to the world.
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
class AttributeUpdateModification(WorldModelModification):
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
        return cls(from_json(kwargs["entity_id"]), from_json(kwargs["updated_kwargs"]))

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
            entity_id=from_json(data["entity_id"]),
            updated_kwargs=from_json(data["updated_kwargs"]),
        )


@dataclass
class AddKinematicStructureEntityModification(WorldModelModification):
    """
    Addition of a body to the world.
    """

    kinematic_structure_entity: KinematicStructureEntity
    """
    The body that was added.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["kinematic_structure_entity"])

    def apply(self, world: World):
        world.add_kinematic_structure_entity(self.kinematic_structure_entity)

    def to_json(self):
        return {**super().to_json(), "body": self.kinematic_structure_entity.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            kinematic_structure_entity=KinematicStructureEntity.from_json(
                data["body"], **kwargs
            )
        )

    def __eq__(self, other: Any) -> bool:
        return (
            self.kinematic_structure_entity.name
            == other.kinematic_structure_entity.name
        )


@dataclass
class RemoveBodyModification(WorldModelModification):
    """
    Removal of a body from the world.
    """

    body_id: UUID
    """
    The UUID of the body that was removed.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["kinematic_structure_entity"].id)

    def apply(self, world: World):
        world.remove_kinematic_structure_entity(
            world.get_kinematic_structure_entity_by_id(self.body_id)
        )

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "body_id": to_json(self.body_id)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(body_id=from_json(data["body_id"]))


@dataclass
class AddConnectionModification(WorldModelModification):
    """
    Addition of a connection to the world.
    """

    connection: Connection
    """
    The connection that was added.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(kwargs["connection"])

    def apply(self, world: World):
        world.add_connection(self.connection)

    def to_json(self):
        return {
            **super().to_json(),
            "connection": self.connection.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(connection=Connection.from_json(data["connection"], **kwargs))

    def __eq__(self, other):
        return (
            isinstance(other, AddConnectionModification)
            and self.connection.name == other.connection.name
        )


@dataclass
class RemoveConnectionModification(WorldModelModification):
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
        world._remove_connection(world.get_connection(parent, child))

    def to_json(self):
        return {
            **super().to_json(),
            "parent_id": to_json(self.parent_id),
            "child_id": to_json(self.child_id),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            parent_id=from_json(data["parent_id"]),
            child_id=from_json(data["child_id"]),
        )


@dataclass
class AddDegreeOfFreedomModification(WorldModelModification):
    """
    Addition of a degree of freedom to the world.
    """

    dof: DegreeOfFreedom
    """
    The degree of freedom that was added.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(dof=kwargs["dof"])

    def apply(self, world: World):
        world.add_degree_of_freedom(self.dof)

    def to_json(self):
        return {
            **super().to_json(),
            "dof": self.dof.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(dof=DegreeOfFreedom.from_json(data["dof"], **kwargs))

    def __eq__(self, other):
        return self.dof.id == other.dof.id


@dataclass
class RemoveDegreeOfFreedomModification(WorldModelModification):
    dof_id: UUID

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(dof_id=kwargs["dof"].id)

    def apply(self, world: World):
        world.remove_degree_of_freedom(world.get_degree_of_freedom_by_id(self.dof_id))

    def to_json(self):
        return {
            **super().to_json(),
            "dof": to_json(self.dof_id),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(dof_id=from_json(data["dof"]))


@dataclass
class AddSemanticAnnotationModification(WorldModelModification):
    semantic_annotation: SemanticAnnotation

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(semantic_annotation=kwargs["semantic_annotation"])

    def apply(self, world: World):
        world.add_semantic_annotation(self.semantic_annotation)

    def to_json(self):
        return {
            **super().to_json(),
            "semantic_annotation": self.semantic_annotation.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            semantic_annotation=SemanticAnnotation.from_json(
                data["semantic_annotation"], **kwargs
            )
        )


@dataclass
class RemoveSemanticAnnotationModification(WorldModelModification):
    semantic_annotation: SemanticAnnotation

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(semantic_annotation=kwargs["semantic_annotation"])

    def apply(self, world: World):
        world.remove_semantic_annotation(self.semantic_annotation)

    def to_json(self):
        return {
            **super().to_json(),
            "semantic_annotation": self.semantic_annotation.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            semantic_annotation=SemanticAnnotation.from_json(
                data["semantic_annotation"], **kwargs
            )
        )


@dataclass
class AddActuatorModification(WorldModelModification):
    actuator: Actuator

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(actuator=kwargs["actuator"])

    def apply(self, world: World):
        world.add_actuator(self.actuator)

    def to_json(self):
        return {
            **super().to_json(),
            "actuator": self.actuator.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(actuator=Actuator.from_json(data["actuator"], **kwargs))


@dataclass
class RemoveActuatorModification(WorldModelModification):
    actuator_id: UUID

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        return cls(actuator_id=kwargs["actuator"].id)

    def apply(self, world: World):
        world.remove_actuator(world.get_actuator_by_id(self.actuator_id))

    def to_json(self):
        return {
            **super().to_json(),
            "dof": to_json(self.actuator_id),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(actuator_id=from_json(data["actuator"]))


@dataclass
class WorldModelModificationBlock(SubclassJSONSerializer):
    """
    A sequence of WorldModelModifications that were applied to the world within one `with world.modify_world()` context.
    """

    modifications: List[WorldModelModification] = field(default_factory=list)
    """
    The list of modifications to apply to the world.
    """

    def apply(self, world: World):
        ...
        with world.modify_world():
            for modification in self.modifications:
                modification.apply(world)

    def to_json(self):
        return {
            **super().to_json(),
            "modifications": to_json(self.modifications),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            modifications=[
                WorldModelModification.from_json(d, **kwargs)
                for d in data["modifications"]
            ],
        )

    def __iter__(self):
        return iter(self.modifications)

    def __getitem__(self, item):
        return self.modifications[item]

    def __len__(self):
        return len(self.modifications)

    def append(self, modification: WorldModelModification):
        self.modifications.append(modification)


@dataclass
class SetDofHasHardwareInterface(WorldModelModification):
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
        diff = shallow_diff_json(object_before_change, object_after_change)

        current_model_modification_block = (
            self._world.get_world_model_manager().current_model_modification_block
        )
        if current_model_modification_block is None:
            raise MissingWorldModificationContextError(func)

        current_model_modification_block.append(
            AttributeUpdateModification.from_kwargs(
                {
                    "entity_id": object_after_change["id"],
                    "updated_kwargs": to_json(diff),
                }
            )
        )
        return result

    return wrapper
