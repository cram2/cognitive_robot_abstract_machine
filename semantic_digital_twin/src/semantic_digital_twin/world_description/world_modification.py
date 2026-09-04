from __future__ import annotations

from functools import wraps

from typing_extensions import (
    List,
    Dict,
    Any,
    Self,
    Optional,
    TYPE_CHECKING,
)

from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    shallow_diff_json,
    JSONAttributeDiff,
    list_like_classes,
    JSONData,
    to_json,
    from_json,
)
from semantic_digital_twin.exceptions import (
    MissingWorldModificationContextError,
    MismatchingIDsInWorldModification,
)
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from uuid import UUID

from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.degree_of_freedom import (
        DegreeOfFreedom,
    )

    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
        Connection,
        Actuator,
        WorldEntityWithID,
        SemanticAnnotation,
    )


@dataclass
class WorldModification(ABC):
    """
    An abstract base class representing a modification to the world which may be
    synchronized.
    """

    @abstractmethod
    def apply(self, world: World):
        """
        Apply this change to the given world.

        :param world: The world to modify.
        """

    @abstractmethod
    def revert(self, world: World):
        """
        Apply the inverse of this change to the given world, restoring it to the state
        it was in before this modification was applied.

        Like :meth:`apply`, reverting a modification is itself recorded in the world's
        modification history, so the history retains a full account of what happened.

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
class WorldModificationWithWorldEntityReference(WorldModification, ABC):
    """
    An abstract base class representing a modification to the world which may be
    synchronized, and which contains a reference to an entity in the world, as those
    cases may sometimes be treated differently (see World.__deepcopy__).
    """

    @abstractmethod
    def update_reference_for_world(self, world: World) -> Self:
        """
        Update the reference to the entity in the world to point to the corresponding
        entity in the given world.

        This is used when applying modifications to a copied world, to ensure that the
        references in the modifications point to the entities in the copied world rather
        than the original world.

        :param world: The world to update the reference for.
        """


@dataclass
class AddKinematicStructureEntityModification(
    WorldModificationWithWorldEntityReference
):
    """
    Addition of a body to the world.
    """

    kinematic_structure_entity: KinematicStructureEntity
    """
    The body that was added.
    """

    original_kinematic_structure_entity_id: Optional[UUID] = field(default=None)
    """
    The ID of the body when this block was created.

    This is used to ensure the KinematicStructureEntity sent is the same as the one that
    was originally added. This should always be the case, but if for some reason its not
    it will be annoying to debug, and this check should be cheap anyways.
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

    def revert(self, world: World):
        world.remove_kinematic_structure_entity(self.kinematic_structure_entity)

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

    kinematic_structure_entity: Optional[KinematicStructureEntity] = field(
        default=None, repr=False
    )
    """
    The body that was removed, kept so this modification can be reverted.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        kinematic_structure_entity = kwargs["kinematic_structure_entity"]
        return cls(kinematic_structure_entity.id, kinematic_structure_entity)

    def apply(self, world: World):
        world.remove_kinematic_structure_entity(
            world.get_kinematic_structure_entity_by_id(self.kinematic_structure_id)
        )

    def revert(self, world: World):
        world.add_kinematic_structure_entity(self.kinematic_structure_entity)


@dataclass
class AddConnectionModification(WorldModificationWithWorldEntityReference):
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

    def revert(self, world: World):
        world.remove_connection(self.connection)

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

    connection: Optional[Connection] = field(default=None, repr=False)
    """
    The connection that was removed, kept so this modification can be reverted.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        connection = kwargs["connection"]
        return cls(connection.parent.id, connection.child.id, connection)

    def apply(self, world: World):
        parent = world.get_kinematic_structure_entity_by_id(self.parent_id)
        child = world.get_kinematic_structure_entity_by_id(self.child_id)
        world.remove_connection(world.get_connection(parent, child))

    def revert(self, world: World):
        world.add_connection(self.connection)


@dataclass
class AddDegreeOfFreedomModification(WorldModificationWithWorldEntityReference):
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

    def revert(self, world: World):
        world.remove_degree_of_freedom(self.degree_of_freedom)

    def update_reference_for_world(self, world: World) -> Self:
        return self.__class__(self.degree_of_freedom.copy_for_world(world))


@dataclass
class RemoveDegreeOfFreedomModification(WorldModification):

    degree_of_freedom_id: UUID

    degree_of_freedom: Optional[DegreeOfFreedom] = field(default=None, repr=False)
    """
    The degree of freedom that was removed, kept so this modification can be reverted.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        dof = kwargs["dof"]
        return cls(degree_of_freedom_id=dof.id, degree_of_freedom=dof)

    def apply(self, world: World):
        world.remove_degree_of_freedom(
            world.get_degree_of_freedom_by_id(self.degree_of_freedom_id)
        )

    def revert(self, world: World):
        world.add_degree_of_freedom(self.degree_of_freedom)


@dataclass
class AddSemanticAnnotationModification(WorldModification, SubclassJSONSerializer):
    semantic_annotation_json: JSONData

    semantic_annotation_id: Optional[UUID] = field(default=None)
    """
    The ID of the semantic annotation that was added, kept so this modification can be
    reverted.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        semantic_annotation = kwargs["semantic_annotation"]
        return cls(
            semantic_annotation_json=to_json(semantic_annotation),
            semantic_annotation_id=semantic_annotation.id,
        )

    @classmethod
    def from_domain_object(cls, domain_object: SemanticAnnotation):
        return cls(
            semantic_annotation_json=to_json(domain_object),
            semantic_annotation_id=domain_object.id,
        )

    def apply(self, world: World):
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        world.add_semantic_annotation(
            from_json(self.semantic_annotation_json, **kwargs)
        )

    def revert(self, world: World):
        world.remove_semantic_annotation(
            world.get_semantic_annotation_by_id(self.semantic_annotation_id)
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "semantic_annotation_json": self.semantic_annotation_json,
            "semantic_annotation_id": to_json(self.semantic_annotation_id),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            semantic_annotation_json=data["semantic_annotation_json"],
            semantic_annotation_id=from_json(data["semantic_annotation_id"], **kwargs),
        )


@dataclass
class RemoveSemanticAnnotationModification(WorldModification, SubclassJSONSerializer):

    semantic_annotation_id: UUID

    semantic_annotation_json: Optional[JSONData] = field(default=None)
    """
    The semantic annotation that was removed, kept so this modification can be reverted.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        semantic_annotation = kwargs["semantic_annotation"]
        return cls(
            semantic_annotation_id=semantic_annotation.id,
            semantic_annotation_json=to_json(semantic_annotation),
        )

    def apply(self, world: World):
        world.remove_semantic_annotation(
            world.get_semantic_annotation_by_id(self.semantic_annotation_id)
        )

    def revert(self, world: World):
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        world.add_semantic_annotation(
            from_json(self.semantic_annotation_json, **kwargs)
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "semantic_annotation_id": to_json(self.semantic_annotation_id),
            "semantic_annotation_json": self.semantic_annotation_json,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            semantic_annotation_id=from_json(data["semantic_annotation_id"], **kwargs),
            semantic_annotation_json=data["semantic_annotation_json"],
        )


@dataclass
class AddActuatorModification(WorldModificationWithWorldEntityReference):
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

    def revert(self, world: World):
        world.remove_actuator(self.actuator)

    def update_reference_for_world(self, world: World) -> Self:
        return self.__class__(self.actuator.copy_for_world(world))


@dataclass
class RemoveActuatorModification(WorldModification):
    actuator_id: UUID

    actuator: Optional[Actuator] = field(default=None, repr=False)
    """
    The actuator that was removed, kept so this modification can be reverted.
    """

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]):
        actuator = kwargs["actuator"]
        return cls(actuator_id=actuator.id, actuator=actuator)

    def apply(self, world: World):
        world.remove_actuator(world.get_actuator_by_id(self.actuator_id))

    def revert(self, world: World):
        world.add_actuator(self.actuator)


@dataclass
class WorldModelModificationBlock:
    """
    A sequence of WorldModelModifications that were applied to the world within one
    `with world.modify_world()` context.
    """

    modifications: List[WorldModification] = field(default_factory=list)
    """
    The list of modifications to apply to the world.
    """

    def apply(self, world: World):
        for modification in self.modifications:
            modification.apply(world)

    def revert(self, world: World):
        """
        Revert all modifications in this block, most recently applied first, restoring
        the world to the state it was in before this block was applied.

        :param world: The world to modify.
        """
        for modification in reversed(self.modifications):
            modification.revert(world)

    def update_references_for_world_and_apply(self, world: World):
        """
        Update the references in the modifications to point to the corresponding
        entities in the given world, and then apply the modifications to the world. This
        is used when applying modifications to a copied world, to ensure that the
        references in the modifications point to the entities in the copied world rather
        than the original world.

        :param world: The world to update the references for.
        """
        for modification in self.modifications:
            if isinstance(modification, WorldModificationWithWorldEntityReference):
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
class DegreeOfFreedomHardwareInterfaceValue:
    """
    The ``has_hardware_interface`` flag one degree of freedom had at some point in time.
    """

    degree_of_freedom_id: UUID
    """
    The UUID of the degree of freedom.
    """

    has_hardware_interface: bool
    """
    Whether the degree of freedom had a hardware interface.
    """


@dataclass
class SetDofHasHardwareInterface(WorldModification):
    degree_of_freedom_ids: List[UUID]
    value: bool

    previous_values: List[DegreeOfFreedomHardwareInterfaceValue] = field(
        default_factory=list
    )
    """
    The ``has_hardware_interface`` value of every affected degree of freedom before this
    modification was applied, kept so this modification can be reverted.
    """

    def apply(self, world: World):
        for dof_id in self.degree_of_freedom_ids:
            world.get_degree_of_freedom_by_id(dof_id).has_hardware_interface = (
                self.value
            )

    def revert(self, world: World):
        dofs_by_previous_value: Dict[bool, List[DegreeOfFreedom]] = {}
        for previous_value in self.previous_values:
            dof = world.get_degree_of_freedom_by_id(previous_value.degree_of_freedom_id)
            dofs_by_previous_value.setdefault(
                previous_value.has_hardware_interface, []
            ).append(dof)
        for value, dofs in dofs_by_previous_value.items():
            world.set_dofs_has_hardware_interface(dofs, value)

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> Self:
        dofs = list(kwargs["dofs"])
        return cls(
            degree_of_freedom_ids=[dof.id for dof in dofs],
            value=kwargs["value"],
            previous_values=[
                DegreeOfFreedomHardwareInterfaceValue(
                    dof.id, dof.has_hardware_interface
                )
                for dof in dofs
            ],
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "degree_of_freedom_ids": [
                to_json(dof_id) for dof_id in self.degree_of_freedom_ids
            ],
            "value": self.value,
            "previous_values": to_json(self.previous_values),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            degree_of_freedom_ids=[
                from_json(_id) for _id in data["degree_of_freedom_ids"]
            ],
            value=data["value"],
            previous_values=from_json(data["previous_values"], **kwargs),
        )


@dataclass
class AttributeUpdateModification(WorldModification, SubclassJSONSerializer):
    """
    An update to one or more attributes of an entity in the world.

    This is used when decorating a method with  @synchronized_attribute_modification
    """

    entity_id: UUID
    """
    The UUID of the entity that was updated.
    """

    updated_kwargs_json_list: List[JSONAttributeDiff]
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
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        entity = world.get_world_entity_with_id_by_id(self.entity_id)
        for diff in self.updated_kwargs_json_list:
            current_value = getattr(entity, diff.attribute_name)
            if isinstance(current_value, list_like_classes):
                self._apply_to_list(current_value, diff, **kwargs)
            else:
                obj = self._resolve_item(
                    world, from_json(diff.added_values[0], **kwargs)
                )
                setattr(entity, diff.attribute_name, obj)
        world._model_manager.current_model_modification_block.append(self)

    def revert(self, world: World):
        tracker = WorldEntityWithIDKwargsTracker.from_world(world)
        kwargs = tracker.create_kwargs()
        entity = world.get_world_entity_with_id_by_id(self.entity_id)
        for diff in self.updated_kwargs_json_list:
            current_value = getattr(entity, diff.attribute_name)
            if isinstance(current_value, list_like_classes):
                inverse_diff = JSONAttributeDiff(
                    attribute_name=diff.attribute_name,
                    added_values=diff.removed_values,
                    removed_values=diff.added_values,
                )
                self._apply_to_list(current_value, inverse_diff, **kwargs)
            else:
                obj = self._resolve_item(
                    world, from_json(diff.removed_values[0], **kwargs)
                )
                setattr(entity, diff.attribute_name, obj)
        world._model_manager.current_model_modification_block.append(self)

    def _apply_to_list(
        self, current_value: List[Any], diff: JSONAttributeDiff, **kwargs
    ):
        world = kwargs["__world_entity_tracker"]._world
        for raw_json in diff.removed_values:
            raw = from_json(raw_json, **kwargs)
            obj = self._resolve_item(world, raw)
            if obj in current_value:
                current_value.remove(obj)

        for raw_json in diff.added_values:
            raw = from_json(raw_json, **kwargs)
            obj = self._resolve_item(world, raw)
            if obj not in current_value:
                current_value.append(obj)

    def _resolve_item(self, world: World, item: Any):
        if isinstance(item, UUID):
            return world.get_world_entity_with_id_by_id(item)
        return item

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "entity_id": to_json(self.entity_id),
            "updated_kwargs_json_list": to_json(self.updated_kwargs_json_list),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            entity_id=from_json(data["entity_id"], **kwargs),
            updated_kwargs_json_list=from_json(
                data["updated_kwargs_json_list"], **kwargs
            ),
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
