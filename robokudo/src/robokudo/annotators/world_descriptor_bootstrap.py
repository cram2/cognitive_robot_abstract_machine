"""
Bootstrap static world descriptor content into the shared world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from timeit import default_timer
from typing import Any
from uuid import UUID

from py_trees.common import Status

import robokudo.world as rk_world
from robokudo.annotators.core import BaseAnnotator
from robokudo.exceptions import WorldDescriptorBootstrapError
from robokudo.utils.world_descriptor import load_world_descriptor
from robokudo.world_descriptor import BaseWorldDescriptor


@dataclass
class _WorldEntitySnapshot:
    """
    Entity IDs present in the runtime world before a descriptor merge.
    """

    kinematic_structure_entity_ids: set[UUID] = field(default_factory=set)
    connection_object_ids: set[int] = field(default_factory=set)
    degree_of_freedom_ids: set[UUID] = field(default_factory=set)
    semantic_annotation_ids: set[UUID] = field(default_factory=set)


@dataclass
class _MergedDescriptorEntities:
    """
    Entities that were added by the last descriptor merge.
    """

    kinematic_structure_entity_ids: set[UUID] = field(default_factory=set)
    connections: list[Any] = field(default_factory=list)
    degree_of_freedom_ids: set[UUID] = field(default_factory=set)
    semantic_annotation_ids: set[UUID] = field(default_factory=set)

    def clear(self) -> None:
        """
        Forget tracked descriptor entities.
        """
        self.kinematic_structure_entity_ids.clear()
        self.connections.clear()
        self.degree_of_freedom_ids.clear()
        self.semantic_annotation_ids.clear()

    def is_empty(self) -> bool:
        """
        Return whether no descriptor entities are currently tracked.
        """
        return (
            len(self.kinematic_structure_entity_ids) == 0
            and len(self.connections) == 0
            and len(self.degree_of_freedom_ids) == 0
            and len(self.semantic_annotation_ids) == 0
        )


class WorldDescriptorBootstrapAnnotator(BaseAnnotator):
    """
    Augment the current shared world with descriptor-defined entities.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """
        Configuration descriptor for world descriptor bootstrap.
        """

        class Parameters:
            """
            Parameters for world descriptor bootstrap.
            """

            def __init__(self) -> None:
                self.world_descriptor_ros_package: str = "robokudo"
                self.world_descriptor_name: str = "world_iai_kitchen20"
                self.reload_on_update: bool = False

        parameters = Parameters()

    def __init__(
        self,
        name: str = "WorldDescriptorBootstrap",
        descriptor: WorldDescriptorBootstrapAnnotator.Descriptor | None = None,
    ) -> None:
        """
        Initialize the world descriptor bootstrap annotator.
        """
        super().__init__(name=name, descriptor=descriptor)
        self._last_augmented_world_id: int | None = None
        self._merged_descriptor_entities = _MergedDescriptorEntities()

    def _update_tracker_if_active(self) -> None:
        """
        Keep tracker state synchronized after world augmentation.
        """
        if rk_world.get_world_entity_tracker() is None:
            return
        rk_world.init_world_entity_tracker_from_world(rk_world.world_instance())

    @staticmethod
    def _snapshot_world_entities(world: Any) -> _WorldEntitySnapshot:
        """
        Capture all entities present before merging a descriptor world.
        """
        return _WorldEntitySnapshot(
            kinematic_structure_entity_ids={
                entity.id for entity in world.kinematic_structure_entities
            },
            connection_object_ids={id(connection) for connection in world.connections},
            degree_of_freedom_ids={dof.id for dof in world.degrees_of_freedom},
            semantic_annotation_ids={
                annotation.id for annotation in world.semantic_annotations
            },
        )

    @staticmethod
    def _entities_added_since(
        world: Any, before: _WorldEntitySnapshot
    ) -> _MergedDescriptorEntities:
        """
        Return entities that were added after a snapshot was taken.
        """
        return _MergedDescriptorEntities(
            kinematic_structure_entity_ids={
                entity.id
                for entity in world.kinematic_structure_entities
                if entity.id not in before.kinematic_structure_entity_ids
            },
            connections=[
                connection
                for connection in world.connections
                if id(connection) not in before.connection_object_ids
            ],
            degree_of_freedom_ids={
                dof.id
                for dof in world.degrees_of_freedom
                if dof.id not in before.degree_of_freedom_ids
            },
            semantic_annotation_ids={
                annotation.id
                for annotation in world.semantic_annotations
                if annotation.id not in before.semantic_annotation_ids
            },
        )

    def _remove_merged_descriptor_entities(self, runtime_world: Any) -> None:
        """
        Remove only the entities added by the previous descriptor merge.
        """
        if self._merged_descriptor_entities.is_empty():
            return

        # Runtime entities are expected to not be parented under descriptor-owned
        # entities. Such attachments may be invalidated when the descriptor is replaced.
        connection_object_ids = {
            id(connection)
            for connection in self._merged_descriptor_entities.connections
        }

        try:
            with runtime_world.modify_world():
                for annotation in list(runtime_world.semantic_annotations):
                    if (
                        annotation.id
                        in self._merged_descriptor_entities.semantic_annotation_ids
                    ):
                        runtime_world.remove_semantic_annotation(annotation)

                for connection in reversed(list(runtime_world.connections)):
                    if id(connection) in connection_object_ids:
                        runtime_world.remove_connection(connection)

                for dof in list(runtime_world.degrees_of_freedom):
                    if dof.id in self._merged_descriptor_entities.degree_of_freedom_ids:
                        runtime_world.remove_degree_of_freedom(dof)

                for entity in reversed(
                    list(runtime_world.kinematic_structure_entities)
                ):
                    if (
                        entity.id
                        in self._merged_descriptor_entities.kinematic_structure_entity_ids
                    ):
                        runtime_world.remove_kinematic_structure_entity(entity)
        except Exception as error:
            raise WorldDescriptorBootstrapError(
                operation="remove previous world descriptor contribution"
            ) from error

        self._merged_descriptor_entities.clear()

    def augment_world(self, world_descriptor: BaseWorldDescriptor) -> None:
        """
        Merge descriptor entities into the current shared world.
        """
        runtime_world = rk_world.world_instance()
        runtime_world_id = id(runtime_world)
        reload_on_update = self.descriptor.parameters.reload_on_update

        if self._last_augmented_world_id == runtime_world_id and not reload_on_update:
            return
        if self._last_augmented_world_id != runtime_world_id:
            self._merged_descriptor_entities.clear()

        if reload_on_update:
            self._remove_merged_descriptor_entities(runtime_world)

        try:
            before = self._snapshot_world_entities(runtime_world)
            runtime_world.merge_world(world_descriptor.world)
            self._merged_descriptor_entities = self._entities_added_since(
                runtime_world, before
            )
        except Exception as error:
            raise WorldDescriptorBootstrapError(
                operation="merge world descriptor into the current world"
            ) from error

        self._update_tracker_if_active()
        self._last_augmented_world_id = runtime_world_id

    def update(self) -> Status:
        """
        Load descriptor world and merge it into the current world.
        """
        start_timer = default_timer()
        runtime_world_id = id(rk_world.world_instance())
        if (
            self._last_augmented_world_id == runtime_world_id
            and not self.descriptor.parameters.reload_on_update
        ):
            return Status.SUCCESS

        descriptor = load_world_descriptor(self)
        self.augment_world(descriptor)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
