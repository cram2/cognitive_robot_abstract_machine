"""Bootstrap static world descriptor content into the shared world."""

from __future__ import annotations

from timeit import default_timer

from py_trees.common import Status

import robokudo.world as rk_world
from robokudo.annotators.core import BaseAnnotator
from robokudo.utils.module_loader import ModuleLoader
from robokudo.world_descriptor import BaseWorldDescriptor


class WorldDescriptorBootstrapError(RuntimeError):
    """Raised when world descriptor bootstrap fails."""


class WorldDescriptorBootstrapAnnotator(BaseAnnotator):
    """Augment the current shared world with descriptor-defined entities."""

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for world descriptor bootstrap."""

        class Parameters:
            """Parameters for world descriptor bootstrap."""

            def __init__(self) -> None:
                self.world_descriptor_ros_package: str = "robokudo"
                self.world_descriptor_name: str = "world_iai_kitchen20"

        parameters = Parameters()

    def __init__(
        self,
        name: str = "WorldDescriptorBootstrap",
        descriptor: "WorldDescriptorBootstrapAnnotator.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the world descriptor bootstrap annotator."""
        super().__init__(name=name, descriptor=descriptor)
        self.module_loader = ModuleLoader()
        self._last_augmented_world_id: int | None = None

    def load_world_descriptor(self) -> BaseWorldDescriptor:
        """Load the configured world descriptor module."""
        ros_package = self.descriptor.parameters.world_descriptor_ros_package
        module_name = self.descriptor.parameters.world_descriptor_name
        try:
            return self.module_loader.load_world_descriptor(
                ros_pkg_name=ros_package,
                module_name=module_name,
            )
        except Exception as error:
            raise WorldDescriptorBootstrapError(
                f"Failed to load world descriptor '{ros_package}.{module_name}'."
            ) from error

    def _update_tracker_if_active(self) -> None:
        """Keep tracker state synchronized after world augmentation."""
        if rk_world.get_world_entity_tracker() is None:
            return
        rk_world.init_world_entity_tracker_from_world(rk_world.world_instance())

    def augment_world(self, world_descriptor: BaseWorldDescriptor) -> None:
        """Merge descriptor entities into the current shared world."""
        runtime_world = rk_world.world_instance()
        runtime_world_id = id(runtime_world)

        if self._last_augmented_world_id == runtime_world_id:
            return

        try:
            runtime_world.merge_world(world_descriptor.world)
        except Exception as error:
            raise WorldDescriptorBootstrapError(
                "Failed to merge world descriptor into the current world."
            ) from error

        self._update_tracker_if_active()
        self._last_augmented_world_id = runtime_world_id

    def update(self) -> Status:
        """Load descriptor world and merge it into the current world."""
        start_timer = default_timer()
        runtime_world_id = id(rk_world.world_instance())
        if self._last_augmented_world_id == runtime_world_id:
            return Status.SUCCESS

        descriptor = self.load_world_descriptor()
        self.augment_world(descriptor)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
