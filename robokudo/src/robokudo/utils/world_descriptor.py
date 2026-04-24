"""World descriptor utilities for Robokudo."""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

import robokudo.utils.module_loader as module_loader

if TYPE_CHECKING:
    from robokudo.annotators.core import BaseAnnotator
    from robokudo.world_descriptor import BaseWorldDescriptor


def load_world_descriptor(annotator: BaseAnnotator) -> BaseWorldDescriptor:
    """Load world descriptor from annotator parameters."""
    loader = module_loader.ModuleLoader()
    return loader.load_world_descriptor(
        annotator.descriptor.parameters.world_descriptor_ros_package,
        annotator.descriptor.parameters.world_descriptor_name,
    )
