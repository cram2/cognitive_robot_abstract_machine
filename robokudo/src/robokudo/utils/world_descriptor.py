"""
World descriptor utilities for Robokudo.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

import robokudo.utils.module_loader as module_loader
from robokudo.exceptions import WorldDescriptorLoadError

if TYPE_CHECKING:
    from robokudo.annotators.core import BaseAnnotator
    from robokudo.world_descriptor import BaseWorldDescriptor


def load_world_descriptor(annotator: BaseAnnotator) -> BaseWorldDescriptor:
    """
    Load world descriptor from annotator parameters.
    """
    ros_package = annotator.descriptor.parameters.world_descriptor_ros_package
    module_name = annotator.descriptor.parameters.world_descriptor_name
    loader = module_loader.ModuleLoader()
    try:
        return loader.load_world_descriptor(ros_package, module_name)
    except Exception as error:
        raise WorldDescriptorLoadError(
            ros_package=ros_package,
            module_name=module_name,
        ) from error
