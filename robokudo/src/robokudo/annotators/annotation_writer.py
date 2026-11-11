"""Annotation writer classes for RoboKudo.

This module provides annotators for writing and publishing annotations.
"""

from __future__ import annotations

# %% Imports
import json
import os
import shutil
from dataclasses import dataclass, field
from timeit import default_timer

from krrood.ormatic.utils import classproperty
from py_trees.common import Status
from rclpy.publisher import Publisher
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from std_msgs.msg import String

from robokudo.annotators.core import BaseAnnotator
from robokudo.datastructures.annotator_descriptor import AnnotatorDescriptor
from robokudo.datastructures.annotator_parameters import AnnotatorParameters
from robokudo.io.cas_annotation_codecs import serialize_annotations

# %% AnnotationStorageWriter parameters


@dataclass(kw_only=True)
class AnnotationStorageWriterParameters(AnnotatorParameters):
    """Parameters for configuring annotation storage."""

    basic_path: str = "annotations"
    """Base directory for storing annotations"""

    suffix: str = "json"
    """File extension for annotation files"""


# %% AnnotationStorageWriter descriptor


@dataclass(kw_only=True)
class AnnotationStorageWriterDescriptor(
    AnnotatorDescriptor[AnnotationStorageWriterParameters]
):
    """Declarative configuration of :class:`AnnotationStorageWriter`."""

    parameters: AnnotationStorageWriterParameters = field(
        default_factory=AnnotationStorageWriterParameters.from_defaults
    )
    """Effective settings initialized from application and annotator defaults."""


# %% AnnotationStorageWriter annotator


@dataclass(eq=False)
class AnnotationStorageWriter(BaseAnnotator[AnnotationStorageWriterDescriptor]):
    """Annotator for writing annotations to storage in JSON format.

    This annotator writes the current CAS annotations to files in a specified
    directory, using JSON serialization.
    """

    name: str = __qualname__
    """Name of this node in the behaviour tree."""

    descriptor: AnnotationStorageWriterDescriptor = field(
        default_factory=AnnotationStorageWriterDescriptor
    )
    """Configuration supplied to this annotator instance."""

    counter: int = field(default=-1, init=False, repr=False)
    """Counter for generating sequential filenames"""

    @classproperty
    def Descriptor(cls) -> type[AnnotationStorageWriterDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return AnnotationStorageWriterDescriptor

    def update(self) -> Status:
        """Write current CAS annotations to storage.

        Serializes the current annotations to JSON and writes them to a file
        in the configured directory. Files are numbered sequentially.

        :return: SUCCESS after writing annotations
        """
        start_timer = default_timer()

        # Encode annotations via KRROOD serializer and dump to JSON.
        json_data_string = json.dumps(serialize_annotations(self.get_cas().annotations))

        dir_path = os.path.join(self.descriptor.parameters.basic_path, self.name)

        # increase count
        self.counter += 1

        if self.counter == 0 and os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        os.makedirs(dir_path, exist_ok=True)

        json_path = os.path.join(
            dir_path, "{}.{}".format(self.counter, self.descriptor.parameters.suffix)
        )
        with open(json_path, "w") as f:
            f.write(json_data_string)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS


# %% AnnotationPublisherWriter parameters


@dataclass(kw_only=True)
class AnnotationPublisherWriterParameters(AnnotatorParameters):
    """Parameters for configuring annotation publishing."""

    topic_name: str = "/annotations"
    """Name of the ROS topic to publish on"""


# %% AnnotationPublisherWriter descriptor


@dataclass(kw_only=True)
class AnnotationPublisherWriterDescriptor(
    AnnotatorDescriptor[AnnotationPublisherWriterParameters]
):
    """Declarative configuration of :class:`AnnotationPublisherWriter`."""

    parameters: AnnotationPublisherWriterParameters = field(
        default_factory=AnnotationPublisherWriterParameters.from_defaults
    )
    """Effective settings initialized from application and annotator defaults."""


# %% AnnotationPublisherWriter annotator


@dataclass(eq=False)
class AnnotationPublisherWriter(BaseAnnotator[AnnotationPublisherWriterDescriptor]):
    """Annotator for publishing annotations via ROS topics.

    This annotator publishes the current CAS annotations as JSON-encoded
    strings over a ROS topic.
    """

    name: str = __qualname__
    """Name of this node in the behaviour tree."""

    descriptor: AnnotationPublisherWriterDescriptor = field(
        default_factory=AnnotationPublisherWriterDescriptor
    )
    """Configuration supplied to this annotator instance."""

    pub: Publisher | None = field(default=None, init=False, repr=False)
    """ROS publisher for annotations"""

    @classproperty
    def Descriptor(cls) -> type[AnnotationPublisherWriterDescriptor]:
        """Specialize :meth:`robokudo.annotators.core.BaseAnnotator.Descriptor`."""
        return AnnotationPublisherWriterDescriptor

    def setup(self, timeout: float) -> None:
        """Set up the ROS2 publisher for annotation messages.

        :param timeout: Maximum time to wait for setup completion
        """
        self.rk_logger.debug("%s.setup()" % self.__class__.__name__)

        node = ROSNodeRegistry().get()
        self.pub = node.create_publisher(
            String, self.descriptor.parameters.topic_name, 10
        )

    def update(self) -> Status:
        """Publish current CAS annotations.

        Serializes the current annotations to JSON and publishes them
        on the configured ROS topic.

        :return: SUCCESS after publishing annotations
        """
        start_timer = default_timer()

        # Encode annotations via KRROOD serializer and dump to JSON.
        json_data_string = json.dumps(serialize_annotations(self.get_cas().annotations))

        # publish encoded data
        if self.pub is None:
            raise RuntimeError("Publisher is not initialized. Call setup() first.")
        msg = String(data=json_data_string)
        self.pub.publish(msg)

        end_timer = default_timer()
        self.feedback_message = f"Processing took {(end_timer - start_timer):.4f}s"
        return Status.SUCCESS
