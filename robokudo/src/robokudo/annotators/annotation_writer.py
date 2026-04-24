"""Annotation writer classes for RoboKudo.

This module provides annotators for writing and publishing annotations.
"""

import os
import shutil
import json
from timeit import default_timer

from py_trees.common import Status
from rclpy.publisher import Publisher
from std_msgs.msg import String
from typing_extensions import Optional

from robokudo.annotators.core import BaseAnnotator
from robokudo.io.cas_annotation_codecs import serialize_annotations
from robokudo.io.ros import get_node


class AnnotationStorageWriter(BaseAnnotator):
    """Annotator for writing annotations to storage in JSON format.

    This annotator writes the current CAS annotations to files in a specified
    directory, using JSON serialization.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for annotation storage writer."""

        class Parameters:
            """Parameters for configuring annotation storage."""

            def __init__(self) -> None:
                self.basic_path: str = "annotations"
                """Base directory for storing annotations, defaults to "annotations"""

                self.suffix: str = "json"
                """File extension for annotation files, defaults to "json"""

        # Overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "AnnotationStorageWriter",
        descriptor: "AnnotationStorageWriter.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the annotation storage writer. Minimal one-time init!

        :param name: Name of the annotator instance, defaults to "AnnotationStorageWriter"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.counter: int = -1
        """Counter for generating sequential filenames"""

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


class AnnotationPublisherWriter(BaseAnnotator):
    """Annotator for publishing annotations via ROS topics.

    This annotator publishes the current CAS annotations as JSON-encoded
    strings over a ROS topic.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        """Configuration descriptor for annotation publisher."""

        class Parameters:
            """Parameters for configuring annotation publishing."""

            def __init__(self) -> None:
                self.topic_name: str = "/annotations"
                """Name of the ROS topic to publish on, defaults to "/annotations"""

        # overwrite the parameters explicitly to enable auto-completion
        parameters = Parameters()

    def __init__(
        self,
        name: str = "AnnotationPublisherWriter",
        descriptor: "AnnotationPublisherWriter.Descriptor" = Descriptor(),
    ) -> None:
        """Initialize the annotation publisher. Minimal one-time init!

        :param name: Name of the annotator instance, defaults to "AnnotationPublisherWriter"
        :param descriptor: Configuration descriptor, defaults to Descriptor()
        """
        super().__init__(name, descriptor)
        self.rk_logger.debug("%s.__init__()" % self.__class__.__name__)

        self.pub: Optional[Publisher] = None
        """ROS publisher for annotations"""

    def setup(self, timeout: float) -> None:
        """Set up the ROS2 publisher for annotation messages.

        :param timeout: Maximum time to wait for setup completion
        """
        self.rk_logger.debug("%s.setup()" % self.__class__.__name__)

        node = get_node()
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
