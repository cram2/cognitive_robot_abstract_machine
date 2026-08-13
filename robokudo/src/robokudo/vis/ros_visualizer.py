"""ROS-based visualization for RoboKudo pipelines.

This module provides ROS-based visualization capabilities for RoboKudo pipelines.
It handles:

* Image publishing to ROS topics
* Single and multi-view visualization
* Pipeline state visualization
* Dynamic topic management
* Image format conversion
"""

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from typing_extensions import Any, Dict
from rclpy.publisher import Publisher

from robokudo.annotators.core import BaseAnnotator
from robokudo.vis.visualizer import Visualizer


class SharedROSVisualizer(Visualizer, Visualizer.Observer, Node):
    """A single-view ROS Image Publisher. It publishes the active annotator from the SharedState."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the shared ROS visualizer.

        .. note::
            This Visualizer works with a shared state and needs notifications
        """
        Visualizer.__init__(self, *args, **kwargs)
        Node.__init__(self, "shared_ros_visualizer")

        self.shared_visualizer_state.register_observer(self)

        self.ros_image_publisher: Publisher = self.create_publisher(
            Image, f"{self.pipeline.name}/output_image", 10
        )
        """Publisher for the image topic"""

        self.ros_image_cv_bridge: CvBridge = CvBridge()
        """Bridge for converting between ROS and OpenCV image formats"""

    def tick(self) -> None:
        """Update the visualization display.

        This method:

        * Gets current annotator outputs
        * Updates display if needed
        * Renders annotator name overlay
        * Publishes image to ROS topic
        """
        annotator_outputs = self.get_visualized_annotator_outputs_for_pipeline()

        assert self.shared_visualizer_state.active_annotator is not None
        active_annotator_instance: BaseAnnotator = (
            self.shared_visualizer_state.active_annotator
        )

        self.update_output_flag_for_new_data()

        # Check if we have to render something
        if self.update_output:
            self.update_output = False
            # 2D imshow output
            img = None
            if active_annotator_instance.name not in annotator_outputs.outputs:
                # We do not yet have visual output set up for this annotator
                # This might happen in dynamic perception pipelines, where annotators have not been set up
                # during construction of the tree AND don't generate image outputs.
                # Create an empty image to show in the visualizer
                img = np.zeros((640, 480, 3), dtype="uint8")
            else:
                img = annotator_outputs.outputs[active_annotator_instance.name].image
            img_with_annotator_text = cv2.putText(
                img,
                active_annotator_instance.name,
                (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            self.ros_image_publisher.publish(
                self.ros_image_cv_bridge.cv2_to_imgmsg(img_with_annotator_text)
            )

    def notify(
        self,
        observable: Visualizer.Observable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle notification of state changes.

        :param observable: The object that sent the notification
        """
        self.update_output = True


class AllAnnotatorROSVisualizer(Visualizer, Node):
    """A ROS Image Publisher that publishes the output of all images in the given Pipeline.

    This class provides visualization of all annotator outputs in a pipeline
    through separate ROS image topics. It supports:

    * Multiple image topic publishing
    * Dynamic topic creation
    * Image format conversion
    * Per-annotator output streams

    .. note::
        This Visualizer works with a shared state and needs notifications
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the multi-view ROS visualizer."""
        Visualizer.__init__(self, *args, **kwargs)
        Node.__init__(self, "all_annotator_ros_visualizer")

        self.ros_image_publishers: Dict[str, Publisher] = {}
        """Mapping of annotator names to ROS publishers"""

        self.ros_image_cv_bridge: CvBridge = CvBridge()
        """Bridge for converting between ROS and OpenCV image formats"""

    def update_ros_image_publishers(self) -> None:
        """Update ROS publishers for all annotators.

        This method:

        * Gets current list of annotators
        * Creates publishers for new annotators
        * Maintains existing publishers

        TODO: Consider removing publishers for non-existing annotators. May not be
         worth the cost if re-instantiation is frequent due to changing annotator lists.
        """
        annotator_list = self.pipeline.get_annotators()
        for annotator in annotator_list:
            if annotator.name not in self.ros_image_publishers:
                self.ros_image_publishers[annotator.name] = self.create_publisher(
                    Image, f"{self.pipeline.name}/{annotator.name}/output_image", 10
                )

    def tick(self) -> None:
        """Update all visualization displays.

        This method:

        * Gets current annotator outputs
        * Updates publishers if needed
        * Publishes all annotator images to ROS topics
        """
        annotator_outputs = self.get_visualized_annotator_outputs_for_pipeline()

        self.update_output_flag_for_new_data()

        # Check if we have to render something
        if self.update_output:
            self.update_output = False

            self.update_ros_image_publishers()

            for annotator_name, annotator_output in annotator_outputs.outputs.items():
                img = annotator_output.image
                self.ros_image_publishers[annotator_name].publish(
                    self.ros_image_cv_bridge.cv2_to_imgmsg(img)
                )

    def notify(
        self,
        observable: Visualizer.Observable,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Handle notification of state changes.

        :param observable: The object that sent the notification
        """
        self.update_output = True
