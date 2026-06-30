import time
from dataclasses import dataclass, field
from typing import Union

import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from typing_extensions import Any
from visualization_msgs.msg import MarkerArray

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    PoseLikeMarkerRenderer,
    SpatialTypeVisualization,
)
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)


@dataclass
class PosePublisher(ModelChangeCallback):
    pose: Union[HomogeneousTransformationMatrix, Pose] = field(kw_only=True)
    """
    The pose to publish.
    """
    node: rclpy.node.Node = field(kw_only=True)
    """
    ROS node handle, used to create the publisher.
    """
    lifetime: int = 0
    """
    Lifetime of the PosePublisher and viz marker in seconds. If the lifetime is 0 the marker will stay indefinitely.
    """
    text: str = None
    """
    Text to display at the pose position
    """
    topic_name: str = "/semworld/viz_marker"
    """
    Topic name to publish the pose marker on.
    """

    publisher: Any = field(init=False)
    """
    Ros publisher for viz marker
    """
    end_time: float = field(init=False)
    """
    End time for this PosePublisher, used for lifetime only if given lifetime is greater than 0
    """
    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """QoS profile for the publisher."""

    def on_model_change(self, **kwargs):
        if self.lifetime > 0 and time.time() >= self.end_time:
            self.pause()
        marker_array = self._create_marker_array()
        self.publisher.publish(marker_array)

    def __post_init__(self):
        if not self._world:
            self._world = self.pose.reference_frame._world
        super().__post_init__()
        self.fixed_frame = str(self._world.root.name)
        self.publisher = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )
        time.sleep(0.2)
        self.end_time = time.time() + self.lifetime

        self.on_model_change()

    def with_tf_publisher(self):
        """
        Launches a tf publisher in conjunction with the PosePublisher.
        """
        TFPublisher(_world=self._world, node=self.node)

    def _create_marker_array(self) -> MarkerArray:
        """
        Creates a MarkerArray to visualize the pose as an RGB axis triad, with an optional text label.
        """
        reference_frame = self.pose.reference_frame
        reference_frame_name = (
            self.fixed_frame if reference_frame is None else str(reference_frame.name)
        )
        request = SpatialTypeVisualization(
            spatial_type=self.pose,
            namespace=f"pose/{reference_frame_name}/{id(self)}",
            label=self.text,
            lifetime_seconds=self._remaining_lifetime(),
        )
        return MarkerArray(
            markers=PoseLikeMarkerRenderer.render_markers(request, self.fixed_frame)
        )

    def _remaining_lifetime(self) -> float:
        """
        The remaining marker lifetime in seconds, or ``0`` if the markers should stay indefinitely.
        """
        if self.lifetime <= 0:
            return 0.0
        return self.end_time - time.time()

    def __hash__(self):
        return hash(id(self))
