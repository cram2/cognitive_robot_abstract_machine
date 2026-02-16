import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID

from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import MarkerArray

from ..msg_converter import SemDTToRos2Converter
from ..tf_publisher import TFPublisher
from ....callbacks.callback import ModelChangeCallback


class ShapeSource(Enum):
    """
    Enum to specify which shapes to use for visualization.
    """

    VISUAL_ONLY = "visual_only"
    """
    The shapes to use for visualization are visual shapes only.
    """

    COLLISION_ONLY = "collision_only"
    """
    The shapes to use for visualization are collision shapes only.
    """

    VISUAL_WITH_COLLISION_BACKUP = "visual_with_collision_backup"
    """
    The shapes to use for visualization are visual shapes, but if there are no visual shapes, use collision shapes as a backup.
    """


@dataclass
class VizMarkerPublisher(ModelChangeCallback):
    """
    Publishes the world model as a visualization marker.
    .. warning:: Relies on the tf tree to correctly position the markers.
        Use TFPublisher to publish the tf tree.
    .. warning:: To see something in Rviz you must:
        1. add a MarkerArray plugin,
        2. set the current topic name,
        3. set DurabilityPolicy.TRANSIENT_LOCAL,
        4. make sure that the fixed frame is the tf root.
    """

    node: Node
    """
    The ROS2 node that will be used to publish the visualization marker.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The name of the topic to which the Visualization Marker should be published.
    """

    shape_source: ShapeSource = field(
        kw_only=True, default=ShapeSource.VISUAL_WITH_COLLISION_BACKUP
    )
    """
    Which shapes to use for each body
    """

    markers: MarkerArray = field(init=False, default_factory=MarkerArray)
    """Maker message to be published."""
    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """QoS profile for the publisher."""

    def __post_init__(self):
        super().__post_init__()

        self.pub = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )
        time.sleep(0.2)
        self.notify()
        time.sleep(0.2)

    def with_tf_publisher(self):
        """
        Launches a tf publisher in conjunction with the VizMarkerPublisher.
        """
        TFPublisher(self.world, self.node)

    def _select_shapes(self, body):
        if self.shape_source is ShapeSource.VISUAL_ONLY:
            return body.visual.shapes
        if self.shape_source is ShapeSource.COLLISION_ONLY:
            return body.collision.shapes
        if self.shape_source is ShapeSource.VISUAL_WITH_COLLISION_BACKUP:
            return body.visual.shapes if body.visual.shapes else body.collision.shapes
        raise ValueError(f"Unsupported shape_source: {self.shape_source!r}")

    def _notify(self, **kwargs):
        self.markers = MarkerArray()
        for body in self.world.bodies:
            shapes = self._select_shapes(body)
            if not shapes:
                continue
            marker_ns = str(body.name)
            for i, shape in enumerate(shapes):
                marker = SemDTToRos2Converter.convert(shape)
                marker.frame_locked = True
                marker.id = i
                marker.ns = marker_ns
                self.markers.markers.append(marker)
        self.pub.publish(self.markers)
