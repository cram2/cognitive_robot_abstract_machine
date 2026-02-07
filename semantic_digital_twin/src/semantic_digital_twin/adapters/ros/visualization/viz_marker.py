from __future__ import annotations
import time
from dataclasses import dataclass, field

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import MarkerArray

from ..msg_converter import SemDTToRos2Converter
from ..tf_publisher import TFPublisher
from ..tfwrapper import TFWrapper
from ....callbacks.callback import ModelChangeCallback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ....world import World


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

    use_visuals: bool = field(kw_only=True, default=True)
    """
    Whether to use the visual shapes of the bodies or the collision shapes.
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
        TFPublisher(self.world, self.node)

    def _notify(self):
        self.markers = MarkerArray()
        for body in self.world.bodies:
            marker_ns = str(body.name)
            if self.use_visuals:
                shapes = body.visual.shapes
            else:
                shapes = body.collision.shapes
            for i, shape in enumerate(shapes):
                marker = SemDTToRos2Converter.convert(shape)
                marker.frame_locked = True
                marker.id = i
                marker.ns = marker_ns
                self.markers.markers.append(marker)
        self.pub.publish(self.markers)


def publish_world(world: World):

    try:
        rclpy.init()
    except RuntimeError:
        pass
    rclpy_node = rclpy.create_node("viz_marker_node")
    tf_wrapper = TFWrapper(node=rclpy_node)
    tf_publisher = TFPublisher(node=rclpy_node, world=world)
    viz = VizMarkerPublisher(world=world, node=rclpy_node, use_visuals=False)
