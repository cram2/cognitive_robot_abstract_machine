import time
from dataclasses import dataclass, field

from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import ColorRGBA
from typing_extensions import ClassVar
from visualization_msgs.msg import MarkerArray

from semantic_digital_twin.adapters.ros.msg_converter import ShapeROS2Converter
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
)


@dataclass
class VizMarkerPublisher(ModelChangeCallback):
    red: ClassVar[ColorRGBA] = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
    yellow: ClassVar[ColorRGBA] = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
    green: ClassVar[ColorRGBA] = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

    node: Node
    """
    The ROS2 node that will be used to publish the visualization marker.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The name of the topic to which the Visualization Marker should be published.
    """

    throttle_state_updates: int = 1
    """
    Only published every n-th state update.
    """

    use_visuals: bool = field(kw_only=True, default=True)

    markers: MarkerArray = field(init=False, default_factory=MarkerArray)
    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )

    def __post_init__(self):
        super().__post_init__()

        self.pub = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )
        time.sleep(0.2)
        self.notify()

    def collision_bodies(self):
        return [
            body
            for body in self.world.bodies_with_enabled_collision
            if body.has_collision()
        ]

    def _notify(self):
        if self.world.state.version % self.throttle_state_updates != 0:
            return
        self.markers = MarkerArray()
        for body in self.collision_bodies():
            marker_ns = str(body.name)
            if self.use_visuals:
                shapes = body.visual.shapes
            else:
                shapes = body.collision.shapes
            for i, shape in enumerate(shapes):
                marker = ShapeROS2Converter.to_ros2_message(shape)
                marker.frame_locked = True
                marker.id = i
                marker.ns = marker_ns
                self.markers.markers.append(marker)
        self.pub.publish(self.markers)
