from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from geometry_msgs.msg import Point as RosPoint
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from semantic_digital_twin.collision_checking.collision_detector import (
    CollisionCheckingResult,
    ClosestPoints,
)
from semantic_digital_twin.collision_checking.collision_manager import CollisionConsumer

if TYPE_CHECKING:
    from ....world import World


@dataclass
class CollisionVisualizationMarkerPublisher(CollisionConsumer):
    """
    Publishes the closest-points results of collision checks as an RViz marker.

    Each contact is drawn as a line segment between the two closest points of the
    checked body pair, colored by distance. This consumer is notified on every
    collision check via the :class:`CollisionConsumer` observer pattern, so the
    visualization stays live without any manual publishing.

    .. warning:: To see something in Rviz add a MarkerArray plugin, set the topic
        name, and make sure the fixed frame is the tf root.
    """

    node: Node = field(kw_only=True)
    """
    The ROS2 node that will be used to publish the visualization marker.
    """

    topic_name: str = "/semworld/viz_marker"
    """
    The name of the topic to which the closest-points marker should be published.
    """

    namespace: str = "__closest_points__"
    """
    The namespace of the marker.
    """

    throttle: int = field(kw_only=True, default=1)
    """
    Publish only on every nth collision check to reduce ROS traffic.
    """

    line_width: float = field(kw_only=True, default=0.005)
    """
    Width of the contact line segments in meters.
    """

    world: World = field(kw_only=True)
    """
    The world where this publisher will be added to.
    """

    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """
    QoS profile for the publisher.

    Uses TRANSIENT_LOCAL because it shares a topic with the VizMarkerPublisher.
    """

    _root_frame_name: str = field(init=False, default="")
    """
    Name of the tf frame the contact points are expressed in (the world root).
    """

    _call_counter: int = field(init=False, default=0)
    """
    Counts collision checks to implement throttling.
    """

    _publisher: Publisher = field(init=False)
    """
    The ROS publisher for the marker.
    """

    def __post_init__(self):
        self._publisher = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )
        time.sleep(0.2)
        self.world.collision_manager.add_collision_consumer(self)

    def on_world_model_update(self, world: World):
        self._root_frame_name = str(world.root.name)

    def on_collision_matrix_update(self):
        pass

    def on_compute_collisions(self, collision_results: CollisionCheckingResult):
        self._call_counter += 1
        if self._call_counter % self.throttle != 0:
            return
        marker_array = MarkerArray()
        marker_array.markers.append(self._build_contact_marker(collision_results))
        self._publisher.publish(marker_array)

    def _build_contact_marker(
        self, collision_results: CollisionCheckingResult
    ) -> Marker:
        """
        Builds a single ``LINE_LIST`` marker holding one segment per contact.

        The marker uses a fixed namespace and id so that each publish fully overwrites
        the previous one, clearing stale contacts.
        """
        marker = Marker()
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.ns = self.namespace
        marker.id = 0
        marker.header.frame_id = self._root_frame_name
        marker.frame_locked = True
        marker.scale.x = self.line_width
        marker.pose.orientation.w = 1.0
        for contact in collision_results.contacts:
            color = self._color_for_distance(contact)
            marker.points.append(self._to_ros_point(contact.root_P_point_on_body_a))
            marker.points.append(self._to_ros_point(contact.root_P_point_on_body_b))
            marker.colors.append(color)
            marker.colors.append(color)
        return marker

    def _color_for_distance(self, contact: ClosestPoints) -> ColorRGBA:
        """
        Returns red for contacts below the threshold and green otherwise.
        """
        if contact.distance < self.collision_manager.get_violated_distance(
            contact.body_a, contact.body_b
        ):
            return ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        elif contact.distance < self.collision_manager.get_buffer_zone_distance(
            contact.body_a, contact.body_b
        ):
            return ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        return ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)

    @staticmethod
    def _to_ros_point(root_point: np.ndarray) -> RosPoint:
        """
        Converts a homogeneous root-frame point into a :class:`RosPoint`.
        """
        return RosPoint(
            x=float(root_point[0]),
            y=float(root_point[1]),
            z=float(root_point[2]),
        )
