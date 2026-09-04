from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum, auto
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


# %% contact classification


class ContactProximity(IntEnum):
    """
    How close the bodies of a contact are relative to the distances of their collision
    rule.
    """

    VIOLATED = auto()
    """
    The bodies are closer than the distance their rule considers violated.
    """

    INSIDE_BUFFER_ZONE = auto()
    """
    The bodies are within the buffer zone of their rule, but not yet violating it.
    """

    CLEAR = auto()
    """
    The bodies are further apart than the buffer zone of their rule.
    """

    @property
    def color(self) -> ColorRGBA:
        """
        :return: The color a contact in this situation is drawn with.
        """
        if self is ContactProximity.VIOLATED:
            return ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        if self is ContactProximity.INSIDE_BUFFER_ZONE:
            return ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        return ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)


@dataclass
class ClassifiedContact:
    """
    A contact together with how close its bodies are relative to their collision rule.
    """

    contact: ClosestPoints
    """
    The closest points reported for the checked body pair.
    """

    proximity: ContactProximity
    """
    How close the bodies of the contact are.
    """

    @property
    def label(self) -> str:
        """
        :return: The names of both bodies and their distance, one per line.
        """
        return (
            f"{self.contact.body_a.name}\n"
            f"{self.contact.body_b.name}\n"
            f"{self.contact.distance:.3f}"
        )

    @property
    def root_P_midpoint(self) -> np.ndarray:
        """
        :return: The point halfway between the closest points, in the world root frame.
        """
        return (
            self.contact.root_P_point_on_body_a + self.contact.root_P_point_on_body_b
        ) / 2


# %% publisher


@dataclass
class CollisionVisualizationMarkerPublisher(CollisionConsumer):
    """
    Publishes the closest-points results of collision checks as an RViz marker.

    Each contact is drawn as a line segment between the two closest points of the
    checked body pair, colored by distance, and every contact that is at most a buffer
    zone away is labeled with the names of both bodies and their distance. This consumer
    is notified on every collision check via the :class:`CollisionConsumer` observer
    pattern, so the visualization stays live without any manual publishing.

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

    label_namespace: str = "__closest_points_labels__"
    """
    The namespace of the contact labels.

    Labels have a namespace of their own so that they can be hidden in Rviz while the
    contact lines stay visible.
    """

    throttle: int = field(kw_only=True, default=1)
    """
    Publish only on every nth collision check to reduce ROS traffic.
    """

    line_width: float = field(kw_only=True, default=0.005)
    """
    Width of the contact line segments in meters.
    """

    label_height: float = field(kw_only=True, default=0.03)
    """
    Height of the label text in meters.
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

    _published_label_count: int = field(init=False, default=0)
    """
    Number of labels of the previous publish, whose surplus ids have to be deleted.
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

    def stop(self):
        """
        Stop consuming collision results.

        The consumer publishes on a node that may already be gone once the world it is
        registered on outlives it.
        """
        self.world.collision_manager.remove_collision_consumer(self)

    def on_world_model_update(self, world: World):
        self._root_frame_name = str(world.root.name)

    def on_collision_matrix_update(self):
        pass

    def on_compute_collisions(self, collision_results: CollisionCheckingResult):
        self._call_counter += 1
        if self._call_counter % self.throttle != 0:
            return
        classified_contacts = [
            ClassifiedContact(
                contact=contact, proximity=self._classify_proximity(contact)
            )
            for contact in collision_results.contacts
        ]
        marker_array = MarkerArray()
        marker_array.markers.append(self._build_contact_marker(classified_contacts))
        marker_array.markers.extend(self._build_label_markers(classified_contacts))
        self._publisher.publish(marker_array)

    def _classify_proximity(self, contact: ClosestPoints) -> ContactProximity:
        """
        Compares the distance of a contact against the distances of the rule that
        applies to its body pair.
        """
        if contact.distance < self.collision_manager.get_violated_distance(
            contact.body_a, contact.body_b
        ):
            return ContactProximity.VIOLATED
        if contact.distance < self.collision_manager.get_buffer_zone_distance(
            contact.body_a, contact.body_b
        ):
            return ContactProximity.INSIDE_BUFFER_ZONE
        return ContactProximity.CLEAR

    def _build_contact_marker(
        self, classified_contacts: list[ClassifiedContact]
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
        for classified_contact in classified_contacts:
            color = classified_contact.proximity.color
            contact = classified_contact.contact
            marker.points.append(self._to_ros_point(contact.root_P_point_on_body_a))
            marker.points.append(self._to_ros_point(contact.root_P_point_on_body_b))
            marker.colors.append(color)
            marker.colors.append(color)
        return marker

    def _build_label_markers(
        self, classified_contacts: list[ClassifiedContact]
    ) -> list[Marker]:
        """
        Builds one text marker per contact that is at most a buffer zone away, followed
        by deletion markers for the labels of the previous publish that are now surplus.
        """
        labeled_contacts = [
            classified_contact
            for classified_contact in classified_contacts
            if classified_contact.proximity is not ContactProximity.CLEAR
        ]
        markers = [
            self._build_label_marker(classified_contact, marker_id)
            for marker_id, classified_contact in enumerate(labeled_contacts)
        ]
        markers.extend(
            self._build_label_deletion_marker(marker_id)
            for marker_id in range(len(labeled_contacts), self._published_label_count)
        )
        self._published_label_count = len(labeled_contacts)
        return markers

    def _build_label_marker(
        self, classified_contact: ClassifiedContact, marker_id: int
    ) -> Marker:
        """
        Builds the label of a contact, placed halfway between its closest points.
        """
        marker = Marker()
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.ns = self.label_namespace
        marker.id = marker_id
        marker.header.frame_id = self._root_frame_name
        marker.frame_locked = True
        marker.text = classified_contact.label
        marker.scale.z = self.label_height
        marker.color = classified_contact.proximity.color
        marker.pose.position = self._to_ros_point(classified_contact.root_P_midpoint)
        marker.pose.orientation.w = 1.0
        return marker

    def _build_label_deletion_marker(self, marker_id: int) -> Marker:
        """
        Builds a marker that removes the label with the given id from Rviz.
        """
        marker = Marker()
        marker.action = Marker.DELETE
        marker.ns = self.label_namespace
        marker.id = marker_id
        return marker

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
