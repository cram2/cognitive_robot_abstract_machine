"""
Draw what perception found into rviz, so detections can be checked against the real
table.

Each detection is drawn as its own measured outline rather than as a stand-in box, so a
misread shape looks wrong on screen instead of looking like a plausible box in the wrong
place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_msgs.msg import ColorRGBA
from typing_extensions import List
from visualization_msgs.msg import Marker, MarkerArray

from experiments.montessori.perception.colors import (
    BOARD_COLOR,
    HOLE_COLOR,
    LABEL_COLOR,
    PIECE_COLOR,
    DetectionColor,
)
from experiments.montessori.perception.detections import (
    MontessoriDetection,
    MontessoriScene,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% what gets drawn


class MarkerNamespace(StrEnum):
    """
    The groups detections are drawn in, so one kind can be switched off in rviz without
    the others.
    """

    PIECES = "montessori_pieces"
    BOARD = "montessori_board"
    HOLES = "montessori_holes"


MARKER_TOPIC = "montessori_perception/detections"
"""
Topic the detection markers are published on.
"""

_OUTLINE_WIDTH = 0.003
"""
Thickness of a drawn outline, in metres.
"""

_LIFETIME_SECONDS = 2
"""
How long a marker stays on screen if perception stops replacing it, so a stale detection
disappears rather than lingering.
"""

# %% publishing


@dataclass
class DetectionMarkerPublisher:
    """
    Publishes one marker array per look at the scene.
    """

    node: Node
    """
    The node the publisher is created on.
    """

    reference_frame: KinematicStructureEntity
    """
    The frame the detections' poses are expressed in.
    """

    _publisher: Publisher = field(init=False)
    """
    The marker array publisher.
    """

    def __post_init__(self) -> None:
        self._publisher = self.node.create_publisher(MarkerArray, MARKER_TOPIC, 1)

    def publish(self, scene: MontessoriScene) -> None:
        """
        Draw everything one look at the scene found.

        :param scene: The detections to draw.
        """
        markers = [
            self._outline(piece, MarkerNamespace.PIECES, index, PIECE_COLOR)
            for index, piece in enumerate(scene.shapes)
        ]
        markers += [
            self._outline(hole, MarkerNamespace.HOLES, index, HOLE_COLOR)
            for index, hole in enumerate(scene.holes)
        ]
        if scene.board is not None:
            markers.append(
                self._outline(scene.board, MarkerNamespace.BOARD, 0, BOARD_COLOR)
            )
        markers += [
            self._label(piece, index)
            for index, piece in enumerate(scene.shapes + scene.holes)
        ]
        self._publisher.publish(MarkerArray(markers=markers))

    def _outline(
        self,
        detection: MontessoriDetection,
        namespace: MarkerNamespace,
        identifier: int,
        color: DetectionColor,
    ) -> Marker:
        """
        Draw one detection's measured outline as a closed line strip.

        :param detection: The detection to draw.
        :param namespace: The group to draw it in.
        :param identifier: Its index within that group.
        :param color: The colour to draw it in.
        :return: The marker.
        """
        marker = self._new_marker(namespace, identifier, Marker.LINE_STRIP)
        marker.scale.x = _OUTLINE_WIDTH
        red, green, blue, alpha = color.to_rgba()
        marker.color = ColorRGBA(r=red, g=green, b=blue, a=alpha)
        height = float(detection.pose.to_position().to_np()[2])
        points = [Point(x=float(x), y=float(y), z=height) for x, y in detection.outline]
        marker.points = points + points[:1]
        return marker

    def _label(self, detection: MontessoriDetection, identifier: int) -> Marker:
        """
        Draw a detection's category above it, so a misclassification is visible on
        screen.

        :param detection: The detection to label.
        :param identifier: Its index among the labelled detections.
        :return: The marker.
        """
        marker = self._new_marker(
            MarkerNamespace.PIECES, 1000 + identifier, Marker.TEXT_VIEW_FACING
        )
        marker.scale.z = 0.02
        red, green, blue, alpha = LABEL_COLOR.to_rgba()
        marker.color = ColorRGBA(r=red, g=green, b=blue, a=alpha)
        position = detection.pose.to_position().to_np()
        marker.pose.position = Point(
            x=float(position[0]), y=float(position[1]), z=float(position[2]) + 0.05
        )
        marker.pose.orientation.w = 1.0
        marker.text = detection.label
        return marker

    def _new_marker(
        self, namespace: MarkerNamespace, identifier: int, marker_type: int
    ) -> Marker:
        """
        A marker already stamped with this publisher's frame and lifetime.

        :param namespace: The group the marker belongs to.
        :param identifier: Its index within that group.
        :param marker_type: Which rviz marker to draw.
        :return: The marker.
        """
        marker = Marker()
        marker.header.frame_id = str(self.reference_frame.name.name)
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = str(namespace)
        marker.id = identifier
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime.sec = _LIFETIME_SECONDS
        return marker
