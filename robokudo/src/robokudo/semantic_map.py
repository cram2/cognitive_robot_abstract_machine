"""
Semantic map management for RoboKudo.

This module provides classes for managing semantic maps in RoboKudo. A semantic map
represents the environment as a collection of named regions with spatial properties.
The module supports:

* Storage and retrieval of semantic map entries
* Visualization of regions using ROS markers
* Region definitions with position, orientation and size
"""

from __future__ import annotations

from dataclasses import dataclass

from rclpy.node import Node
from typing_extensions import TYPE_CHECKING, Set, Tuple, List, Dict, Optional
from visualization_msgs.msg import Marker, MarkerArray

from . import defs

if TYPE_CHECKING:
    from rclpy.publisher import Publisher


@dataclass
class SemanticMapEntry(defs.Region3DWithName):
    """A single entry in a semantic map.

    This class extends Region3DWithName to add semantic information to regions.
    Each entry has a reference frame ID and a semantic type.
    """

    frame_id: str = ""
    """The reference frame this region is defined in"""

    type: str = ""
    """The semantic type of this region (e.g. "CounterTop")"""

    plane_equation: Optional[List[float]] = None
    """Plane equation coefficients"""


class BaseSemanticMap:
    """Base class for managing semantic maps.

    This class provides functionality to store and visualize semantic map entries.
    Each entry represents a named region in the environment with spatial properties.
    The map can be visualized using ROS visualization markers.
    """

    def __init__(self) -> None:
        """
        Initialize an empty semantic map with visualization support.
        """

        self.entries: Dict[str, SemanticMapEntry] = dict()
        """Dictionary mapping region names to SemanticMapEntry objects."""

        self.node: Node = Node("base_semantic_map")
        """ROS node for publishing visualization markers."""

        self.vis_publisher: Publisher = self.node.create_publisher(
            MarkerArray, "visualization_marker_array", 10
        )
        """ROS publisher for visualization markers"""

    def add_entry(self, entry: SemanticMapEntry) -> None:
        """Add a single semantic map entry.

        :param entry: The semantic map entry to add
        :raises Exception: If the entry has no name attribute
        """
        if not hasattr(entry, "name"):
            raise Exception("Can't read name from SemanticMapEntry")

        self.entries[entry.name] = entry

    def add_entries(self, entries: List[SemanticMapEntry]) -> None:
        """Add multiple semantic map entries.

        :param entries: List of semantic map entries to add
        """
        for entry in entries:
            self.add_entry(entry)

    def publish_visualization_markers(
        self,
        highlighted: Optional[Set[str]] = None,
        highlight_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        default_color: Tuple[float, float, float] = (0.0, 0.0, 0.7),
    ) -> None:
        """Publish visualization markers for all map entries.

        Creates and publishes ROS visualization markers for each semantic map entry.
        Each entry is visualized as a cube with a text label showing its name.

        :param highlighted: Set of region names to highlight (default: None)
        :param highlight_color: Color to use for highlighted regions (default: green)
        :param default_color: Color to use for non-highlighted regions (default: dark blue)
        """
        vis_marker = MarkerArray()

        highlighted = highlighted or set()

        for name, entry in self.entries.items():
            marker = Marker()

            marker.header.frame_id = entry.frame_id
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.id = 0
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(entry.position_x)
            marker.pose.position.y = float(entry.position_y)
            marker.pose.position.z = float(entry.position_z)
            marker.pose.orientation.x = float(entry.orientation_x)
            marker.pose.orientation.y = float(entry.orientation_y)
            marker.pose.orientation.z = float(entry.orientation_z)
            marker.pose.orientation.w = float(entry.orientation_w)

            # Set color: default dark blue, switch to highlight color if intersected/highlighted
            if name in highlighted:
                r, g, b = highlight_color
            else:
                r, g, b = default_color
            marker.color.r = float(r)
            marker.color.g = float(g)
            marker.color.b = float(b)
            marker.color.a = 0.5
            marker.scale.x = float(entry.x_size)
            marker.scale.y = float(entry.y_size)
            marker.scale.z = float(entry.z_size)
            marker.ns = f"SemanticMap-{name}"
            vis_marker.markers.append(marker)

            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.id = 1
            text_marker.text = name
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.color.a = float(1)
            text_marker.pose = marker.pose
            text_marker.scale.z = 0.08
            text_marker.ns = f"SemanticMap-{name}"
            vis_marker.markers.append(text_marker)

        self.vis_publisher.publish(vis_marker)
