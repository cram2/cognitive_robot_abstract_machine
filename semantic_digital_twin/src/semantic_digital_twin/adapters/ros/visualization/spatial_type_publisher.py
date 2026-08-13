from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray

from semantic_digital_twin.adapters.ros.visualization.exceptions import (
    WorldNotResolvableError,
)
from semantic_digital_twin.adapters.ros.visualization.spatial_type_marker_renderer import (
    SpatialTypeMarkerRenderer,
    SpatialTypeVisualization,
)
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.spatial_types.spatial_types import SpatialType
from semantic_digital_twin.world_description.geometry import Color

if TYPE_CHECKING:
    from rclpy.publisher import Publisher

    from semantic_digital_twin.world import World


@dataclass(frozen=True)
class MarkerIdentity:
    """Identifies a published marker by the fields RViz uses to track it."""

    namespace: str
    """The marker namespace (``Marker.ns``)."""

    marker_id: int
    """The marker id within the namespace (``Marker.id``)."""


@dataclass(eq=False)
class SpatialTypePublisher(StateChangeCallback):
    """
    Publishes spatial types as RViz markers and keeps them updated as the robot moves.

    Registered spatial types are re-evaluated against the live world state and
    republished on every joint-state update, so symbolic expressions track motion.
    """

    node: Node = field(kw_only=True)
    """The ROS2 node used to create the marker publisher."""

    topic_name: str = "/semworld/viz_marker"
    """The topic the markers are published on."""

    qos_profile: QoSProfile = field(
        default_factory=lambda: QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
    )
    """The QoS profile of the publisher. Latched so RViz sees the last markers on connect."""

    _requests: list[SpatialTypeVisualization] = field(init=False, default_factory=list)
    """The spatial types that are re-evaluated and republished on every state change."""

    publisher: Publisher = field(init=False)
    """The ROS2 marker array publisher."""

    _published_marker_identities: set[MarkerIdentity] = field(
        init=False, default_factory=set
    )
    """Identities of the markers published on the previous call, used to delete stale ones."""

    def __post_init__(self):
        super().__post_init__()
        self.publisher = self.node.create_publisher(
            MarkerArray, self.topic_name, self.qos_profile
        )

    def add(self, request: SpatialTypeVisualization) -> None:
        """Register a spatial type for visualization and publish immediately."""
        self._requests.append(request)
        self.publish()

    def add_all(self, requests: Iterable[SpatialTypeVisualization]) -> None:
        """Register several spatial types for visualization and publish immediately."""
        self._requests.extend(requests)
        self.publish()

    def set_requests(self, requests: Iterable[SpatialTypeVisualization]) -> None:
        """Replace all registered spatial types and publish immediately."""
        self._requests = list(requests)
        self.publish()

    def clear(self) -> None:
        """Remove all registered spatial types and delete their markers."""
        self._requests = []
        self.publish()

    def on_state_change(self, **kwargs) -> None:
        """Republish all registered spatial types whenever the world state changes."""
        self.publish()

    def publish(self) -> None:
        """Render all registered spatial types and publish them as a single marker array.

        Markers published on the previous call that are no longer rendered are
        appended as ``DELETE`` markers so RViz removes them.
        """
        root_frame_name = str(self._world.root.name)
        marker_array = MarkerArray()
        for request in self._requests:
            marker_array.markers.extend(
                SpatialTypeMarkerRenderer.render(request, root_frame_name)
            )
        current_identities = {
            MarkerIdentity(marker.ns, marker.id) for marker in marker_array.markers
        }
        for identity in self._published_marker_identities - current_identities:
            marker_array.markers.append(self._delete_marker(identity))
        self.publisher.publish(marker_array)
        self._published_marker_identities = current_identities

    @staticmethod
    def _delete_marker(identity: MarkerIdentity) -> Marker:
        """Build a marker that deletes the previously published marker with this identity."""
        marker = Marker()
        marker.action = Marker.DELETE
        marker.ns = identity.namespace
        marker.id = identity.marker_id
        return marker

    @classmethod
    def publish_once(
        cls,
        *,
        spatial_type: SpatialType,
        node: Node,
        color: Color | None = None,
        label: str | None = None,
        world: World | None = None,
    ) -> SpatialTypePublisher:
        """
        Render and publish a single spatial type immediately.

        Convenience entry point for debugging: resolves the world from the spatial
        type's reference frame when not given. The returned publisher keeps tracking
        the spatial type until :meth:`stop` is called.

        :raises WorldNotResolvableError: When no world is given and the spatial type has no reference frame.
        """
        if world is None:
            reference_frame = spatial_type.reference_frame
            if reference_frame is None:
                raise WorldNotResolvableError(spatial_type=spatial_type)
            world = reference_frame._world
        publisher = cls(node=node, _world=world)
        publisher.add(
            SpatialTypeVisualization(
                spatial_type=spatial_type, color=color or Color(), label=label
            )
        )
        return publisher
