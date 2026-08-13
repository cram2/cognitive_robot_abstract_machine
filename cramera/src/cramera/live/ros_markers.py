"""
The optional ROS subscriber feeding the debug-marker overlay.

When the demo process has ROS available, this listens to the CRAM system's marker topics
and hands every received ``MarkerArray`` to the bridge (which excludes the world-
geometry markers and publishes the rest to the viewer — see
:mod:`cramera.live.markers`). Without ROS the listener simply reports itself unavailable
and the overlay stays empty.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

from typing_extensions import List, Optional, TYPE_CHECKING

from cramera.logging_setup import get_logger

if TYPE_CHECKING:
    from cramera.live.bridge import Bridge

logger = get_logger(__name__)

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import MarkerArray

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

MARKER_TOPICS_VARIABLE = "CRAMERA_MARKER_TOPICS"
"""
Environment variable holding a comma-separated list of marker topics to watch.
"""

DEFAULT_MARKER_TOPICS = ["/semworld/viz_marker", "/coraplex/viz_marker"]
"""
The topics the CRAM system publishes markers on.
"""

SUBSCRIPTION_QUEUE_DEPTH = 100
"""
How many marker arrays the subscription buffers.
"""


def marker_topics() -> List[str]:
    """
    The marker topics to watch: the environment's list, or the CRAM defaults.
    """
    configured = os.environ.get(MARKER_TOPICS_VARIABLE, "")
    if not configured.strip():
        return list(DEFAULT_MARKER_TOPICS)
    return [topic.strip() for topic in configured.split(",") if topic.strip()]


@dataclass
class RosMarkerListener:
    """
    Subscribes to the CRAM marker topics and feeds the bridge's marker overlay.
    """

    bridge: Bridge
    """
    The bridge receiving every marker array.
    """

    topics: List[str] = field(default_factory=marker_topics)
    """
    The topics being watched.
    """

    _node: object = field(init=False, default=None)
    """
    The listener's own ROS node, while started.
    """

    _subscriptions: dict = field(init=False, default_factory=dict)
    """
    The live subscriptions by topic, so the viewer can add and remove them.
    """

    _executor: object = field(init=False, default=None)
    """
    The executor spinning the node on a daemon thread, while started.
    """

    @classmethod
    def start_if_available(cls, bridge: Bridge) -> Optional[RosMarkerListener]:
        """
        Start a listener when ROS is importable, or report why not.

        :param bridge: The bridge receiving the markers.
        :return: The started listener, or None without ROS.
        """
        if not ROS_AVAILABLE:
            logger.info("ROS not importable — the marker overlay stays empty")
            return None
        listener = cls(bridge=bridge)
        listener.start()
        return listener

    def start(self) -> None:
        """
        Create the node and subscriptions and spin them on a daemon thread.
        """
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("cramera_markers")
        for topic in self.topics:
            self.subscribe(topic)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        threading.Thread(target=self._executor.spin, daemon=True).start()
        logger.info("watching markers on %s", ", ".join(self.topics))

    def subscribe(self, topic: str) -> None:
        """
        Start watching one marker topic; already watched topics stay as they are.

        :param topic: The topic to watch.
        """
        if topic in self._subscriptions:
            return
        durable = QoSProfile(
            depth=SUBSCRIPTION_QUEUE_DEPTH,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._subscriptions[topic] = self._node.create_subscription(
            MarkerArray, topic, self._on_markers(topic), durable
        )

    def unsubscribe(self, topic: str) -> None:
        """
        Stop watching one marker topic.

        :param topic: The topic to stop watching.
        """
        subscription = self._subscriptions.pop(topic, None)
        if subscription is not None:
            self._node.destroy_subscription(subscription)

    def subscribed_topics(self) -> List[str]:
        """
        The topics currently being watched.
        """
        return sorted(self._subscriptions)

    def available_marker_topics(self) -> List[str]:
        """
        Every ``MarkerArray`` topic currently advertised in the ROS graph.
        """
        return sorted(
            name
            for name, types in self._node.get_topic_names_and_types()
            if any(type_name.endswith("/MarkerArray") for type_name in types)
        )

    def _on_markers(self, topic: str):
        """
        The subscription callback handing a topic's arrays to the bridge.

        :param topic: The topic the subscription listens on.
        """

        def deliver(array: MarkerArray) -> None:
            self.bridge.observe_ros_markers(topic, list(array.markers))

        return deliver

    def stop(self) -> None:
        """
        Stop spinning and destroy the node.
        """
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
