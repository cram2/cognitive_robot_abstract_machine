"""Per-node TF listener management for RoboKudo.

ROS 2 transform lookups are performed through a :class:`tf2_ros.Buffer`, while
the corresponding :class:`tf2_ros.TransformListener` must stay alive to keep the
buffer populated from ``/tf`` and ``/tf_static``.  This module exposes a
KRROOD-style singleton manager that creates and retains exactly one listener and
buffer pair per ROS node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from krrood.singleton import SingletonMeta
from rclpy.node import Node
from tf2_ros import TransformListener, Buffer


@dataclass
class _NodeTFListener:
    """TF listener state owned for a single ROS node."""

    buffer: Buffer
    listener: TransformListener


@dataclass
class TFListenerProxy(metaclass=SingletonMeta):
    """Singleton entry point for per-node TF listener buffers.

    The proxy itself is process-global, but the cached listener state is keyed
    by ROS node identity. Reusing a node returns the same buffer; using a
    different node creates a separate ``Buffer``/``TransformListener`` pair.
    """

    _listeners_by_node_id: dict[int, _NodeTFListener] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def buffer_for_node(self, node: Node) -> Buffer:
        """Return the TF buffer associated with ``node``.

        The listener is retained internally so subscriptions keep feeding the
        returned buffer for as long as the proxy cache entry exists.

        :param node: The ROS 2 node that owns the transform listener.
        :return: The TF buffer populated by that node's listener.
        """
        node_id = id(node)

        with self._lock:
            node_listener = self._listeners_by_node_id.get(node_id)
            if node_listener is None:
                buffer = Buffer()
                node_listener = _NodeTFListener(
                    buffer=buffer,
                    listener=TransformListener(buffer, node),
                )
                self._listeners_by_node_id[node_id] = node_listener

            return node_listener.buffer

    def clear_node(self, node: Node) -> None:
        """Forget the cached listener for ``node`` if one exists."""
        with self._lock:
            self._listeners_by_node_id.pop(id(node), None)

    def clear_cached_listeners(self) -> None:
        """Forget all cached per-node TF listeners while keeping this proxy."""
        with self._lock:
            self._listeners_by_node_id.clear()

    @classmethod
    def clear(cls) -> None:
        """Clear all cached TF listener state for this process."""
        cls().clear_cached_listeners()
        SingletonMeta.clear_instance(cls)
