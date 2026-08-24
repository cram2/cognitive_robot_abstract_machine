from _thread import LockType
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from krrood.singleton import SingletonMeta
from rclpy import create_node
from rclpy.node import Node

from robokudo.exceptions import RoboKudoROSNodeMissing

# %% central node registry


@dataclass
class RoboKudoNodeRegistry(metaclass=SingletonMeta):
    """
    Own RoboKudo's central ROS node.
    """

    _node: Node | None = field(default=None, init=False, repr=False)
    """
    Currently registered central ROS node.
    """

    _lock: LockType = field(default_factory=Lock, init=False, repr=False)
    """
    Synchronize access to the registered node.
    """

    def initialize(self, node_name: str, *args: Any, **kwargs: Any) -> Node:
        """
        Create and return the central node when none is registered.
        """
        with self._lock:
            if self._node is None:
                self._node = create_node(node_name, *args, **kwargs)
            return self._node

    def register(self, node: Node) -> None:
        """
        Register an existing node as the central node.
        """
        with self._lock:
            self._node = node

    def get(self) -> Node:
        """
        Return the central node.
        """
        with self._lock:
            if self._node is None:
                raise RoboKudoROSNodeMissing()
            return self._node

    def clear(self, node: Node | None = None) -> None:
        """
        Clear the central node.
        """
        with self._lock:
            if node is None or self._node is node:
                self._node = None


def init_node(node_name: str, *args: Any, **kwargs: Any) -> Node:
    """
    Initialize the central RoboKudo ROS node.

    Args and kwargs are passed directly to rclpy.create_node().     Initializes the
    global rk_node variable if not already initialized. The node can simply be accessed
    through     robokudo.io.ros.rk_node at any time.

    :param node_name: Name of the ROS node
    :return: The newly created ROS node
    """
    return RoboKudoNodeRegistry().initialize(node_name, *args, **kwargs)


def register_node(node: Node) -> None:
    """
    Register an existing ROS node as RoboKudo's central node.

    :param node: ROS node to make available through :func:`get_node`.
    """
    RoboKudoNodeRegistry().register(node)


def clear_node(node: Node | None = None) -> None:
    """
    Clear the central RoboKudo ROS node.

    :param node: Registered node to clear, or None to force a reset.
    """
    RoboKudoNodeRegistry().clear(node)


def get_node() -> Node:
    """
    Get the central RoboKudo ROS node instance.

    :return: The central ROS node instance
    :raises RoboKudoROSNodeMissing: If the node has not been initialized yet.
    """
    return RoboKudoNodeRegistry().get()
