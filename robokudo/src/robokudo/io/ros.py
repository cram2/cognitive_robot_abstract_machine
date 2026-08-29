"""
Access to RoboKudo's shared ROS node.

The application lifecycle creates or registers one node and keeps it alive while
RoboKudo runs. Annotators may retrieve that node at any time, including while its
executor is spinning, and may create and use ROS entities on it. They must not shut down
or destroy the shared node.

Calls that change the registered node are lifecycle operations and must be serialized by
their caller. Clearing the registry only removes its reference; it does not destroy the
node or any entities created from it.
"""

from dataclasses import dataclass, field
from typing import Any

from krrood.singleton import SingletonMeta
from rclpy import create_node
from rclpy.node import Node

from robokudo.exceptions import RoboKudoROSNodeMissing

# %% central node registry


@dataclass
class RoboKudoNodeRegistry(metaclass=SingletonMeta):
    """
    Keep the reference to RoboKudo's shared ROS node.

    The registry does not manage the node's ROS lifecycle. The application that
    initializes or registers the node must keep it alive until all users and its
    executor have stopped, then destroy it explicitly. Calls to :meth:`initialize`,
    :meth:`register`, and :meth:`clear` must not run concurrently.
    """

    _node: Node | None = field(default=None, init=False, repr=False)
    """
    Currently registered shared ROS node.
    """

    def initialize(self, node_name: str, *args: Any, **kwargs: Any) -> Node:
        """
        Create and return the central node when none is registered.
        """
        if self._node is None:
            self._node = create_node(node_name, *args, **kwargs)
        return self._node

    def register(self, node: Node) -> None:
        """
        Register an existing node without taking ownership of its lifecycle.
        """
        self._node = node

    def get(self) -> Node:
        """
        Return the shared node without taking ownership of its lifecycle.
        """
        if self._node is None:
            raise RoboKudoROSNodeMissing()
        return self._node

    def clear(self, node: Node | None = None) -> None:
        """
        Remove the matching node reference without destroying the node.
        """
        if node is None or self._node is node:
            self._node = None


def init_node(node_name: str, *args: Any, **kwargs: Any) -> Node:
    """
    Initialize RoboKudo's shared ROS node during application startup.

    Arguments are passed directly to :func:`rclpy.create_node`. The application owns the
    returned node and must keep it alive until its executor and all users have stopped.
    Startup and teardown code must serialize calls that initialize, register, or clear
    the shared node.

    :param node_name: Name of the ROS node
    :return: The newly created or previously registered ROS node
    """
    return RoboKudoNodeRegistry().initialize(node_name, *args, **kwargs)


def register_node(node: Node) -> None:
    """
    Register an existing ROS node as RoboKudo's shared node.

    Registration does not transfer lifecycle ownership. The caller remains responsible
    for the node's executor and eventual destruction.

    :param node: ROS node to make available through :func:`get_node`.
    """
    RoboKudoNodeRegistry().register(node)


def clear_node(node: Node | None = None) -> None:
    """
    Remove RoboKudo's reference to the matching shared ROS node.

    This function does not stop the executor or destroy the node. The application
    lifecycle must stop all users before clearing and destroying the node.

    :param node: Registered node to clear, or None to force a reset.
    """
    RoboKudoNodeRegistry().clear(node)


def get_node() -> Node:
    """
    Get RoboKudo's shared ROS node without taking lifecycle ownership.

    Annotators may call this while the node's executor is spinning and may create or use
    ROS entities on the returned node. They must not call
    :meth:`rclpy.node.Node.destroy_node` or shut down its context.

    :return: The shared ROS node instance
    :raises RoboKudoROSNodeMissing: If the node has not been initialized yet.
    """
    return RoboKudoNodeRegistry().get()
