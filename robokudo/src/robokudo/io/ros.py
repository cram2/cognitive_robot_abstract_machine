from threading import Lock

from rclpy import create_node
from rclpy.node import Node
from typing_extensions import Any

from robokudo.exceptions import RoboKudoROSNodeMissing

_rk_node: Node = None
"""
Central RoboKudo ROS node.
"""

_rk_node_lock = Lock()
"""
Lock for safe creation of the central ROS node.
"""


def init_node(node_name: str, *args: Any, **kwargs: Any) -> Node:
    """
    Initialize the central RoboKudo ROS node.

    Args and kwargs are passed directly to rclpy.create_node().     Initializes the
    global rk_node variable if not already initialized. The node can simply be accessed
    through     robokudo.io.ros.rk_node at any time.

    :param node_name: Name of the ROS node
    :return: The newly created ROS node
    """
    global _rk_node
    with _rk_node_lock:
        if _rk_node is None:
            _rk_node = create_node(node_name, *args, **kwargs)
    return _rk_node


def register_node(node: Node) -> None:
    """
    Register an existing ROS node as RoboKudo's central node.

    :param node: ROS node to make available through :func:`get_node`.
    """
    global _rk_node
    with _rk_node_lock:
        _rk_node = node


def clear_node(node: Node) -> None:
    """
    Clear the central ROS node if ``node`` is currently registered.

    :param node: Node whose registration should be removed.
    """
    global _rk_node
    with _rk_node_lock:
        if _rk_node is node:
            _rk_node = None


def get_node() -> Node:
    """
    Get the central RoboKudo ROS node instance.

    :return: The central ROS node instance
    :raises RoboKudoROSNodeMissing: If the node has not been initialized yet.
    """
    global _rk_node
    with _rk_node_lock:
        if _rk_node is None:
            raise RoboKudoROSNodeMissing()
        else:
            return _rk_node
