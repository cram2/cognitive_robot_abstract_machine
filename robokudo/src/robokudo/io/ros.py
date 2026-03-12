import threading
import rclpy
import rclpy.node
from typing_extensions import Optional, Any

_rk_node = None
"""Central RoboKudo ROS node."""

_rk_node_lock = threading.Lock()
"""Lock for safe creation of the central ROS node."""


def init_node(node_name: str, *args: Any, **kwargs: Any) -> rclpy.node.Node:
    """Initialize the central RoboKudo ROS node. Args and kwargs are passed directly to rclpy.create_node().

    Initializes the global rk_node variable if not already initialized. The node can simply be accessed through
    robokudo.io.ros.rk_node at any time.

    :param node_name: Name of the ROS node
    :return: The newly created ROS node
    """
    global _rk_node
    with _rk_node_lock:
        if _rk_node is None:
            _rk_node = rclpy.create_node(node_name, *args, **kwargs)
    return _rk_node


def get_node() -> Optional[rclpy.node.Node]:
    """Get the central RoboKudo ROS node instance.

    :return: The central ROS node instance
    :raises RuntimeError: If the node has not been initialized yet
    """
    global _rk_node
    with _rk_node_lock:
        if _rk_node is None:
            raise RuntimeError("RoboKudo ROS node not initialized yet!")
        else:
            return _rk_node
