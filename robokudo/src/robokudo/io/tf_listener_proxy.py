"""
Transform listener proxy for RoboKudo.

This module provides a singleton proxy for ROS transform listeners. It ensures
that only one transform listener instance is created per node, which is important
because:

* The tf_static topic is latched and requires a single listener
* Multiple listeners in one node can cause buffer synchronization issues
* Transform lookups are more efficient with a shared listener

The module maintains a single global instance that can be accessed by all
components needing transform information.
"""

import sys

from rclpy.node import Node
from tf2_ros import TransformListener, Buffer

# Module-level singleton-like variables
this = sys.modules[__name__]
this.tf_buffer = None
this.tf_listener = None


def instance(node: Node) -> Buffer:
    """
    A singleton-like TransformListener instance.

    In ROS 2, a single TransformListener is recommended per node to ensure proper
    handling of the tf_static topic. Use this instance to access the tf_buffer.

    :param node: The ROS 2 node that will own the TransformListener.
    :return: The tf_buffer associated with the TransformListener.
    """
    if this.tf_listener is None:
        print("Initializing TF Listener")
        this.tf_buffer = Buffer()
        this.tf_listener = TransformListener(this.tf_buffer, node)

    return this.tf_buffer
