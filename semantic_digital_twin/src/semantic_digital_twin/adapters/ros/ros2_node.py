from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.node import Node

# %% the node everything talks over


@dataclass(eq=False)
class Ros2Node:
    """
    Base class for everything that communicates over a ROS2 node.
    """

    node: Node = field(kw_only=True)
    """
    The node the communication happens on.
    """
