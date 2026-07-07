from dataclasses import dataclass

try:
    from rclpy.node import Node
except ImportError:
    from semantic_digital_twin.utils import MockedNodeClass as Node

from giskardpy.motion_statechart.context import ContextExtension


@dataclass
class RosContextExtension(ContextExtension):
    ros_node: Node
