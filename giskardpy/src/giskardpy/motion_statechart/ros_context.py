from dataclasses import dataclass

from rclpy.node import Node

from giskardpy.motion_statechart.context import ContextExtension


@dataclass
class RosContextExtension(ContextExtension):
    ros_node: Node
