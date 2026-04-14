from typing import List

from py_trees.common import Status
from std_msgs.msg import Float64MultiArray

from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


class JointGroupVelController(GiskardBehavior):
    connections: List[ActiveConnection1DOF]

    def __init__(self, cmd_topic: str, connections: List[ActiveConnection1DOF]):
        super().__init__()
        self.cmd_topic = cmd_topic
        self.cmd_pub = rospy.node.create_publisher(
            Float64MultiArray, self.cmd_topic, 10
        )

        self.connections = connections
        for connection in self.connections:
            connection.has_hardware_interface = True
        self.msg = None
        rospy.node.get_logger().info(
            f"Created publisher for {self.cmd_topic} for {[c.name.name for c in self.connections]}"
        )

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        msg = Float64MultiArray()
        for i, connection in enumerate(self.connections):
            msg.data.append(connection.velocity)
        self.cmd_pub.publish(msg)
        return Status.RUNNING

    def terminate(self, new_status):
        msg = Float64MultiArray()
        for _ in self.connections:
            msg.data.append(0.0)
        self.cmd_pub.publish(msg)
        super().terminate(new_status)
