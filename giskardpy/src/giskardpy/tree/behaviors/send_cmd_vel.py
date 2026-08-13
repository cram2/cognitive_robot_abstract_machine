import numpy as np
from geometry_msgs.msg import Twist
from py_trees.common import Status

from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)
from semantic_digital_twin.world_description.connections import OmniDrive


# can be used during closed-loop control, instead of for tracking a trajectory
class SendCmdVelTwist(GiskardBehavior):
    supported_state_types = [Twist]

    def __init__(self, topic_name: str, joint: OmniDrive = None):
        super().__init__()
        self.threshold = np.array([0.0, 0.0, 0.0])
        self.cmd_topic = topic_name
        self.vel_pub = rospy.node.create_publisher(Twist, self.cmd_topic, 10)

        self.joint = joint
        self.joint.has_hardware_interface = True
        rospy.node.get_logger().info(f"Created publisher for {self.cmd_topic}.")

    def solver_cmd_to_twist(self, twist) -> Twist:
        try:
            if abs(twist.linear.x) < self.threshold[0]:
                twist.linear.x = 0.0
        except:
            twist.linear.x = 0.0
        try:
            if abs(twist.linear.y) < self.threshold[1]:
                twist.linear.y = 0.0
        except:
            twist.linear.y = 0.0
        try:
            if abs(twist.angular.z) < self.threshold[2]:
                twist.angular.z = 0.0
        except:
            twist.angular.z = 0.0
        return twist

    @catch_and_raise_to_blackboard
    def update(self):
        cmd = Twist()

        x_vel = (
            GiskardBlackboard()
            .executor.context.world.state[self.joint.x_velocity.id]
            .velocity
        )
        if isinstance(self.joint, OmniDrive):
            y_vel = (
                GiskardBlackboard()
                .executor.context.world.state[self.joint.y_velocity.id]
                .velocity
            )
            cmd.linear.y = y_vel
        yaw_vel = (
            GiskardBlackboard().executor.context.world.state[self.joint.yaw.id].velocity
        )

        cmd.linear.x = x_vel
        cmd.angular.z = yaw_vel
        cmd = self.solver_cmd_to_twist(cmd)
        self.vel_pub.publish(cmd)
        return Status.RUNNING

    def terminate(self, new_status):
        self.vel_pub.publish(Twist())
        super().terminate(new_status)
