from typing import List

from py_trees.common import Status
from std_msgs.msg import Float64

from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class JointVelController(GiskardBehavior):
    last_time: float

    @record_time
    def __init__(self, namespaces: List[str]):
        super().__init__("joint velocity publisher")
        self.namespaces = namespaces
        self.publishers = []
        self.cmd_topics = []
        self.joint_names = []
        for namespace in self.namespaces:
            cmd_topic = f"/{namespace}/command"
            self.cmd_topics.append(cmd_topic)
            wait_for_topic_to_appear(topic_name=cmd_topic, supported_types=[Float64])
            self.publishers.append(rospy.node.create_publisher(Float64, cmd_topic, 10))
            self.joint_names.append(rospy.get_param(f"{namespace}/joint"))
        for i in range(len(self.joint_names)):
            self.joint_names[
                i
            ] = GiskardBlackboard().executor.context.world.search_for_joint_name(
                self.joint_names[i]
            )
        GiskardBlackboard().executor.context.world.register_controlled_joints(
            self.joint_names
        )

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        msg = Float64()
        for i, joint_name in enumerate(self.joint_names):
            msg.data = (
                GiskardBlackboard().executor.context.world.state[joint_name].velocity
            )
            self.publishers[i].publish(msg)
        return Status.RUNNING

    def terminate(self, new_status):
        super().terminate(new_status)
        msg = Float64()
        for i, joint_name in enumerate(self.joint_names):
            self.publishers[i].publish(msg)
