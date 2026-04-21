from typing import Optional

from py_trees.common import Status
from sensor_msgs.msg import JointState

from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_state import WorldState


class SyncJointState(GiskardBehavior):

    @record_time
    def __init__(self, group_name: str, joint_state_topic: str = "joint_states"):
        self.data = None
        self.joint_state_topic = joint_state_topic
        if not self.joint_state_topic.startswith("/"):
            self.joint_state_topic = "/" + self.joint_state_topic
        super().__init__(str(self))

    @record_time
    def setup(self, **kwargs):
        # wait_for_topic_to_appear(topic_name=self.joint_state_topic, supported_types=[JointState])
        self.joint_state_sub = rospy.node.create_subscription(
            JointState, self.joint_state_topic, self.cb, 1
        )
        return super().setup(**kwargs)

    def cb(self, data):
        self.data = data

    @record_time
    def update(self):
        if self.data:
            for i, joint_name in enumerate(self.data.name):
                connection: (
                    ActiveConnection1DOF
                ) = GiskardBlackboard().executor.context.world.get_connection_by_name(
                    joint_name
                )
                connection._world.state[connection.raw_dof.id].position = (
                    self.data.position[i]
                )
            self.data = None
            return Status.SUCCESS
        return Status.RUNNING

    def __str__(self):
        return f"{super().__str__()} ({self.joint_state_topic})"


class SyncJointStatePosition(GiskardBehavior):
    """
    Listens to a joint state topic, transforms it into a dict and writes it to the got map.
    Gets replace with a kinematic sim plugin during a parallel universe.
    """

    msg: JointState

    @record_time
    def __init__(self, group_name: str, joint_state_topic="joint_states"):
        super().__init__(str(self))
        self.joint_state_topic = joint_state_topic
        if not self.joint_state_topic.startswith("/"):
            self.joint_state_topic = "/" + self.joint_state_topic
        # wait_for_topic_to_appear(topic_name=self.joint_state_topic, supported_types=[JointState])
        super().__init__(str(self))
        self.mjs: Optional[WorldState] = None
        self.group_name = group_name

    @record_time
    def setup(self, **kwargs):
        self.joint_state_sub = rospy.node.create_subscription(
            JointState, self.joint_state_topic, self.cb, 1
        )
        rospy.node.get_logger().info(f"Subscribed to {self.joint_state_topic}")
        return super().setup(**kwargs)

    def cb(self, data):
        self.msg = data

    @record_time
    def update(self):
        for joint_name, position in zip(self.msg.name, self.msg.position):
            connection: (
                ActiveConnection1DOF
            ) = GiskardBlackboard().executor.context.world.get_connection_by_name(
                joint_name
            )
            connection._world.state[connection.raw_dof.id].position = position
        return Status.SUCCESS
