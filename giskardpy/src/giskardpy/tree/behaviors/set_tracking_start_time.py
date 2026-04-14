from py_trees.common import Status
from rclpy.duration import Duration

from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard


class SetTrackingStartTime(GiskardBehavior):
    def __init__(self, name, offset: float = 0.5):
        super().__init__(name)
        self.offset = Duration(seconds=offset)

    def initialise(self):
        super().initialise()
        GiskardBlackboard().motion_start_time = (
            rospy.node.get_clock().now() + self.offset
        ).nanoseconds / 1e9

    def update(self):
        return Status.SUCCESS
