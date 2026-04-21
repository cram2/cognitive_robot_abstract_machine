from typing import Optional

from py_trees.common import Status

from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard


class RosTime(GiskardBehavior):
    def __init__(self, name: Optional[str] = "ros time"):
        super().__init__(name)

    @property
    def start_time(self) -> float:
        return GiskardBlackboard().motion_start_time

    def update(self):
        GiskardBlackboard().giskard.executor._time = (
            rospy.node.get_clock().now().nanoseconds / 1e9 - self.start_time
        )
        return Status.SUCCESS
