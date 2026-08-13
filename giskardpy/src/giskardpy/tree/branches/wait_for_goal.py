from json_msgs.action import JsonAction
from py_trees.composites import Sequence
from py_trees.decorators import FailureIsSuccess

from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.action_server import ActionServerHandler
from giskardpy.tree.behaviors.goal_received import GoalReceived
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from giskardpy.tree.branches.publish_state import PublishState
from giskardpy.tree.branches.synchronization import Synchronization
from giskardpy.tree.branches.update_world import UpdateWorld


class WaitForGoal(Sequence):
    synchronization: Synchronization
    publish_state: PublishState
    goal_received: GoalReceived
    world_updater: UpdateWorld

    def __init__(self, name: str = "wait for goal"):
        super().__init__(name, memory=True)
        GiskardBlackboard().move_action_server = ActionServerHandler(
            action_name=f"{rospy.node.get_name()}/command", action_type=JsonAction
        )
        self.world_updater = UpdateWorld()
        self.world_updater_failure_is_success = FailureIsSuccess(
            "ignore failure", self.world_updater
        )
        self.synchronization = Synchronization()
        self.publish_state = PublishState()
        self.goal_received = GoalReceived(
            action_server=GiskardBlackboard().move_action_server
        )
        self.add_child(self.world_updater_failure_is_success)
        self.add_child(self.synchronization)
        self.add_child(self.publish_state)
        self.add_child(self.goal_received)
