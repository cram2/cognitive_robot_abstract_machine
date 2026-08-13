import json

from json_msgs.action import JsonAction
from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2.exceptions import (
    ExecutionCanceledException,
    ExecutionAbortedException,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard


class SetMoveResult(GiskardBehavior):

    def __init__(self, name, context: str, print=True):
        self.print = print
        self.context = context
        super().__init__(name)

    @record_time
    def update(self):
        e = self.get_blackboard_exception()
        move_result = JsonAction.Result()
        match e:
            case ExecutionCanceledException():
                GiskardBlackboard().move_action_server.set_canceled()
            case None:
                GiskardBlackboard().move_action_server.set_succeeded()
            case _:
                GiskardBlackboard().move_action_server.set_aborted()

        result = {
            "life_cycle_state": GiskardBlackboard().motion_statechart.life_cycle_state.to_json(),
            "observation_state": GiskardBlackboard().motion_statechart.observation_state.to_json(),
        }

        move_result.result = json.dumps(result)
        if isinstance(e, ExecutionCanceledException):
            rospy.node.get_logger().warning(f"Goal canceled by user.")
        else:
            if self.print:
                rospy.node.get_logger().info(f"{self.context} succeeded.")

        GiskardBlackboard().move_action_server.result_msg = move_result
        return Status.SUCCESS
