from py_trees.common import Status

from giskardpy.tree.behaviors.action_server import ActionServerHandler
from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2.exceptions import ExecutionCanceledException
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import raise_to_blackboard, GiskardBlackboard


class GoalCanceled(GiskardBehavior):

    def __init__(self, action_server: ActionServerHandler):
        name = f"is '{action_server.name}' cancelled?"
        self.action_server = action_server
        super().__init__(name)

    @record_time
    def update(self) -> Status:
        if (
            self.action_server.is_cancel_requested()
            and self.get_blackboard_exception() is None
            or not self.action_server.is_client_alive()
        ):
            self.action_server.loginfo("canceled")
            raise_to_blackboard(
                ExecutionCanceledException(
                    action_server_name=self.action_server.name,
                    goal_id=self.action_server.goal_id,
                )
            )
        if self.get_blackboard_exception() is not None:
            return Status.SUCCESS
        else:
            return Status.FAILURE
