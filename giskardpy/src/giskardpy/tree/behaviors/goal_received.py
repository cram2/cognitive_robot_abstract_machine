from py_trees.common import Status

from giskardpy.tree.behaviors.action_server import ActionServerHandler
from giskardpy.tree.behaviors.plugin import GiskardBehavior


class GoalReceived(GiskardBehavior):
    def __init__(self, action_server: ActionServerHandler):
        name = f"has '{action_server.name}' goal?"
        self.action_server = action_server
        super().__init__(name)

    def update(self):
        if self.action_server.has_goal():
            self.action_server.accept_goal()
            return Status.SUCCESS
        return Status.FAILURE
