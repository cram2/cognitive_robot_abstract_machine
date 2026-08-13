from py_trees.common import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.utils.decorators import record_time


class ClearBlackboardException(GiskardBehavior):
    @record_time
    def update(self):
        if self.get_blackboard_exception() is not None:
            self.clear_blackboard_exception()
        return Status.SUCCESS
