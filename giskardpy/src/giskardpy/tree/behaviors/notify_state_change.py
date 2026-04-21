from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class NotifyStateChange(GiskardBehavior):

    @record_time
    def update(self):
        if GiskardBlackboard().executor.context.world.world_is_being_modified:
            return Status.RUNNING
        GiskardBlackboard().executor.context.world.notify_state_change()
        return Status.SUCCESS


class NotifyModelChange(GiskardBehavior):

    @catch_and_raise_to_blackboard(skip_on_exception=False)
    @record_time
    def update(self):
        GiskardBlackboard().executor.context.world._notify_model_change()
        return Status.SUCCESS
