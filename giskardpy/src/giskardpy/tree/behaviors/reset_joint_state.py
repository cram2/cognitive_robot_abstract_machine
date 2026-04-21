from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class ResetWorldState(GiskardBehavior):
    @record_time
    def __init__(self, name: str = "reset world state"):
        super().__init__(name)

    @catch_and_raise_to_blackboard
    @record_time
    def update(self):
        js = GiskardBlackboard().executor.context.world_state_trajectory[-1]
        GiskardBlackboard().executor.context.world.state = js
        GiskardBlackboard().executor.context.world.notify_state_change()
        return Status.SUCCESS
