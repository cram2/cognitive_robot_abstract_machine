from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class CleanUp(GiskardBehavior):

    @record_time
    def initialise(self):
        if hasattr(GiskardBlackboard().executor, "motion_statechart"):
            GiskardBlackboard().motion_statechart.cleanup_nodes(
                context=GiskardBlackboard().executor.context
            )
        self.get_blackboard().runtime = None

    def update(self):
        return Status.SUCCESS


class CleanUpPlanning(CleanUp):
    def initialise(self):
        super().initialise()
        GiskardBlackboard().fill_trajectory_velocity_values = None

    @catch_and_raise_to_blackboard
    def update(self):
        return super().update()
