from line_profiler.explicit_profiler import profile
from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    catch_and_raise_to_blackboard,
    GiskardBlackboard,
)


class ControllerPlugin(GiskardBehavior):

    @catch_and_raise_to_blackboard(skip_on_exception=False)
    @record_time
    @profile
    def update(self):
        GiskardBlackboard().executor.tick()
        if GiskardBlackboard().motion_statechart.is_end_motion():
            return Status.SUCCESS
        return Status.RUNNING
