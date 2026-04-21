from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.spatial_types.derivatives import Derivatives


class SetZeroVelocity(GiskardBehavior):

    def __init__(self, name=None):
        if name is None:
            name = "set velocity to zero"
        super().__init__(name)

    @record_time
    def update(self):
        for (
            free_variable,
            state,
        ) in GiskardBlackboard().executor.context.world.state.items():
            for derivative in Derivatives.range(Derivatives.velocity, Derivatives.jerk):
                if derivative == Derivatives.position:
                    continue
                GiskardBlackboard().executor.context.world.state[free_variable][
                    derivative
                ] = 0
        GiskardBlackboard().executor.context.world.notify_state_change()
        return Status.SUCCESS
