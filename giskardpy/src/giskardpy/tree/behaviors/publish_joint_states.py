from py_trees.common import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    GiskardBlackboard,
    catch_and_raise_to_blackboard,
)


class PublishWorldState(GiskardBehavior):

    @catch_and_raise_to_blackboard
    def update(self):
        GiskardBlackboard().giskard.world_synchronizer.on_state_change(
            publish_changes=True
        )
        return Status.SUCCESS
