from py_trees.common import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    GiskardBlackboard,
    catch_and_raise_to_blackboard,
)


class PublishWorldState(GiskardBehavior):

    @catch_and_raise_to_blackboard
    def update(self):
        sync = GiskardBlackboard().giskard.state_synchronizer
        for msg in sync.missed_messages:
            sync.apply_message(msg)
        sync.missed_messages.clear()
        GiskardBlackboard().giskard.state_synchronizer.world_callback()
        return Status.SUCCESS
