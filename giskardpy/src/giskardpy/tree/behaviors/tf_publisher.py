from py_trees.common import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    GiskardBlackboard,
    catch_and_raise_to_blackboard,
)


class TFPublisher(GiskardBehavior):
    """
    Published tf for attached and environment objects.
    """

    @catch_and_raise_to_blackboard
    def update(self):
        GiskardBlackboard().giskard.tf_publisher._notify()
        return Status.SUCCESS
