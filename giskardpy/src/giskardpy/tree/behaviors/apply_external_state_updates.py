from py_trees.common import Status

from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import (
    GiskardBlackboard,
    catch_and_raise_to_blackboard,
)


class ApplyExternalStateUpdates(GiskardBehavior):
    """Apply buffered external state updates for externally-updatable DOFs each control cycle.

    Runs inside the closed-loop synchronization, before the controller, so an external source (e.g.
    perception) can drive a flagged DOF's state during a running motion. Model changes and
    controller-owned DOFs are left untouched by the underlying synchronizer.
    """

    @catch_and_raise_to_blackboard
    def update(self) -> Status:
        GiskardBlackboard().giskard.world_synchronizer.apply_external_state_updates()
        return Status.SUCCESS
