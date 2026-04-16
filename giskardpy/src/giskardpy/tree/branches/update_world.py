from py_trees.common import Status
from py_trees.composites import Sequence

from .publish_state import PublishState
from .synchronization import Synchronization
from ..behaviors.plugin import GiskardBehavior
from ..behaviors.world_updater import ProcessWorldUpdate
from ..blackboard_utils import GiskardBlackboard


class HasWorldUpdate(GiskardBehavior):
    def __init__(self):
        super().__init__("has world update?")

    def update(self) -> Status:
        if len(GiskardBlackboard().giskard.model_synchronizer.missed_messages) > 0:
            return Status.SUCCESS
        return Status.FAILURE


class UpdateWorld(Sequence):
    synchronization: Synchronization
    publish_state: PublishState
    goal_received: HasWorldUpdate
    process_goal: ProcessWorldUpdate

    def __init__(self):
        name = "update world"
        super().__init__(name, memory=True)
        self.goal_received = HasWorldUpdate()
        self.process_goal = ProcessWorldUpdate()

        self.add_child(self.goal_received)
        self.add_child(self.process_goal)
