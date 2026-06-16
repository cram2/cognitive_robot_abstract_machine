from threading import Thread

from py_trees.common import Status

from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from giskardpy.utils import objgraph_debug


class ProcessWorldUpdate(GiskardBehavior):

    def __init__(self):
        name = f"Processing world updates"
        self.worker_thread = None
        super().__init__(name)

    def update(self) -> Status:
        if self.worker_thread is None:
            self.worker_thread = Thread(target=self.process_goal, name=self.name)
            self.worker_thread.start()
        else:
            if not self.worker_thread.is_alive():
                self.worker_thread = None
                version = GiskardBlackboard().executor.context.world._model_manager.version
                rospy.node.get_logger().info(
                    f"Finished world update, model version: {version}."
                )
                objgraph_debug.report_growth(label=f"After world update version {version}")
                objgraph_debug.count_types(
                    [
                        "Compiled",
                        "Collision",
                        "Matrix",
                        "Connection",
                        "Body",
                        "DegreeOfFreedom",
                        "Monitor",
                        "Task",
                        "CollisionObject",
                    ],
                    label=f"v={version}",
                )
                return Status.SUCCESS
        return Status.RUNNING

    def process_goal(self):
        GiskardBlackboard().giskard.world_synchronizer.apply_missed_messages()
