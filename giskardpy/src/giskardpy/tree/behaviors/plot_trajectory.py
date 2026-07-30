import os
import tempfile
import traceback
from threading import Thread

from py_trees.common import Status

from giskardpy.utils.decorators import record_time
from giskardpy.middleware.ros2 import rospy
from giskardpy.tree.behaviors.plugin import GiskardBehavior
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)


class PlotTrajectory(GiskardBehavior):
    plot_thread: Thread

    def __init__(
        self,
        name,
        wait=False,
        normalize_position: bool = False,
        **kwargs,
    ):
        super().__init__(name)
        self.wait = wait
        self.normalize_position = normalize_position
        self.kwargs = kwargs
        GiskardBlackboard().executor.trajectory_plotter = WorldStateTrajectoryPlotter()

    def initialise(self):
        self.plot_thread = Thread(target=self.plot, name=self.name)
        self.plot_thread.start()

    def plot(self):
        try:
            plotter = GiskardBlackboard().executor.trajectory_plotter
            if len(plotter.world_state_trajectory.times) <= 1:
                return
            file_name = os.path.join(
                tempfile.gettempdir(),
                "trajectories",
                f"goal_{GiskardBlackboard().move_action_server.goal_id}.pdf",
            )
            GiskardBlackboard().executor.plot_trajectory(file_name)
            rospy.node.get_logger().info(f"saved {file_name}")
        except Exception as e:
            traceback.print_exc()
            rospy.node.get_logger().warning(e)
            rospy.node.get_logger().warning("failed to save trajectory.pdf")

    @record_time
    def update(self):
        if self.wait and self.plot_thread.is_alive():
            return Status.RUNNING
        return Status.SUCCESS
