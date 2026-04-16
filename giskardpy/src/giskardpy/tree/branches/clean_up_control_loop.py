from py_trees.composites import Sequence
from py_trees.decorators import FailureIsSuccess

from giskardpy.tree.behaviors.plot_goal_gantt_chart import PlotGanttChart
from giskardpy.tree.behaviors.plot_trajectory import PlotTrajectory
from giskardpy.tree.behaviors.publish_feedback import ForcePublishFeedback
from giskardpy.tree.behaviors.reset_joint_state import ResetWorldState
from giskardpy.utils.decorators import toggle_on, toggle_off
from giskardpy.tree.behaviors.append_zero_velocity import SetZeroVelocity


class CleanupControlLoop(Sequence):
    reset_world_state = ResetWorldState

    def __init__(self, name: str = "clean up control loop"):
        super().__init__(name, memory=True)
        self.add_child(ForcePublishFeedback())
        self.add_child(SetZeroVelocity("set zero vel 1"))
        self.reset_world_state = ResetWorldState()
        self.reset_world_state_failure_is_success = FailureIsSuccess(
            "ignore failure", self.reset_world_state
        )
        self.remove_reset_world_state()

    def add_plot_trajectory(self, normalize_position: bool = False, wait: bool = False):
        self.insert_child(
            PlotTrajectory(
                "plot trajectory", wait=wait, normalize_position=normalize_position
            ),
            index=-1,
        )

    def add_plot_gantt_chart(self):
        self.insert_child(PlotGanttChart(), 2)

    @toggle_on("has_reset_world_state")
    def add_reset_world_state(self):
        self.add_child(self.reset_world_state_failure_is_success)

    @toggle_off("has_reset_world_state")
    def remove_reset_world_state(self):
        try:
            self.remove_child(self.reset_world_state_failure_is_success)
        except ValueError as e:
            pass  # it's fine, happens if it's called before add
