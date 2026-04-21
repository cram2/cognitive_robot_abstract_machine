from py_trees.composites import Sequence

from giskardpy.tree.behaviors.cleanup import CleanUpPlanning
from giskardpy.tree.behaviors.plot_motion_graph import PlotMotionGraph
from giskardpy.tree.behaviors.ros_msg_to_goal import (
    SetExecutionMode,
    ParseActionGoal,
    AddBaseTrajFollowerGoal,
)
from giskardpy.tree.behaviors.set_tracking_start_time import SetTrackingStartTime


class PrepareControlLoop(Sequence):
    has_compile_debug_expressions: bool

    def __init__(self, name: str = "prepare control loop"):
        super().__init__(name, memory=True)
        self.has_compile_debug_expressions = False
        self.add_child(CleanUpPlanning("CleanUpPlanning"))
        self.add_child(SetExecutionMode())
        self.add_child(ParseActionGoal("RosMsgToGoal"))
        self.add_child(SetTrackingStartTime("start tracking time"))

    def add_plot_goal_graph(self):
        self.add_child(PlotMotionGraph())

    def add_compile_debug_expressions(self):
        if not self.has_compile_debug_expressions:
            self.has_compile_debug_expressions = True


class PrepareBaseTrajControlLoop(Sequence):
    has_compile_debug_expressions: bool

    def __init__(self, name: str = "prepare control loop"):
        super().__init__(name, memory=True)
        self.has_compile_debug_expressions = False
        self.add_child(CleanUpPlanning("CleanUpPlanning"))
        self.add_child(AddBaseTrajFollowerGoal())
        self.add_child(SetTrackingStartTime("start tracking time"))

    def add_plot_goal_graph(self):
        self.add_child(PlotMotionGraph())

    def add_compile_debug_expressions(self):
        if not self.has_compile_debug_expressions:
            self.has_compile_debug_expressions = True
