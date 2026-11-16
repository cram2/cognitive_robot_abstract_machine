"""
Debug plots Giskard draws of a finished goal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cramph.executor import StatechartExecutor
from semantic_digital_twin.adapters.ros import background_node
from giskardpy.middleware.ros2.plot_archive import GoalPlotKind, RunPlotDirectory


@dataclass(kw_only=True)
class PostGoalPlotter(ABC):
    """
    Writes a debug plot of the finished motion to a file.
    """

    executor: StatechartExecutor
    """
    The executor holding the data that is plotted.
    """

    plot_directory: RunPlotDirectory
    """
    Where the plot of each goal is written.
    """

    @abstractmethod
    def plot(self, *, goal_id: int) -> None:
        """
        Plot the data recorded for the goal with the given id.
        """


@dataclass(kw_only=True)
class GoalGanttChartPlotter(PostGoalPlotter):
    """
    Plots when each motion statechart node was active.
    """

    second_length_in_cm: float = 1.5
    """
    Width of one second of motion in the chart.
    """

    def plot(self, *, goal_id: int) -> None:
        if not self.executor.statechart.state_history:
            return
        file_name = self.plot_directory.goal_plot_path(
            plot_kind=GoalPlotKind.GANTT_CHART, goal_id=goal_id
        )
        self.executor.statechart.plot_gantt_chart(
            path=str(file_name),
            context=self.executor.context,
            second_length_in_cm=self.second_length_in_cm,
        )
        background_node.get_node().get_logger().info(f"saved {file_name}")


@dataclass(kw_only=True)
class MotionStatechartPlotter(PostGoalPlotter):
    """
    Draws the structure of the motion statechart that was executed.
    """

    def plot(self, *, goal_id: int) -> None:
        file_name = self.plot_directory.goal_plot_path(
            plot_kind=GoalPlotKind.MOTION_STATECHART, goal_id=goal_id
        )
        self.executor.statechart.draw(file_name=str(file_name))
        background_node.get_node().get_logger().info(f"saved {file_name}")
