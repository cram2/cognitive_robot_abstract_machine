"""
Debug plots Giskard draws of a finished goal.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

from cramph.executor import StatechartExecutor
from semantic_digital_twin.adapters.ros import background_node
from giskardpy.utils.utils import create_path


@dataclass(kw_only=True)
class PostGoalPlotter(ABC):
    """
    Writes a debug plot of the finished motion to a file.
    """

    executor: StatechartExecutor
    """
    The executor holding the data that is plotted.
    """

    @abstractmethod
    def plot(self, *, goal_id: int) -> None:
        """
        Plot the data recorded for the goal with the given id.
        """

    def create_file_name(self, *, folder_name: str, goal_id: int) -> str:
        """
        Build the path of a per-goal pdf inside the temporary directory and make sure
        its folder exists.
        """
        file_name = os.path.join(
            tempfile.gettempdir(), folder_name, f"goal_{goal_id}.pdf"
        )
        create_path(file_name)
        return file_name


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
        file_name = self.create_file_name(folder_name="gantt_charts", goal_id=goal_id)
        self.executor.statechart.plot_gantt_chart(
            path=file_name,
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
        file_name = self.create_file_name(
            folder_name="motion_statecharts", goal_id=goal_id
        )
        self.executor.statechart.draw(file_name=file_name)
        background_node.get_node().get_logger().info(f"saved {file_name}")
