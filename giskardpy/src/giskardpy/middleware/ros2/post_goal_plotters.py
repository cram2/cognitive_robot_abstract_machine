from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from giskardpy.executor import Executor
from giskardpy.middleware.ros2 import rospy
from giskardpy.utils.utils import create_path
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)


@dataclass
class PostGoalPlotter(ABC):
    """
    Writes a debug plot of the finished motion to a file.
    """

    executor: Executor
    """
    The executor holding the data that is plotted.
    """

    def start_recording(self) -> None:
        """
        Turn on whatever the executor has to collect for this plot.

        Most plots are drawn from data the executor keeps anyway, so nothing has to be
        turned on by default.
        """

    @abstractmethod
    def plot(self, goal_id: int) -> None:
        """
        Plot the data recorded for the goal with the given id.
        """

    def create_file_name(self, folder_name: str, goal_id: int) -> str:
        """
        Build the path of a per-goal pdf inside the temporary directory and make sure
        its folder exists.
        """
        file_name = os.path.join(
            tempfile.gettempdir(), folder_name, f"goal_{goal_id}.pdf"
        )
        create_path(file_name)
        return file_name


@dataclass
class GoalTrajectoryPlotter(PostGoalPlotter):
    """
    Plots the trajectory the robot followed.
    """

    trajectory_plotter: WorldStateTrajectoryPlotter = field(
        default_factory=WorldStateTrajectoryPlotter
    )
    """
    Collects the trajectory while the executor ticks; only recorded once
    :meth:`start_recording` handed it to the executor.
    """

    def start_recording(self) -> None:
        self.executor.trajectory_plotter = self.trajectory_plotter

    def plot(self, goal_id: int) -> None:
        if len(self.trajectory_plotter.world_state_trajectory.times) <= 1:
            return
        file_name = self.create_file_name("trajectories", goal_id)
        self.trajectory_plotter.plot_trajectory(file_name)
        rospy.node.get_logger().info(f"saved {file_name}")


@dataclass
class GoalGanttChartPlotter(PostGoalPlotter):
    """
    Plots when each motion statechart node was active.
    """

    second_length_in_cm: float = 1.5
    """
    Width of one second of motion in the chart.
    """

    def plot(self, goal_id: int) -> None:
        if not self.executor.motion_statechart.history:
            return
        file_name = self.create_file_name("gantt_charts", goal_id)
        self.executor.motion_statechart.plot_gantt_chart(
            file_name,
            context=self.executor.context,
            second_length_in_cm=self.second_length_in_cm,
        )
        rospy.node.get_logger().info(f"saved {file_name}")


@dataclass
class MotionStatechartPlotter(PostGoalPlotter):
    """
    Draws the structure of the motion statechart that was executed.
    """

    def plot(self, goal_id: int) -> None:
        file_name = self.create_file_name("motion_statecharts", goal_id)
        self.executor.motion_statechart.draw(file_name)
        rospy.node.get_logger().info(f"saved {file_name}")
