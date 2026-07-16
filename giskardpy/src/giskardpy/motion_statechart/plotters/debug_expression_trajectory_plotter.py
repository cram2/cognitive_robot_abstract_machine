from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib
import numpy as np
from typing_extensions import List

# Force non-interactive backend to avoid GUI backend requirements in headless/test environments.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from giskardpy.motion_statechart.debug_expression_trajectory import (
    DebugExpressionTrajectory,
    RecordedDebugExpression,
)
from giskardpy.motion_statechart.graph_node import DebugExpression


@dataclass
class DebugExpressionTrajectoryPlotter:
    """
    Plot how the debug expressions of a motion statechart evolved over time.

    Produces one subplot per recorded debug expression, sharing a common time x-axis.
    Each scalar component of an expression (for example the x, y, z of a point) is drawn
    as its own labelled line.
    """

    debug_expression_trajectory: DebugExpressionTrajectory = field(init=False)
    """
    The recorded debug expression time series to plot.
    """

    subplot_height_in_cm: float = 6.0
    """
    Height of each debug expression subplot in cm.
    """

    second_width_in_cm: float = 2.0
    """
    Width of a second in cm.
    """

    legend: bool = True
    """
    If True, a legend is added to each subplot.
    """

    def reset(self, debug_expressions: List[DebugExpression]) -> None:
        """
        Prepare to record the given debug expressions, discarding any previous data.
        """
        self.debug_expression_trajectory = (
            DebugExpressionTrajectory.from_debug_expressions(debug_expressions)
        )

    def _seconds_to_inches(self, seconds: float) -> float:
        """
        Convert a duration in seconds to the drawable width in inches.
        """
        return max(0.0, float(seconds)) * (self.second_width_in_cm / 2.54)

    def _build_figure(
        self, duration: float, number_of_subplots: int
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Create a stacked subplot figure sized so the axes width matches the duration.

        :param duration: Duration of the trajectory in seconds.
        :param number_of_subplots: Number of subplots to create.
        :return: The figure and list of subplot axes.
        """
        inner_width_in = max(1.0, self._seconds_to_inches(duration))
        left_margin_in = 0.8
        right_margin_in = 2.0
        top_margin_in = 0.3
        bottom_margin_in = 1.0
        figure_width_in = inner_width_in + left_margin_in + right_margin_in
        inner_height_in = max(
            1.0, number_of_subplots * (self.subplot_height_in_cm / 2.54)
        )
        figure_height_in = inner_height_in + top_margin_in + bottom_margin_in
        figure, axes = plt.subplots(
            nrows=number_of_subplots,
            ncols=1,
            sharex=True,
            figsize=(figure_width_in, figure_height_in),
            constrained_layout=False,
        )
        figure.subplots_adjust(
            left=left_margin_in / figure_width_in,
            right=1.0 - (right_margin_in / figure_width_in),
            bottom=bottom_margin_in / figure_height_in,
            top=1.0 - (top_margin_in / figure_height_in),
        )
        if number_of_subplots == 1:
            axes = [axes]
        return figure, axes

    def _choose_ticks(self, duration: float) -> np.ndarray:
        """
        Choose tick spacing: either 0.5s or integer seconds with adaptive spacing.

        :param duration: Duration of the trajectory in seconds.
        :return: The x-ticks.
        """
        if duration <= 0:
            return np.array([0.0])
        max_ticks = 12
        if duration / 0.5 <= max_ticks:
            step = 0.5
        else:
            step = max(1, int(np.ceil(duration / max_ticks)))
        return np.arange(0.0, duration + 1e-9, step)

    def _plot_recorded_debug_expression(
        self,
        axis: plt.Axes,
        recorded_debug_expression: RecordedDebugExpression,
        time: np.ndarray,
    ) -> None:
        """
        Plot every scalar component of one debug expression onto the given axis.

        :param axis: The axis the debug expression is plotted on.
        :param recorded_debug_expression: The recorded debug expression to plot.
        :param time: The time values used for the x-axis.
        """
        values = recorded_debug_expression.values
        name = recorded_debug_expression.name
        color = recorded_debug_expression.debug_expression.color
        number_of_components = values.shape[1]
        for component_index in range(number_of_components):
            if number_of_components == 1:
                axis.plot(
                    time,
                    values[:, component_index],
                    label=name,
                    color=(color.R, color.G, color.B, color.A),
                )
            else:
                axis.plot(
                    time, values[:, component_index], label=f"{name}[{component_index}]"
                )
        axis.grid(True)
        axis.set_title(name)
        if self.legend:
            axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))

    def plot(self, file_name: str) -> None:
        """
        Plot the recorded debug expressions and save the result to a PDF file.

        :param file_name: The path where the plot is saved.
        """
        trajectory = self.debug_expression_trajectory
        recorded = [
            recorded_debug_expression
            for recorded_debug_expression in trajectory.recorded_debug_expressions
            if len(recorded_debug_expression.values) > 0
        ]
        if len(trajectory.times) == 0 or len(recorded) == 0:
            return
        time = trajectory.times - float(trajectory.times[0])
        duration = float(time[-1])

        figure, axes = self._build_figure(duration, len(recorded))
        for axis, recorded_debug_expression in zip(axes, recorded):
            self._plot_recorded_debug_expression(axis, recorded_debug_expression, time)

        ticks = self._choose_ticks(duration)
        axes[-1].set_xlim(0.0, max(duration, 1e-9))
        axes[-1].set_xticks(ticks)
        axes[-1].set_xlabel("Time [s]")

        plt.savefig(file_name, bbox_inches="tight")
        plt.close()
