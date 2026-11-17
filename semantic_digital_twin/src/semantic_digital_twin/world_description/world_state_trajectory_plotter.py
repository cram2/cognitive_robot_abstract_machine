"""
Plots how the degrees of freedom of a world moved over a recorded trajectory.
"""

import itertools
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from uuid import UUID

import matplotlib
import numpy as np


# Force non-interactive backend to avoid GUI backend requirements in headless/test environments.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from semantic_digital_twin import logger
from semantic_digital_twin.world_description.derivatives import Derivatives
from semantic_digital_twin.world_description.world_state import (
    WorldStateTrajectory,
    WorldState,
)

# %% spacing of the time axis


@dataclass(frozen=True, kw_only=True)
class TickSpacing:
    """
    Where the time axis of a trajectory plot puts its ticks and grid lines.
    """

    seconds_between_numbered_ticks: float
    """
    The time between two ticks that carry a number.
    """

    seconds_between_grid_lines: float
    """
    The time between two vertical grid lines; those between the numbered ticks carry no
    number.
    """


class TimeAxisTickSpacing(Enum):
    """
    The spacings a trajectory plot chooses its time axis from, finest first.
    """

    HALF_SECOND = TickSpacing(
        seconds_between_numbered_ticks=0.5, seconds_between_grid_lines=0.1
    )
    ONE_SECOND = TickSpacing(
        seconds_between_numbered_ticks=1.0, seconds_between_grid_lines=0.2
    )
    TWO_SECONDS = TickSpacing(
        seconds_between_numbered_ticks=2.0, seconds_between_grid_lines=0.5
    )
    FIVE_SECONDS = TickSpacing(
        seconds_between_numbered_ticks=5.0, seconds_between_grid_lines=1.0
    )
    TEN_SECONDS = TickSpacing(
        seconds_between_numbered_ticks=10.0, seconds_between_grid_lines=2.0
    )
    TWENTY_SECONDS = TickSpacing(
        seconds_between_numbered_ticks=20.0, seconds_between_grid_lines=5.0
    )
    THIRTY_SECONDS = TickSpacing(
        seconds_between_numbered_ticks=30.0, seconds_between_grid_lines=10.0
    )
    ONE_MINUTE = TickSpacing(
        seconds_between_numbered_ticks=60.0, seconds_between_grid_lines=10.0
    )


# %% plotting a trajectory


@dataclass
class WorldStateTrajectoryPlotter:
    """
    Plot a trajectory of world states with one subplot per derivative.

    Subplots share the x-axis (time in seconds, normalized so that the first time maps
    to 0).
    """

    world_state_trajectory: WorldStateTrajectory = field(init=False)
    """
    The trajectory of the robot's world state.
    """

    derivatives_to_plot: list[Derivatives] = field(
        default_factory=lambda: [
            Derivatives.position,
            Derivatives.velocity,
            Derivatives.acceleration,
            Derivatives.jerk,
        ]
    )
    """
    A plot will be generated for each entry in this list.
    """

    subplot_height_in_cm: float = 6.0
    """
    Height of each derivative subplot in cm.
    """

    legend: bool = True
    """
    If True, a legend will be added to the plot.
    """

    sort_degrees_of_freedom: bool = True
    """
    If True, the degrees of freedom will be sorted by name before plotting.
    """

    second_width_in_cm: float = 0.5
    """
    Width of a second in cm, unless the plot would be narrower than
    :attr:`minimum_plot_width_in_cm`.
    """

    minimum_plot_width_in_cm: float = 10.0
    """
    The narrowest the subplots are drawn; a shorter trajectory gets wider seconds.
    """

    minimum_distance_between_numbered_ticks_in_cm: float = 2.5
    """
    How close two numbered ticks of the time axis may come; the axis takes the finest
    :class:`TimeAxisTickSpacing` that keeps them this far apart.
    """

    unnumbered_grid_line_width: float = 0.4
    """
    The width, in points, of a vertical grid line without a numbered tick.
    """

    unnumbered_grid_line_opacity: float = 0.4
    """
    The opacity of a vertical grid line without a numbered tick, from 0 to 1.
    """

    y_label: dict[Derivatives, str] = field(
        default_factory=lambda: {
            Derivatives.position: "rad or m",
            Derivatives.velocity: "rad/s or m/s",
            Derivatives.acceleration: "rad/s² or m/s²",
            Derivatives.jerk: "rad/s³ or m/s³",
        }
    )
    """
    Label of the y-axis.
    """

    center_positions: bool = False
    """
    If True, the position plots will be centered, such that they start at 0 instead of
    the initial value.

    This may be useful if continues joints are used, because they can achieve very large
    values.
    """

    plot_constant_lines: bool = False
    """
    If False, lines of degrees of freedom that are constant 0 will not be plotted to
    reduce clutter.

    For positions this suppresses lines that are always equal to the initial value.
    """

    line_styles: list[str] = field(default_factory=lambda: ["-", "--", "-.", ":"])
    """
    The matplotlib line styles that tell lines of the same color apart, each used with
    every color before the next.
    """

    color_map: dict[UUID, str] | None = None
    """
    A color map to use for plotting the trajectory.

    Use can use matplotlib styles, like 'r:' for dotted red lines. It should map the
    UUIDs of degrees of freedom in the world state trajectory to colors. If None, a
    default color map will be used.

    Each degree of freedom will have the same color in each subplot.
    """

    def reset(self, initial_state: WorldState, time: float = 0.0):
        self.world_state_trajectory = WorldStateTrajectory.from_world_state(
            state=initial_state, time=time
        )

    def _widened_second_width_in_cm(self, *, duration: float) -> float:
        """
        :return: The width of a second in a plot of `duration` seconds:
            :attr:`second_width_in_cm`, widened until the plot is
            :attr:`minimum_plot_width_in_cm` wide.
        """
        return max(self.second_width_in_cm, self.minimum_plot_width_in_cm / duration)

    def _tick_spacing(self, *, duration: float) -> TickSpacing:
        """
        :return: The finest spacing that keeps the numbered ticks of a plot of
            `duration` seconds :attr:`minimum_distance_between_numbered_ticks_in_cm`
            apart, or the coarsest one if none does.
        """
        second_width_in_cm = self._widened_second_width_in_cm(duration=duration)
        spacings = [spacing.value for spacing in TimeAxisTickSpacing]
        return next(
            (
                spacing
                for spacing in spacings
                if spacing.seconds_between_numbered_ticks * second_width_in_cm
                >= self.minimum_distance_between_numbered_ticks_in_cm
            ),
            spacings[-1],
        )

    def _build_figure(
        self, *, duration: float, number_of_subplots: int
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Create a stacked subplot figure sized so the inner axes width matches duration_s
        in physical units.

        :param duration: Duration of the trajectory in seconds.
        :param number_of_subplots: Number of subplots to create.
        :return: The figure and list of subplot axes.
        """
        # Add margins in inches around the drawable width.
        inner_w_in = (
            duration * self._widened_second_width_in_cm(duration=duration) / 2.54
        )
        # Margins tuned to avoid clipping of yticks, ylabel; right margin reserved for legend
        left_margin_in = 0.8
        right_margin_in = 2.0
        top_margin_in = 0.3
        bottom_margin_in = 1.0
        # store margins for potential dynamic adjustment during legend placement
        self._margins = {
            "left": left_margin_in,
            "right": right_margin_in,
            "top": top_margin_in,
            "bottom": bottom_margin_in,
        }
        fig_w_in = inner_w_in + left_margin_in + right_margin_in
        # Stack subplots vertically: add top/bottom margins to the inner subplot heights
        inner_h_in = max(1.0, number_of_subplots * (self.subplot_height_in_cm / 2.54))
        fig_h_in = inner_h_in + top_margin_in + bottom_margin_in
        fig, axes = plt.subplots(
            nrows=number_of_subplots,
            ncols=1,
            sharex=True,
            figsize=(fig_w_in, fig_h_in),
            constrained_layout=False,
        )
        # position the subplots so that the inner width is exactly inner_w_in inches
        left = left_margin_in / fig_w_in
        right = 1.0 - (right_margin_in / fig_w_in)
        bottom = bottom_margin_in / fig_h_in
        top = 1.0 - (top_margin_in / fig_h_in)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        # store layout metadata for potential dynamic adjustments later
        self._layout_meta = {
            "inner_w_in": inner_w_in,
            "fig_h_in": fig_h_in,
        }
        if number_of_subplots == 1:
            axes = [axes]
        # Set explicit normalized margins so that inner area keeps the intended physical width
        # (Already applied above via fig.subplots_adjust)
        return fig, axes

    def _times_up_to(self, *, duration: float, step: float) -> np.ndarray:
        """
        :return: The times from 0 to `duration` seconds, `step` seconds apart, including
            `duration` itself when it falls on a step.
        """
        return np.arange(0.0, duration + 1e-9, step)

    def _sort_degrees_of_freedom(self, traj: WorldStateTrajectory) -> list[UUID]:
        """
        Return DOF ids ordered based on self.sort_degrees_of_freedom.

        :param traj: The trajectory to sort.
        :return: Degrees of freedom, either sorted by name or in original order
        """
        dof_ids = list(traj._ids)
        if self.sort_degrees_of_freedom:
            dof_ids.sort(
                key=lambda d: str(traj.world.get_degree_of_freedom_by_id(d).name)
            )
        return dof_ids

    def _should_plot_series(self, series: np.ndarray) -> bool:
        """
        Decide whether to plot a DOF series according to suppression rules.

        :param derivative: Derivative of the subplot
        :param series: Data to be plotted.
        :return: whether to plot the series.
        """
        if self.plot_constant_lines:
            return True
        return not np.all(series == float(series[0]))

    def _is_drawn(
        self, *, trajectory: WorldStateTrajectory, degree_of_freedom_id: UUID
    ) -> bool:
        """
        :return: Whether any subplot draws the degree of freedom with
            `degree_of_freedom_id`.
        """
        column = trajectory._index[degree_of_freedom_id]
        return any(
            self._should_plot_series(trajectory.data[:, derivative, column])
            for derivative in self.derivatives_to_plot
        )

    def _create_styles_for_degrees_of_freedom(
        self, *, degree_of_freedom_ids: list[UUID]
    ) -> dict[UUID, dict]:
        """
        Give each degree of freedom the format :attr:`color_map` assigns it, or else the
        next unused combination of a color and one of :attr:`line_styles`; the
        combinations repeat only once all are taken.

        :return: The matplotlib style of each of `degree_of_freedom_ids`.
        """
        color_map = self.color_map or {}
        styles = {
            degree_of_freedom_id: {"fmt": color_map[degree_of_freedom_id]}
            for degree_of_freedom_id in degree_of_freedom_ids
            if degree_of_freedom_id in color_map
        }
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        combinations = itertools.cycle(itertools.product(self.line_styles, colors))
        for degree_of_freedom_id in degree_of_freedom_ids:
            if degree_of_freedom_id in styles:
                continue
            line_style, color = next(combinations)
            styles[degree_of_freedom_id] = {"color": color, "linestyle": line_style}
        return styles

    def _create_legend(self, figure: plt.Figure, axes: list[plt.Axes]):
        # Place a single shared legend at the top-right, outside the plotting area but inside the figure.
        handles, labels = axes[-1].get_legend_handles_labels()
        if not handles:
            return
        # First, create a temporary legend to measure its size
        temp_leg = figure.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            ncol=1,
            frameon=True,
            borderaxespad=0.0,
        )
        try:
            figure.canvas.draw()
            bbox = temp_leg.get_window_extent(figure.canvas.get_renderer())
            legend_w_in = bbox.width / float(figure.dpi)
        except Exception:
            legend_w_in = 0.0
        # Compute if we need more right margin to avoid overlap with axes area
        right_margin_in = float(self._margins.get("right", 2.0))
        left_margin_in = float(self._margins.get("left", 0.8))
        top_margin_in = float(self._margins.get("top", 0.3))
        bottom_margin_in = float(self._margins.get("bottom", 1.0))
        inner_w_in = float(self._layout_meta.get("inner_w_in", 0.0))
        fig_h_in = float(self._layout_meta.get("fig_h_in", figure.get_figheight()))
        pad_in = 0.2
        required_right_in = legend_w_in + pad_in
        if required_right_in > right_margin_in and inner_w_in > 0.0:
            # Expand figure width to fit legend while preserving inner drawable width
            delta = required_right_in - right_margin_in
            new_fig_w_in = inner_w_in + left_margin_in + right_margin_in + delta
            figure.set_size_inches(new_fig_w_in, fig_h_in)
            # Update subplot positions to keep inner width constant
            left = left_margin_in / new_fig_w_in
            right = 1.0 - ((right_margin_in + delta) / new_fig_w_in)
            bottom = bottom_margin_in / fig_h_in
            top = 1.0 - (top_margin_in / fig_h_in)
            figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
            # Update stored margins
            self._margins["right"] = right_margin_in + delta
            # Remove the temporary legend and create the final one after resize
            try:
                temp_leg.remove()
            except Exception:
                pass
            figure.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.99),
                ncol=1,
                frameon=True,
                borderaxespad=0.0,
            )

    def _plot_derivatives(
        self,
        axes: list[plt.Axes],
        styles: dict[UUID, dict],
        dof_ids: list[UUID],
        traj: WorldStateTrajectory,
        t: np.ndarray,
    ):
        """
        Actually plots the derivatives.

        :param axes: List of matplotlib Axes objects where the trajectory data for
            different derivatives will be plotted.
        :param styles: Dictionary mapping DOF UUIDs to plot style definitions (e.g.,
            line formats). Key is a UUID, value is a dictionary with style attributes.
        :param dof_ids: List of UUIDs representing degrees of freedom for which
            trajectory data will be plotted.
        :param traj: WorldStateTrajectory object that contains trajectory data for
            plotting.
        :param t: Numpy array representing the time steps corresponding to the
            trajectory data.
        """
        for axis, derivative in zip(axes, self.derivatives_to_plot):
            axis.grid(True, axis="x")
            axis.grid(
                True,
                axis="x",
                which="minor",
                linewidth=self.unnumbered_grid_line_width,
                alpha=self.unnumbered_grid_line_opacity,
            )
            axis.grid(True, axis="y")
            # Collect series for all DOFs
            for dof_id in dof_ids:
                col = traj._index[dof_id]
                y = traj.data[:, derivative, col].astype(float)
                if derivative == Derivatives.position and self.center_positions:
                    y = y - float(y[0])
                if not self._should_plot_series(y):
                    continue
                style = styles.get(dof_id, {})
                name = str(traj.world.get_degree_of_freedom_by_id(dof_id).name)
                # Support format strings
                if "fmt" in style:
                    axis.plot(t, y, style["fmt"], label=name)
                else:
                    axis.plot(t, y, label=name, **style)
            # Axis formatting
            axis.set_title(str(derivative.name).capitalize())
            axis.set_ylabel(self.y_label.get(derivative, ""))

    @property
    def has_trajectory_to_plot(self) -> bool:
        """
        Whether the recorded trajectory spans time, which a plot needs; a single state
        does not.
        """
        return len(self.world_state_trajectory.times) > 1

    def plot_trajectory(self, *, file_name: Path) -> None:
        """
        Plot the recorded trajectory, one subplot per derivative in
        :attr:`derivatives_to_plot`, save it to `file_name` and log where; nothing is
        plotted unless :attr:`has_trajectory_to_plot`.

        :param file_name: The path the plot is saved to.
        """
        if not self.has_trajectory_to_plot:
            return
        trajectory = self.world_state_trajectory
        times = trajectory.times - float(trajectory.times[0])
        duration = float(times[-1])

        figure, axes = self._build_figure(
            duration=duration, number_of_subplots=len(self.derivatives_to_plot)
        )

        degree_of_freedom_ids = [
            degree_of_freedom_id
            for degree_of_freedom_id in self._sort_degrees_of_freedom(trajectory)
            if self._is_drawn(
                trajectory=trajectory, degree_of_freedom_id=degree_of_freedom_id
            )
        ]
        styles = self._create_styles_for_degrees_of_freedom(
            degree_of_freedom_ids=degree_of_freedom_ids
        )
        self._plot_derivatives(axes, styles, degree_of_freedom_ids, trajectory, times)

        spacing = self._tick_spacing(duration=duration)
        ticks = self._times_up_to(
            duration=duration, step=spacing.seconds_between_numbered_ticks
        )
        axes[-1].set_xlim(0.0, duration)
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticks(
            self._times_up_to(
                duration=duration, step=spacing.seconds_between_grid_lines
            ),
            minor=True,
        )
        axes[-1].set_xticklabels(
            [str(int(x)) if abs(x - int(x)) < 1e-9 else f"{x:.1f}" for x in ticks]
        )
        axes[-1].set_xlabel("Time [s]")
        for axis in axes[:-1]:
            axis.set_xlabel("")

        if self.legend:
            self._create_legend(figure, axes)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_name)
        plt.close()
        logger.info(f"Saved the trajectory plot to {file_name}.")
