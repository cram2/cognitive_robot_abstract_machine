from dataclasses import dataclass, field
from uuid import UUID

import matplotlib
import numpy as np
from typing_extensions import List, Dict

# Force non-interactive backend to avoid GUI backend requirements in headless/test environments.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from ..spatial_types.derivatives import Derivatives
from .world_state import WorldStateTrajectory


@dataclass
class WorldStateTrajectoryPlotter:
    """
    Plot a trajectory of world states with one subplot per derivative.

    Subplots share the x-axis (time in seconds, normalized so that the first time maps to 0).
    """

    derivatives_to_plot: List[Derivatives] = field(
        default_factory=lambda: [
            Derivatives.position,
            Derivatives.velocity,
            Derivatives.acceleration,
            Derivatives.jerk,
        ]
    )
    """A plot will be generated for each entry in this list."""

    subplot_height_in_cm: float = 6.0
    """Height of each derivative subplot in cm."""

    legend: bool = True
    """If True, a legend will be added to the plot."""

    sort_degrees_of_freedom: bool = True
    """If True, the degrees of freedom will be sorted by name before plotting."""

    second_width_in_cm: float = 2.0
    """Width of a second in cm."""

    y_label: Dict[Derivatives, str] = field(
        default_factory=lambda: {
            Derivatives.position: "rad or m",
            Derivatives.velocity: "rad/s or m/s",
            Derivatives.acceleration: "rad/s² or m/s²",
            Derivatives.jerk: "rad/s³ or m/s³",
        }
    )
    """Label of the y-axis."""

    center_positions: bool = False
    """
    If True, the position plots will be centered, such that they start at 0 instead of the initial value.
    This may be useful if continues joints are used, because they can achieve very large values.
    """

    plot_constant_lines: bool = False
    """
    If False, lines of degrees of freedom that are constant 0 will not be plotted to reduce clutter.
    For positions this suppresses lines that are always equal to the initial value.
    """

    color_map: Dict[UUID, str] | None = None
    """
    A color map to use for plotting the trajectory.
    Use can use matplotlib styles, like 'r:' for dotted red lines.
    It should map the UUIDs of degrees of freedom in the world state trajectory to colors. 
    If None, a default color map will be used.
    
    Each degree of freedom will have the same color in each subplot.
    """

    def _seconds_to_inches(self, seconds: float) -> float:
        """
        Converts seconds to inches.
        :param seconds: Input duration in seconds.
        :return: The drawable width in inches for a given duration.
        """
        return max(0.0, float(seconds)) * (self.second_width_in_cm / 2.54)

    def _build_figure(
        self, duration: float, number_of_subplots: int
    ) -> tuple[plt.Figure, list[plt.Axes]]:
        """
        Create a stacked subplot figure sized so the inner axes width matches duration_s in physical units.
        :param duration: Duration of the trajectory in seconds.
        :param number_of_subplots: Number of subplots to create.
        :return: The figure and list of subplot axes.
        """
        # Inner drawable width must equal seconds-to-inches. Add margins around it in inches
        inner_w_in = max(0.0, self._seconds_to_inches(duration))
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

    def _choose_ticks(self, duration: float) -> np.ndarray:
        """
        Choose semi-automatic tick spacing: either 0.5s or integer seconds with adaptive spacing.
        :param duration: Duration of the trajectory in seconds.
        :return: x-ticks
        """
        if duration <= 0:
            return np.array([0.0])
        # Prefer 0.5 if it does not create too many ticks; otherwise use integers with adaptive step
        max_ticks = 12
        if duration / 0.5 <= max_ticks:
            step = 0.5
        else:
            # choose integer step so that count <= max_ticks
            step = max(1, int(np.ceil(duration / max_ticks)))
        return np.arange(0.0, duration + 1e-9, step)

    def _sort_degrees_of_freedom(self, traj: WorldStateTrajectory) -> List[UUID]:
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

    def _create_styles_for_degrees_of_freedom(
        self, dof_ids: List[UUID]
    ) -> Dict[UUID, dict]:
        """
        Assign styles to DOFs, respecting an optional color_map of matplotlib format strings.
        :param dof_ids: DOF ids for which to create styles.
        :return: Dict mapping DOF ids to matplotlib style dicts.
        """
        styles: Dict[UUID, dict] = {}
        if self.color_map is not None:
            for d in dof_ids:
                s = self.color_map.get(d)
                if s is not None:
                    styles[d] = {"fmt": s}
        # Default cycle without wrapping colors; then reuse with styles
        if len(styles) >= len(dof_ids):
            return styles
        prop_cycle = plt.rcParams.get("axes.prop_cycle")
        colors = list(prop_cycle.by_key().get("color", [])) if prop_cycle else []
        line_styles = ["-", "--", "-.", ":"]
        assigned = 0
        for ls in line_styles:
            for c in colors:
                if assigned >= len(dof_ids):
                    break
                d = dof_ids[assigned]
                if d not in styles:
                    styles[d] = {"color": c, "linestyle": ls}
                    assigned += 1
            if assigned >= len(dof_ids):
                break
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
        styles: Dict[UUID, dict],
        dof_ids: List[UUID],
        traj: WorldStateTrajectory,
        t: np.ndarray,
    ):
        """
        Actually plots the derivatives.

        :param axes: List of matplotlib Axes objects where the trajectory data
            for different derivatives will be plotted.
        :param styles: Dictionary mapping DOF UUIDs to plot style definitions
            (e.g., line formats). Key is a UUID, value is a dictionary with style
            attributes.
        :param dof_ids: List of UUIDs representing degrees of freedom for which
            trajectory data will be plotted.
        :param traj: WorldStateTrajectory object that contains trajectory data
            for plotting.
        :param t: Numpy array representing the time steps corresponding to the
            trajectory data.
        """
        for axis, derivative in zip(axes, self.derivatives_to_plot):
            axis.grid(True, axis="x")
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

    def plot_trajectory(
        self, world_state_trajectory: WorldStateTrajectory, file_name: str
    ):
        """
        Plots the trajectory and saves the resulting plot to the specified file.

        :param world_state_trajectory: The trajectory to plot.
        :param file_name: The name of the file where the plot will be saved.
        """
        traj = world_state_trajectory
        if len(traj.times) == 0:
            return
        t0 = float(traj.times[0])
        t = traj.times - t0
        duration_s = float(t[-1])

        figure, axes = self._build_figure(duration_s, len(self.derivatives_to_plot))

        dof_ids = self._sort_degrees_of_freedom(traj)
        styles = self._create_styles_for_degrees_of_freedom(dof_ids)

        # Plot per derivative
        self._plot_derivatives(axes, styles, dof_ids, traj, t)

        # Shared x axis settings (apply to last axis)
        ticks = self._choose_ticks(duration_s)
        axes[-1].set_xlim(0.0, duration_s)
        axes[-1].set_xticks(ticks)
        axes[-1].set_xticklabels(
            [str(int(x)) if abs(x - int(x)) < 1e-9 else f"{x:.1f}" for x in ticks]
        )
        axes[-1].set_xlabel("Time [s]")
        for axis in axes[:-1]:
            axis.set_xlabel("")

        if self.legend:
            self._create_legend(figure, axes)
        plt.savefig(file_name)
        plt.close()
