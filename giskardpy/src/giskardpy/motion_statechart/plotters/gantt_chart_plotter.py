from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from .styles import LiftCycleStateToColor, ObservationStateToColor
from ..context import BuildContext
from ..data_types import LifeCycleValues, ObservationStateValues
from ..graph_node import Goal, MotionStatechartNode
from ...middleware import get_middleware
from ...utils.utils import create_path

if TYPE_CHECKING:
    from ..motion_statechart import MotionStatechart


@dataclass
class HistoryGanttChartPlotter:
    """
    Plot a hierarchy of a MotionStatechart as a Gantt chart.

    Each node is represented by two horizontal bars:
        Top bar is life cycle state.
        Bottom bar is observation state.
    At the end, there is a final short bar plot highlighting the final state of each node,
    because they may be difficult to see otherwise.

    Shows parent-child relationships of Goals by ordering rows in
    preorder and by prefixing labels with tree glyphs (├─, └─, │).
    Optional background bands and goal outlines emphasize grouping.
    """

    motion_statechart: MotionStatechart
    """Plots history of this motion statechart."""
    context: BuildContext | None = None
    """
    Optional context to use for time conversion. 
    If not provided, control cycles are used instead of second.
    """
    second_width_in_cm: float = 2.0
    """Width of a second in cm."""
    final_state_band_height_in_cm: float = 0.5
    """Height of the final state band in cm."""

    @property
    def x_width_per_control_cycle(self) -> float:
        if self.context is None:
            return 1
        return self.context.qp_controller_config.control_dt

    @property
    def total_control_cycles(self) -> int:
        return self.motion_statechart.history.history[-1].control_cycle

    @property
    def num_bars(self) -> int:
        return len(self.motion_statechart.history.history[0].life_cycle_state)

    @property
    def use_seconds_for_x_axis(self) -> bool:
        return self.context is not None

    @property
    def figure_height(self) -> float:
        return 0.7 + self.num_bars * 0.25

    @property
    def figure_width(self) -> float:
        if not self.use_seconds_for_x_axis:
            return 0.5 * float((self.total_control_cycles or 0) + 1)
        # 1 inch = 2.54 cm; map seconds to figure width via second_length_in_cm
        inches_per_second = self.second_width_in_cm / 2.54
        return inches_per_second * self.time_span_seconds

    @property
    def time_span_seconds(self) -> float | None:
        return (
            self.total_control_cycles * self.x_width_per_control_cycle
            if self.x_width_per_control_cycle
            else None
        )

    def plot_gantt_chart(self, file_name: str) -> None:
        """
        Render the Gantt chart and save it.

        The chart shows life cycle (top half) and observation state (bottom half)
        per node over time. If a context with dt is provided, the x-axis is in seconds; otherwise, control cycles are used.

        This renders two side-by-side plots:
        - Left: the normal timeline over control cycles or seconds
        - Right: a compact column showing only the final state for each node, with the x label "final"
        Y-axis labels are shown only once on the right plot.

        :param file_name: File name to save the plot to.
        """

        nodes = self.motion_statechart.nodes
        if len(nodes) == 0:
            get_middleware().logwarn(
                "Gantt chart skipped: no nodes in motion statechart."
            )
            return

        history = self.motion_statechart.history.history
        if len(history) == 0:
            get_middleware().logwarn("Gantt chart skipped: empty StateHistory.")
            return

        ordered_nodes = self._sort_nodes_by_parents()

        ax_main, ax_final = self._build_subplots(ordered_nodes)

        for node_idx, node in enumerate(ordered_nodes):
            self._plot_lifecycle_bar(axis=ax_main, node=node, node_idx=node_idx)
            self._plot_observation_bar(axis=ax_main, node=node, node_idx=node_idx)
            # Draw the final-state-only blocks on the right axis
            self._plot_final_state_column(axis=ax_final, node=node, node_idx=node_idx)

        self._format_axes(
            main_axis=ax_main, final_state_axis=ax_final, ordered_nodes=ordered_nodes
        )
        self._save_figure(file_name=file_name)

    def _build_subplots(
        self, ordered_nodes: List[MotionStatechartNode]
    ) -> tuple[plt.Axes, plt.Axes]:
        """
        Builds a subplot layout with a main axis and a fixed-width final-state axis for the
        visualization of motion statechart nodes. Adaptively calculates layout dimensions,
        padding, and margins to ensure proper alignment and display of node labels.

        :param ordered_nodes: A list of MotionStatechartNode objects representing the nodes
            to be plotted.
        :type ordered_nodes: List[MotionStatechartNode]
        :return: A tuple containing the main axis and the fixed-width final-state axis.
        """

        # Build node label list early so we can size the right margin adaptively
        node_names: List[str] = []
        for idx, n in enumerate(ordered_nodes):
            prev_depth = 0 if idx == 0 else ordered_nodes[idx - 1].depth
            node_names.append(self._make_label(n, prev_depth))

        # Build figure so that axes widths are fixed in physical units (inches)
        # Main axis width = length_in_units * second_width_in_cm; Final axis width = fixed value independent of second_width_in_cm
        inches_per_unit = self.second_width_in_cm / 2.54
        length_in_units = (
            self.time_span_seconds
            if self.use_seconds_for_x_axis
            else self.total_control_cycles
        )
        main_w_inches = inches_per_unit * float(length_in_units)
        final_w_inches = (
            self.final_state_band_height_in_cm * inches_per_unit
        )  # inches, fixed
        pad_inches = 0.25
        # Base margins in inches
        left_margin_inches = 0.3
        bottom_margin_inches = 0.5
        top_margin_inches = 0.2

        # Measure required width for right-side y tick labels and set right margin adaptively
        labels_w_inches = self._measure_labels_width_in(node_names)
        label_pad_inches = 0.2
        right_margin_inches = max(0.8, labels_w_inches + label_pad_inches)

        fig_w_inches = (
            left_margin_inches
            + main_w_inches
            + pad_inches
            + final_w_inches
            + right_margin_inches
        )
        fig_h_inches = self.figure_height

        fig, ax_main = plt.subplots(
            figsize=(fig_w_inches, fig_h_inches), constrained_layout=False
        )
        # Apply margins explicitly
        fig.subplots_adjust(
            left=left_margin_inches / fig_w_inches,
            right=1 - right_margin_inches / fig_w_inches,
            bottom=bottom_margin_inches / fig_h_inches,
            top=1 - top_margin_inches / fig_h_inches,
        )

        # Compute inner area (after margins) and set main axis position to exact width
        inner_left = left_margin_inches / fig_w_inches
        inner_bottom = bottom_margin_inches / fig_h_inches
        inner_top = 1 - top_margin_inches / fig_h_inches
        inner_h_norm = inner_top - inner_bottom
        # Pre-allocate extra width equal to (final width + pad). axes_grid1 will
        # carve that space out from ax_main when appending the right axis, leaving
        # the main axis with exactly main_w_inches of drawable width.
        preallocated_main_w_inches = main_w_inches + final_w_inches + pad_inches
        main_w_norm_of_fig = preallocated_main_w_inches / fig_w_inches
        ax_main.set_position(
            [
                inner_left,
                inner_bottom,
                main_w_norm_of_fig,
                inner_h_norm,
            ]
        )

        ax_main.grid(True, axis="x", zorder=-1)

        # Append a fixed-width final-state axis on the right with a fixed pad
        divider = make_axes_locatable(ax_main)
        ax_final = divider.append_axes(
            "right",
            size=axes_size.Fixed(final_w_inches),
            pad=axes_size.Fixed(pad_inches),
            sharey=ax_main,
            axes_class=ax_main.__class__,
        )
        return ax_main, ax_final

    def _sort_nodes_by_parents(self) -> List[MotionStatechartNode]:
        """
        Sorts nodes of a motion statechart by their parent-child hierarchy.
        This method organizes nodes of the motion statechart such that child nodes
        appear directly after their respective parents in depth-first traversal.

        :return: A list of MotionStatechartNode objects ordered by their parent-child
            relationships in reversed order.
        """

        def return_children_in_order(n: MotionStatechartNode):
            yield n
            if isinstance(n, Goal):
                for c in n.nodes:
                    yield from return_children_in_order(c)

        ordered_: List[MotionStatechartNode] = []
        for root in self.motion_statechart.top_level_nodes:
            ordered_.extend(list(return_children_in_order(root)))
        # reverse list because plt plots bars bottom to top
        return list(reversed(ordered_))

    def _plot_lifecycle_bar(
        self,
        axis: plt.Axes,
        node: MotionStatechartNode,
        node_idx: int,
    ):
        """
        Plots the lifecycle bar for a given node onto the supplied Axes object. The plot
        visualizes the lifecycle state of the node as a bar in the context of control
        cycles. This method utilizes the lifecycle history of the node and maps the states
        to corresponding colors defined in the color map.

        :param axis: The matplotlib Axes on which to plot the lifecycle bar.
        :param node: The specific motion statechart node whose lifecycle is being plotted.
        :param node_idx: The index of the node being plotted in the node list.
        """
        life_cycle_history = (
            self.motion_statechart.history.get_life_cycle_history_of_node(node)
        )
        control_cycle_indices = [
            h.control_cycle for h in self.motion_statechart.history.history
        ]
        self._plot_node_bar(
            axis=axis,
            node_idx=node_idx,
            history=life_cycle_history,
            control_cycle_indices=control_cycle_indices,
            color_map=LiftCycleStateToColor,
            top=True,
        )

    def _plot_observation_bar(
        self,
        axis: plt.Axes,
        node: MotionStatechartNode,
        node_idx: int,
    ):
        """
        Plots the observation state bar for a given motion statechart node in the
        specified matplotlib Axes. The visualization represents the changes in observation states over
        control cycles, providing insights into the node's observation behavior
        over time.

        :param axis: The matplotlib Axes object where the observation bar will be plotted.
        :param node: The motion statechart node for which the observation history will
                     be represented.
        :param node_idx: Index of the node in the motion statechart used for positioning
                         the bar in the plot.
        """
        obs_history = self.motion_statechart.history.get_observation_history_of_node(
            node
        )
        control_cycle_indices = [
            h.control_cycle for h in self.motion_statechart.history.history
        ]
        self._plot_node_bar(
            axis=axis,
            node_idx=node_idx,
            history=obs_history,
            control_cycle_indices=control_cycle_indices,
            color_map=ObservationStateToColor,
            top=False,
        )

    def _plot_final_state_column(
        self,
        axis: plt.Axes,
        node: MotionStatechartNode,
        node_idx: int,
        column_padding: float = 0.1,
    ):
        """
        Draw the final state for both lifecycle (top half) and observation (bottom half)
        as a compact column on the right axes.

        :param axis: The matplotlib axis on which the final state column will be plotted.
        :param node: The motion statechart node whose final state is to be plotted.
        :param node_idx: The index of the node within the motion statechart.
        :param column_padding: The padding on each side of the column. Determines how far
            the edges of the column are from the axis boundaries. Default is 0.1.
        """
        # Determine last lifecycle and observation states
        life_cycle_history = (
            self.motion_statechart.history.get_life_cycle_history_of_node(node)
        )
        obs_history = self.motion_statechart.history.get_observation_history_of_node(
            node
        )
        last_lifecycle = life_cycle_history[-1]
        last_observation = obs_history[-1]

        # Column spans from padding to 1 - padding
        width = max(0.0, 1.0 - 2 * column_padding)
        start = column_padding

        # Draw top (lifecycle) and bottom (observation) halves
        self._draw_block(
            axis=axis,
            node_idx=node_idx,
            block_start=start,
            block_width=width,
            color=LiftCycleStateToColor[last_lifecycle],
            top=True,
        )
        self._draw_block(
            axis=axis,
            node_idx=node_idx,
            block_start=start,
            block_width=width,
            color=ObservationStateToColor[last_observation],
            top=False,
        )

    def _plot_node_bar(
        self,
        axis: plt.Axes,
        node_idx: int,
        history: List[LifeCycleValues | ObservationStateValues],
        control_cycle_indices: List[int],
        color_map: Dict[LifeCycleValues | ObservationStateValues, str],
        top: bool,
    ):
        """
        Plots a bar segment corresponding to the state changes of a node as per the history
        and its associated control cycle indices. Each state transition is represented as
        a colored block determined by the color mapping.

        :param axis: The matplotlib Axes instance where the bar will be plotted.
        :param node_idx: The index of the node for which the bar is being plotted.
        :param history: A list of state values indicating the historical lifecycle
            or observation state of the node.
        :param control_cycle_indices: A list of indices representing the control cycles
            associated with the state transitions.
        :param color_map: A mapping between lifecycle or observation states and their
            associated colors used for visualization.
        :param top: Indicates if the bar is to be plotted in the upper or lower part
            of the chart.
        """
        current_state = history[0]
        start_idx = 0
        for idx, next_state in zip(control_cycle_indices[1:], history[1:]):
            if current_state != next_state:
                life_cycle_width = (idx - start_idx) * self.x_width_per_control_cycle
                self._draw_block(
                    axis=axis,
                    node_idx=node_idx,
                    block_start=start_idx * self.x_width_per_control_cycle,
                    block_width=life_cycle_width,
                    color=color_map[current_state],
                    top=top,
                )
                start_idx = idx
                current_state = next_state
        # plot last stretch until final index
        last_idx = control_cycle_indices[-1]
        life_cycle_width = (last_idx - start_idx) * self.x_width_per_control_cycle
        self._draw_block(
            axis=axis,
            node_idx=node_idx,
            block_start=start_idx * self.x_width_per_control_cycle,
            block_width=life_cycle_width,
            color=color_map[current_state],
            top=top,
        )

    def _draw_block(
        self,
        axis: plt.Axes,
        node_idx,
        block_start,
        block_width,
        color,
        top: bool,
        bar_height: float = 0.8,
    ):
        """
        Draws a block in a horizontal bar chart with specified parameters.

        :param axis: The matplotlib Axes object where the block will be drawn.
        :param node_idx: The y-axis index position of the node on the chart.
        :param block_start: The starting position of the block along the x-axis.
        :param block_width: The width of the block along the x-axis.
        :param color: The fill color of the block.
        :param top: A flag indicating whether the block should be positioned
            above the baseline (True) or below it (False).
        :param bar_height: The total height of the bar containing the block.
            Defaults to 0.8.
        """
        if top:
            y = node_idx + bar_height / 4
        else:
            y = node_idx - bar_height / 4
        axis.barh(
            y,
            block_width,
            height=bar_height / 2,
            left=block_start,
            color=color,
            zorder=2,
        )

    def _format_axes(
        self,
        main_axis: plt.Axes,
        final_state_axis: plt.Axes,
        ordered_nodes: List[MotionStatechartNode],
    ):
        """
        Configure and format axes for visualizing motion statechart nodes.

        This function modifies the provided matplotlib axes to display a timeline
        and statechart information for a motion control simulation or experiment.
        Additionally, it prepares the axes to show details such as time units,
        control cycle labels, final-state configurations, and node-specific
        labels, enabling clear visual representation of the motion statechart.

        :param main_axis: Matplotlib Axes object used for the main timeline display.
        :param final_state_axis: Matplotlib Axes object used for the final-state
                                 representation.
        :param ordered_nodes: List of MotionStatechartNode objects to determine
                              the y-axis labels and structure.
        """
        # Configure x-axis for main timeline
        if self.use_seconds_for_x_axis:
            main_axis.set_xlabel("Time [s]")
            base_ticks = np.arange(0.0, self.time_span_seconds + 1e-9, 0.5).tolist()
            main_axis.set_xlim(0, self.time_span_seconds)
        else:
            main_axis.set_xlabel("Control cycle")
            step = max(int(self.x_width_per_control_cycle), 1)
            base_ticks = list(range(0, self.total_control_cycles + 1, step))
            main_axis.set_xlim(0, self.total_control_cycles)
        main_axis.set_xticks(base_ticks)
        main_axis.set_xticklabels([str(t) for t in base_ticks])

        # Configure final-state column x-axis
        final_state_axis.set_xlim(0.0, 1.0)
        final_state_axis.set_xticks([0.5])
        final_state_axis.set_xticklabels(["final"])
        final_state_axis.grid(False)

        # Y axis labels and limits shown only on the right (final column)
        ymin, ymax = -0.8, self.num_bars - 1 + 0.8
        final_state_axis.set_ylim(ymin, ymax)

        node_names = []
        for idx, n in enumerate(ordered_nodes):
            prev_depth = 0 if idx == 0 else ordered_nodes[idx - 1].depth
            node_names.append(self._make_label(n, prev_depth))
        node_idx = list(range(len(node_names)))

        # Hide y ticks on main axis but keep the shared tick locations intact
        main_axis.tick_params(axis="y", left=False, labelleft=False)
        main_axis.set_ylabel("Nodes")

        # Put all y tick labels on the right (final axis)
        final_state_axis.set_yticks(node_idx)
        final_state_axis.set_yticklabels(node_names)
        final_state_axis.set_ylabel("")
        final_state_axis.yaxis.set_ticks_position("right")
        final_state_axis.tick_params(
            axis="y", right=True, labelright=True, left=False, labelleft=False
        )

    def _make_label(self, node: MotionStatechartNode, prev_depth: int) -> str:
        """
        Generates a formatted label for a given node in a motion statechart by incorporating
        its depth and using ASCII art for hierarchical representation.

        :param node: The motion statechart node for which the label is created.
        :param prev_depth: The depth of the previously processed node in the hierarchy.
        :return: A string representing the hierarchical label of the node.
        """
        depth = node.depth
        if depth == 0:
            return node.unique_name
        diff = depth - prev_depth
        if diff > 0:
            return (
                "│  " * (depth - diff)
                + "└─"  # no space because the formatting is weird otherwise
                * (diff - 1)
                + "└─ "
                + node.unique_name
            )
        else:
            return "│  " * (depth - 1) + "├─ " + node.unique_name

    def _measure_labels_width_in(self, labels: List[str]) -> float:
        """
        Measures the maximum width of a list of text labels when rendered in a temporary
        figure. This is useful to determine the required space for labels in a plot.

        :param labels: A list of strings representing the text labels whose widths need
            to be measured.
        :return: The maximum width of the rendered labels in the figure's DPI units.
        """
        # Use a small temporary figure
        fig = plt.figure(figsize=(2, 2))
        try:
            # Ensure renderer exists
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            max_width_px = 0.0
            temp_texts = []
            for s in labels:
                t = fig.text(0, 0, s)
                temp_texts.append(t)
                bbox = t.get_window_extent(renderer=renderer)
                if bbox.width > max_width_px:
                    max_width_px = bbox.width
            for t in temp_texts:
                t.remove()
            return max_width_px / fig.dpi if fig.dpi else 0.0
        finally:
            plt.close(fig)

    def _save_figure(self, file_name: str) -> None:
        """
        Saves the current figure to the specified file.

        :param file_name: The complete path and file name where the figure
            should be saved.
        """
        create_path(file_name)
        plt.savefig(file_name)
        plt.close()
        get_middleware().loginfo(f"Saved gantt chart to {file_name}.")
