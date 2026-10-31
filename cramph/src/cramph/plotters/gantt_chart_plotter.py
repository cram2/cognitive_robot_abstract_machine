from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.node import StatechartNode
from semantic_digital_twin.world_description.geometry import Color

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cramph.statechart import Statechart


# %% bar data


@dataclass
class StateSpan:
    """
    A stretch of ticks over which a node kept one state.
    """

    state: LifeCycleValues | ObservationStateValues
    """
    The state the node kept.
    """

    start_tick: int
    """
    The tick of the first snapshot showing the node in :attr:`state`.
    """

    end_tick: int
    """
    The tick of the snapshot showing the node's next state, or the last recorded tick if
    the node kept :attr:`state` until then.
    """


# %% plotter


@dataclass
class HistoryGanttChartPlotter:
    """
    Plot a hierarchy of a Statechart as a Gantt chart.

    Each node is represented by two horizontal bars:
        Top bar is life cycle state.
        Bottom bar is observation state.
    At the end, there is a final short bar plot highlighting the final state of each node,
    because they may be difficult to see otherwise.

    Shows parent-child relationships of Goals by ordering rows in
    preorder and by prefixing labels with tree glyphs (├─, └─, │).
    Optional background bands and goal outlines emphasize grouping.
    """

    statechart: Statechart
    """
    Plots history of this statechart.
    """

    context: StatechartContext | None = None
    """
    Optional context to use for time conversion.

    If not provided, ticks are used instead of seconds.
    """

    second_width_in_cm: float = 2.0
    """
    Width of a second in cm.
    """

    final_state_band_height_in_cm: float = 0.5
    """
    Height of the final state band in cm.
    """

    gap_between_axes_in_inches: float = 0.25
    """
    Horizontal gap between the main timeline and the final state column.
    """

    @property
    def x_width_per_tick(self) -> float:
        """
        :return: The width of one tick on the time axis, which is the tick duration of
            the context, or 1 without a context.
        """
        if self.context is None:
            return 1
        return self.context.require_tick_duration()

    @property
    def total_ticks(self) -> int:
        """
        :return: The number of ticks the history of the statechart covers.
        """
        return self.statechart.state_history.latest_item.tick_count

    @property
    def num_bars(self) -> int:
        """
        :return: The number of rows of the chart, one per node of the statechart.
        """
        return len(self.statechart.nodes)

    def life_cycle_spans(self, node: StatechartNode) -> list[StateSpan]:
        """
        :param node: The node to look up.
        :return: The life cycle states `node` went through since it joined the
            statechart, in order; empty if no snapshot records it.
        """
        snapshots = self.statechart.state_history.items_recording(node)
        return self._merge_into_spans(
            ticks=[snapshot.tick_count for snapshot in snapshots],
            states=[snapshot.life_cycle_state[node] for snapshot in snapshots],
        )

    def observation_spans(self, node: StatechartNode) -> list[StateSpan]:
        """
        :param node: The node to look up.
        :return: The observation states `node` went through since it joined the
            statechart, in order; empty if no snapshot records it.
        """
        snapshots = self.statechart.state_history.items_recording(node)
        return self._merge_into_spans(
            ticks=[snapshot.tick_count for snapshot in snapshots],
            states=[snapshot.observation_state[node] for snapshot in snapshots],
        )

    def _merge_into_spans(
        self,
        ticks: list[int],
        states: list[LifeCycleValues] | list[ObservationStateValues],
    ) -> list[StateSpan]:
        """
        :param ticks: The tick of every snapshot recording a node, in order.
        :param states: The state of that node in each of those snapshots.
        :return: One span per run of equal consecutive states, the last one reaching
            the last recorded tick.
        """
        change_indices = [
            index
            for index in range(len(states))
            if index == 0 or states[index] != states[index - 1]
        ]
        end_ticks = [ticks[index] for index in change_indices[1:]] + [self.total_ticks]
        return [
            StateSpan(state=states[index], start_tick=ticks[index], end_tick=end_tick)
            for index, end_tick in zip(change_indices, end_ticks)
        ]

    @property
    def use_seconds_for_x_axis(self) -> bool:
        return self.context is not None

    @property
    def figure_height(self) -> float:
        return 0.7 + self.num_bars * 0.25

    @property
    def figure_width(self) -> float:
        if not self.use_seconds_for_x_axis:
            return 0.5 * float((self.total_ticks or 0) + 1)
        # 1 inch = 2.54 cm; map seconds to figure width via second_length_in_cm
        inches_per_second = self.second_width_in_cm / 2.54
        return inches_per_second * self.time_span_seconds

    @property
    def time_span_seconds(self) -> float | None:
        return (
            self.total_ticks * self.x_width_per_tick if self.x_width_per_tick else None
        )

    def plot_gantt_chart(self, file_name: str) -> None:
        """
        Render the Gantt chart and save it.

        The chart shows life cycle (top half) and observation state (bottom half)
        per node over time. If a context is provided, the x-axis is in seconds; otherwise, ticks are used.

        This renders two side-by-side plots:
        - Left: the normal timeline over ticks or seconds
        - Right: a compact column showing only the final state for each node, with the x label "final"
        Y-axis labels are shown only once on the right plot.

        :param file_name: File name to save the plot to.
        """
        nodes = self.statechart.nodes
        if len(nodes) == 0:
            logger.warning("Gantt chart skipped: no nodes in statechart.")
            return

        if self.statechart.state_history.latest_item is None:
            logger.warning("Gantt chart skipped: empty StateHistory.")
            return

        ordered_nodes = self._sort_nodes_by_parents()

        ax_main, ax_final = self._build_subplots(ordered_nodes)

        for node_idx, node in enumerate(ordered_nodes):
            self._plot_spans(
                axis=ax_main,
                node_idx=node_idx,
                spans=self.life_cycle_spans(node),
                top=True,
            )
            self._plot_spans(
                axis=ax_main,
                node_idx=node_idx,
                spans=self.observation_spans(node),
                top=False,
            )
            self._plot_final_state_column(axis=ax_final, node=node, node_idx=node_idx)

        self._format_axes(
            main_axis=ax_main, final_state_axis=ax_final, ordered_nodes=ordered_nodes
        )
        self._save_figure(file_name=file_name)

    def _build_subplots(
        self, ordered_nodes: list[StatechartNode]
    ) -> tuple[plt.Axes, plt.Axes]:
        """
        Builds a subplot layout with a main axis and a fixed-width final-state axis for
        the visualization of statechart nodes. Adaptively calculates layout dimensions,
        padding, and margins to ensure proper alignment and display of node labels.

        :param ordered_nodes: The nodes to be plotted, in the order they appear on the
            y-axis.
        :return: A tuple containing the main axis and the fixed-width final-state axis.
        """
        # Build node label list early so we can size the right margin adaptively
        node_names: list[str] = []
        for idx, n in enumerate(ordered_nodes):
            prev_depth = 0 if idx == 0 else ordered_nodes[idx - 1].depth
            node_names.append(self._make_label(n, prev_depth))

        # Build figure so that axes widths are fixed in physical units (inches)
        # Main axis width = length_in_units * second_width_in_cm; Final axis width = fixed value independent of second_width_in_cm
        inches_per_unit = self.second_width_in_cm / 2.54
        length_in_units = (
            self.time_span_seconds if self.use_seconds_for_x_axis else self.total_ticks
        )
        main_w_inches = inches_per_unit * float(length_in_units)
        final_w_inches = (
            self.final_state_band_height_in_cm * inches_per_unit
        )  # inches, fixed
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
            + self.gap_between_axes_in_inches
            + final_w_inches
            + right_margin_inches
        )
        fig_h_inches = self.figure_height

        fig = plt.figure(figsize=(fig_w_inches, fig_h_inches))

        # Both axes are placed by hand so their widths stay fixed in inches,
        # independent of the figure size the margins add up to.
        inner_bottom = bottom_margin_inches / fig_h_inches
        inner_top = 1 - top_margin_inches / fig_h_inches
        inner_height_in_figure_fraction = inner_top - inner_bottom

        ax_main = fig.add_axes(
            [
                left_margin_inches / fig_w_inches,
                inner_bottom,
                main_w_inches / fig_w_inches,
                inner_height_in_figure_fraction,
            ]
        )
        ax_main.grid(True, axis="x", zorder=-1)

        ax_final = fig.add_axes(
            [
                (left_margin_inches + main_w_inches + self.gap_between_axes_in_inches)
                / fig_w_inches,
                inner_bottom,
                final_w_inches / fig_w_inches,
                inner_height_in_figure_fraction,
            ],
            sharey=ax_main,
        )
        return ax_main, ax_final

    def _sort_nodes_by_parents(self) -> list[StatechartNode]:
        """
        Orders the nodes of the statechart so that each one is followed by its own
        subtree, reversed because bars are plotted bottom to top.

        :return: The nodes of the statechart, bottom bar first.
        """
        ordered = [
            node
            for root in self.statechart.top_level_nodes
            for node in [root, *root.descendants]
        ]
        return list(reversed(ordered))

    def _plot_final_state_column(
        self,
        axis: plt.Axes,
        node: StatechartNode,
        node_idx: int,
        column_padding: float = 0.1,
    ) -> None:
        """
        Draw the last recorded life cycle state (upper half) and observation state
        (lower half) of a node; nothing if no snapshot records it.

        :param axis: The axis to draw on.
        :param node: The node whose final state is drawn.
        :param node_idx: The row of the node.
        :param column_padding: The space left free on each side of the column, as a
            fraction of its width.
        """
        snapshots = self.statechart.state_history.items_recording(node)
        if not snapshots:
            return
        final_states = [
            snapshots[-1].life_cycle_state[node],
            snapshots[-1].observation_state[node],
        ]
        for state, top in zip(final_states, [True, False]):
            self._draw_block(
                axis=axis,
                node_idx=node_idx,
                block_start=column_padding,
                block_width=max(0.0, 1.0 - 2 * column_padding),
                color=state.color,
                top=top,
            )

    def _plot_spans(
        self,
        axis: plt.Axes,
        node_idx: int,
        spans: list[StateSpan],
        top: bool,
    ) -> None:
        """
        Draw one block per span, in the color of its state.

        :param axis: The axis to draw on.
        :param node_idx: The row of the node the spans belong to.
        :param spans: The spans to draw.
        :param top: Whether to draw in the upper half of the row instead of the lower.
        """
        for span in spans:
            self._draw_block(
                axis=axis,
                node_idx=node_idx,
                block_start=span.start_tick * self.x_width_per_tick,
                block_width=(span.end_tick - span.start_tick) * self.x_width_per_tick,
                color=span.state.color,
                top=top,
            )

    def _draw_block(
        self,
        axis: plt.Axes,
        node_idx,
        block_start,
        block_width,
        color: Color,
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
        :param top: A flag indicating whether the block should be positioned above the
            baseline (True) or below it (False).
        :param bar_height: The total height of the bar containing the block. Defaults to
            0.8.
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
            color=color.to_hex(),
            zorder=2,
        )

    def _format_axes(
        self,
        main_axis: plt.Axes,
        final_state_axis: plt.Axes,
        ordered_nodes: list[StatechartNode],
    ):
        """
        Configure and format axes for visualizing statechart nodes.

        This function modifies the provided matplotlib axes to display a timeline and
        statechart information for a simulation or experiment. Additionally, it prepares
        the axes to show details such as time units, tick labels, final-state
        configurations, and node-specific labels, enabling clear visual representation
        of the statechart.

        :param main_axis: Matplotlib Axes object used for the main timeline display.
        :param final_state_axis: Matplotlib Axes object used for the final-state
            representation.
        :param ordered_nodes: List of StatechartNode objects to determine the y-axis
            labels and structure.
        """
        # Configure x-axis for main timeline
        if self.use_seconds_for_x_axis:
            main_axis.set_xlabel("Time [s]")
            base_ticks = np.arange(0.0, self.time_span_seconds + 1e-9, 0.5).tolist()
            main_axis.set_xlim(0, self.time_span_seconds)
        else:
            main_axis.set_xlabel("Tick")
            step = max(int(self.x_width_per_tick), 1)
            base_ticks = list(range(0, self.total_ticks + 1, step))
            main_axis.set_xlim(0, self.total_ticks)
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

    def _make_label(self, node: StatechartNode, prev_depth: int) -> str:
        """
        Generates a formatted label for a given node in a statechart by incorporating
        its depth and using ASCII art for hierarchical representation.

        :param node: The statechart node for which the label is created.
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

    def _measure_labels_width_in(self, labels: list[str]) -> float:
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

        :param file_name: The complete path and file name where the figure should be
            saved.
        """
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_name)
        plt.close()
        logger.info(f"Saved gantt chart to {file_name}.")
