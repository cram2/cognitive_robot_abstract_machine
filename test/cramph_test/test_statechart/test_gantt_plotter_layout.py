import matplotlib

from cramph.context import StatechartContext

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from cramph.exceptions import TickDurationUnknownError
from cramph.monitors import CountTicks
from cramph.node import EndStatechart
from cramph.plotters.gantt_chart_plotter import HistoryGanttChartPlotter
from cramph.nodes_for_testing import (
    CompositeNodeWithNestedCompositeChild,
    ConstTrueNode,
)
from semantic_digital_twin.world import World


def _axes_width_in(ax: plt.Axes) -> float:
    """
    Return the drawable width of an axes in inches based on its position box, excluding
    figure margins and avoiding text extents influencing the result.
    """
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_position()  # in figure fraction
    return bbox.width * fig.get_figwidth()


def _rightmost_text_pixel_x(texts, fig) -> float:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer=r) for t in texts if t.get_visible()]
    return max((bb.x1 for bb in bboxes), default=0.0)


def _render_and_capture_axes(plotter: HistoryGanttChartPlotter, monkeypatch):
    # Avoid file output
    monkeypatch.setattr(plotter, "_save_figure", lambda file_name=None, **kwargs: None)
    captured = {}

    original = plotter._build_subplots

    def spy(labels):
        ax_main, ax_final = original(labels)
        captured["main"], captured["final"] = ax_main, ax_final
        return ax_main, ax_final

    monkeypatch.setattr(plotter, "_build_subplots", spy)

    plotter.plot_gantt_chart("/dev/null")
    return captured


@pytest.mark.parametrize("ticks", [3, 50])
def test_main_and_final_widths_ticks(monkeypatch, ticks) -> None:
    # Build a small statechart that runs for `ticks` ticks
    kin = StatechartExecutor(context=StatechartContext(world=World()))
    msc = Statechart(context=kin.context)
    counter = CountTicks(ticks=ticks)
    msc.add_node(counter)
    msc.add_node(EndStatechart.when_true(counter))

    kin.compile(msc)
    kin.tick_until_end(ticks + 5)

    # Use ticks (no context)
    plotter = HistoryGanttChartPlotter(msc, context=None, second_width_in_cm=2.0)
    axes = _render_and_capture_axes(plotter, monkeypatch)

    ax_main, ax_final = axes["main"], axes["final"]

    # Expected: main axis width in inches equals ticks * (cm per unit)/2.54
    cm_per_unit = plotter.second_width_in_cm
    # Derive expected width from axis limits to avoid off-by-one assumptions
    x0, x1 = ax_main.get_xlim()
    expected_main_w_in = (x1 - x0) * (cm_per_unit / 2.54)
    assert _axes_width_in(ax_main) == pytest.approx(
        expected_main_w_in, rel=0.05, abs=0.05
    )

    # Final column’s physical width is fixed by the plotter’s implementation (1 cm band height -> 0.5 cm configured -> 0.5/2.54 in?)
    # In our implementation final width equals final_state_band_height_in_cm * inches_per_unit.
    # With default final_state_band_height_in_cm=0.5 and second_width_in_cm=2.0, inches_per_unit=2.0/2.54 -> final_w_in = 0.5 * 2.0/2.54
    inches_per_unit = plotter.second_width_in_cm / 2.54
    expected_final_w_in = plotter.final_state_band_height_in_cm * inches_per_unit
    assert _axes_width_in(ax_final) == pytest.approx(
        expected_final_w_in, rel=0.05, abs=0.05
    )


def test_final_column_placed_right_of_main_axis(monkeypatch):
    kin = StatechartExecutor(context=StatechartContext(world=World()))
    msc = Statechart(context=kin.context)
    counter = CountTicks(ticks=4)
    msc.add_node(counter)
    msc.add_node(EndStatechart.when_true(counter))

    kin.compile(msc)
    kin.tick_until_end()

    plotter = HistoryGanttChartPlotter(msc, context=None, second_width_in_cm=2.0)
    axes = _render_and_capture_axes(plotter, monkeypatch)
    ax_main, ax_final = axes["main"], axes["final"]

    fig = ax_main.figure
    fig.canvas.draw()
    main_box, final_box = ax_main.get_position(), ax_final.get_position()
    gap_in_inches = (final_box.x0 - main_box.x1) * fig.get_figwidth()

    assert gap_in_inches == pytest.approx(plotter.gap_between_axes_in_inches, abs=1e-6)
    assert final_box.y0 == pytest.approx(main_box.y0, abs=1e-6)
    assert final_box.y1 == pytest.approx(main_box.y1, abs=1e-6)
    assert ax_final.get_ylim() == ax_main.get_ylim()


def test_long_labels_not_clipped_on_right(monkeypatch):
    kin = StatechartExecutor(context=StatechartContext(world=World()))
    msc = Statechart(context=kin.context)
    # Create a few nodes with long names
    n1 = ConstTrueNode(name="NODE_" + ("LONG_" * 10))
    n2 = ConstTrueNode(name="NODE_" + ("VERY_LONG_LABEL_" * 6))
    msc.add_nodes([n1, n2])
    msc.add_node(EndStatechart.when_true(n2))

    kin.compile(msc)
    kin.tick()

    plotter = HistoryGanttChartPlotter(msc, context=None, second_width_in_cm=2.0)
    axes = _render_and_capture_axes(plotter, monkeypatch)

    ax_final = axes["final"]
    fig = ax_final.figure
    fig.canvas.draw()

    rightmost = _rightmost_text_pixel_x(ax_final.get_yticklabels(), fig)
    # Rightmost point must be within the figure width (allow tiny tolerance)
    assert rightmost <= fig.bbox.width + 1


def test_x_axis_units_ticks_vs_seconds(
    monkeypatch, statechart_executor: StatechartExecutor
) -> None:
    msc = Statechart(context=statechart_executor.context)
    counter = CountTicks(ticks=5)
    msc.add_nodes([counter])
    msc.add_node(EndStatechart.when_true(counter))

    kin = statechart_executor
    kin.compile(msc)
    kin.tick_until_end()

    # Ticks (no context)
    plotter_ticks = HistoryGanttChartPlotter(msc, context=None, second_width_in_cm=2.0)
    axes_ticks = _render_and_capture_axes(plotter_ticks, monkeypatch)
    ax_main_ticks = axes_ticks["main"]
    assert ax_main_ticks.get_xlabel() == "Tick"
    assert tuple(ax_main_ticks.get_xlim())[0] == 0.0

    # Seconds (with context)
    context = kin.context
    plotter_seconds = HistoryGanttChartPlotter(
        msc, context=context, second_width_in_cm=2.0
    )
    axes_seconds = _render_and_capture_axes(plotter_seconds, monkeypatch)
    ax_main_seconds = axes_seconds["main"]
    assert ax_main_seconds.get_xlabel() == "Time [s]"
    # Upper xlim should equal total_ticks * dt
    total_ticks = msc.state_history.latest_item.tick_count
    expected_span = total_ticks * context.tick_duration
    assert ax_main_seconds.get_xlim()[1] == pytest.approx(
        expected_span, rel=1e-6, abs=1e-6
    )


def test_seconds_cannot_be_plotted_without_a_tick_duration(
    monkeypatch, statechart_context_without_tick_duration: StatechartContext
) -> None:
    executor = StatechartExecutor(context=statechart_context_without_tick_duration)
    msc = Statechart(context=executor.context)
    counter = CountTicks(ticks=2)
    msc.add_nodes([counter, EndStatechart.when_true(counter)])
    executor.compile(msc)
    executor.tick_until_end()

    plotter = HistoryGanttChartPlotter(
        msc, context=statechart_context_without_tick_duration
    )

    with pytest.raises(TickDurationUnknownError):
        _render_and_capture_axes(plotter, monkeypatch)


def test_tree_glyphs_in_labels(monkeypatch):
    kin = StatechartExecutor(context=StatechartContext(world=World()))
    msc = Statechart(context=kin.context)
    root1 = ConstTrueNode(name="A")
    nested = CompositeNodeWithNestedCompositeChild(name="B")
    msc.add_nodes([root1, nested])
    msc.add_node(EndStatechart.when_true(root1))

    kin.compile(msc)
    kin.tick()

    plotter = HistoryGanttChartPlotter(msc, context=None, second_width_in_cm=2.0)
    axes = _render_and_capture_axes(plotter, monkeypatch)

    ax_final = axes["final"]
    labels = [t.get_text() for t in ax_final.get_yticklabels() if t.get_text()]

    # Expect presence of box-drawing characters used for tree glyphs
    assert labels[1].startswith("└─└─ ")
    assert labels[2].startswith("│  ├─ ")
    assert labels[3].startswith("├─ ")
