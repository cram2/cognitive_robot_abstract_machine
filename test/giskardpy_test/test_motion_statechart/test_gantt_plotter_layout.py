import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.monitors.payload_monitors import CountControlCycles
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.plotters.gantt_chart_plotter import (
    HistoryGanttChartPlotter,
)
from giskardpy.motion_statechart.test_nodes.test_nodes import (
    TestNestedGoal,
    ConstTrueNode,
)
from semantic_digital_twin.world import World


def _axes_width_in(ax: plt.Axes) -> float:
    """
    Return the drawable width of an axes in inches based on its position box,
    excluding figure margins and avoiding text extents influencing the result.
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
def test_main_and_final_widths_control_cycles(monkeypatch, ticks):
    # Build a small statechart that runs for `ticks` control cycles
    msc = MotionStatechart()
    counter = CountControlCycles(control_cycles=ticks)
    msc.add_node(counter)
    msc.add_node(EndMotion.when_true(counter))

    kin = Executor(world=World())
    kin.compile(msc)
    kin.tick_until_end(ticks + 5)

    # Use control cycles (no context)
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


def test_long_labels_not_clipped_on_right(monkeypatch):
    msc = MotionStatechart()
    # Create a few nodes with long names
    n1 = ConstTrueNode(name="NODE_" + ("LONG_" * 10))
    n2 = ConstTrueNode(name="NODE_" + ("VERY_LONG_LABEL_" * 6))
    msc.add_nodes([n1, n2])
    msc.add_node(EndMotion.when_true(n2))

    kin = Executor(world=World())
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


def test_x_axis_units_control_cycles_vs_seconds(monkeypatch):
    msc = MotionStatechart()
    counter = CountControlCycles(control_cycles=5)
    msc.add_nodes([counter])
    msc.add_node(EndMotion.when_true(counter))

    kin = Executor(world=World())
    kin.compile(msc)
    kin.tick_until_end()

    # Control cycles (no context)
    plotter_cycles = HistoryGanttChartPlotter(msc, context=None, second_width_in_cm=2.0)
    axes_cycles = _render_and_capture_axes(plotter_cycles, monkeypatch)
    ax_main_cycles = axes_cycles["main"]
    assert ax_main_cycles.get_xlabel() == "Control cycle"
    assert tuple(ax_main_cycles.get_xlim())[0] == 0.0

    # Seconds (with context)
    context = kin.build_context
    plotter_seconds = HistoryGanttChartPlotter(
        msc, context=context, second_width_in_cm=2.0
    )
    axes_seconds = _render_and_capture_axes(plotter_seconds, monkeypatch)
    ax_main_seconds = axes_seconds["main"]
    assert ax_main_seconds.get_xlabel() == "Time [s]"
    # Upper xlim should equal total_cycles * dt
    total_cycles = msc.history.history[-1].control_cycle
    expected_span = total_cycles * context.qp_controller_config.control_dt
    assert ax_main_seconds.get_xlim()[1] == pytest.approx(
        expected_span, rel=1e-6, abs=1e-6
    )


def test_tree_glyphs_in_labels(monkeypatch):
    msc = MotionStatechart()
    root1 = ConstTrueNode(name="A")
    nested = TestNestedGoal(name="B")
    msc.add_nodes([root1, nested])
    msc.add_node(EndMotion.when_true(root1))

    kin = Executor(world=World())
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
