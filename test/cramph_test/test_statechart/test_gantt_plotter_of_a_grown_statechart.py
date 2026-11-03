from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.plotters.gantt_chart_plotter import HistoryGanttChartPlotter, StateSpan
from cramph.statechart import Statechart

# %% helpers


def _run_a_statechart_gaining_a_node(
    executor: StatechartExecutor,
) -> NodeSucceedingOnObservingTrue:
    """
    Tick a statechart holding one node a few times, add a second node, and tick until
    some ticks after that one succeeded.

    :return: The node added late.
    """
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        NodeSucceedingOnObservingTrue(
            name="first", observation=ObservationStateValues.TRUE
        )
    )
    executor.compile(statechart)
    for _ in range(4):
        executor.tick()
    late = NodeSucceedingOnObservingTrue(
        name="late", observation=ObservationStateValues.TRUE
    )
    statechart.add_node(late)
    for _ in range(4):
        executor.tick()
    return late


# %% bars of a node that joined late


def test_the_bars_of_a_late_node_span_the_ticks_it_was_recorded_at(
    statechart_executor: StatechartExecutor,
) -> None:
    late = _run_a_statechart_gaining_a_node(statechart_executor)
    statechart = statechart_executor.statechart
    run = statechart.state_history.get_current_run_ticks_of_node(late)

    spans = HistoryGanttChartPlotter(statechart).life_cycle_spans(late)

    assert spans == [
        StateSpan(
            state=LifeCycleValues.RUNNING,
            start_tick=run.start_tick,
            end_tick=run.end_tick,
        ),
        StateSpan(
            state=LifeCycleValues.SUCCEEDED,
            start_tick=run.end_tick,
            end_tick=statechart.state_history.latest_item.tick_count,
        ),
    ]


def test_the_observation_bars_of_a_late_node_start_when_it_joined(
    statechart_executor: StatechartExecutor,
) -> None:
    late = _run_a_statechart_gaining_a_node(statechart_executor)
    statechart = statechart_executor.statechart
    joining_snapshot = statechart.state_history.items_recording(late)[0]

    spans = HistoryGanttChartPlotter(statechart).observation_spans(late)

    assert spans[0].start_tick == joining_snapshot.tick_count
    assert spans[-1].end_tick == statechart.state_history.latest_item.tick_count


def test_a_node_no_snapshot_records_has_no_bars(
    statechart_executor: StatechartExecutor, tmp_path: Path
) -> None:
    _run_a_statechart_gaining_a_node(statechart_executor)
    statechart = statechart_executor.statechart
    unrecorded = NodeSucceedingOnObservingTrue(
        name="unrecorded", observation=ObservationStateValues.TRUE
    )
    statechart.add_node(unrecorded)
    plotter = HistoryGanttChartPlotter(statechart)

    assert plotter.life_cycle_spans(unrecorded) == []
    assert plotter.observation_spans(unrecorded) == []
    plotter.plot_gantt_chart(str(tmp_path / "gantt_chart.pdf"))


# %% rows of a statechart that grew


def test_the_chart_has_a_row_for_every_node_the_statechart_holds_at_the_end(
    statechart_executor: StatechartExecutor,
) -> None:
    _run_a_statechart_gaining_a_node(statechart_executor)
    statechart = statechart_executor.statechart

    assert HistoryGanttChartPlotter(statechart).num_bars == len(statechart.nodes)
