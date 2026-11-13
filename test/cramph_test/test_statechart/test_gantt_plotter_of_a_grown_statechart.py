from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from cramph.data_types import LifeCycleValue, ObservationStateValue
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
        node=NodeSucceedingOnObservingTrue(
            name="first", observation=ObservationStateValue.TRUE
        )
    )
    executor.compile(statechart=statechart)
    for _ in range(4):
        executor.tick()
    late = NodeSucceedingOnObservingTrue(
        name="late", observation=ObservationStateValue.TRUE
    )
    statechart.add_node(node=late)
    for _ in range(4):
        executor.tick()
    return late


# %% bars of a node that joined late


def test_the_bars_of_a_late_node_span_the_ticks_it_was_recorded_at(
    statechart_executor: StatechartExecutor,
) -> None:
    late = _run_a_statechart_gaining_a_node(statechart_executor)
    statechart = statechart_executor.statechart
    run = statechart.state_history.get_current_run_ticks_of_node(node=late)

    spans = HistoryGanttChartPlotter(statechart=statechart).life_cycle_spans(node=late)

    assert spans == [
        StateSpan(
            state=LifeCycleValue.RUNNING,
            start_tick=run.start_tick,
            end_tick=run.end_tick,
        ),
        StateSpan(
            state=LifeCycleValue.SUCCEEDED,
            start_tick=run.end_tick,
            end_tick=statechart.state_history.latest_item.tick_count,
        ),
    ]


def test_the_observation_bars_of_a_late_node_start_when_it_joined(
    statechart_executor: StatechartExecutor,
) -> None:
    late = _run_a_statechart_gaining_a_node(statechart_executor)
    statechart = statechart_executor.statechart
    joining_snapshot = statechart.state_history.items_recording(node=late)[0]

    spans = HistoryGanttChartPlotter(statechart=statechart).observation_spans(node=late)

    assert spans[0].start_tick == joining_snapshot.tick_count
    assert spans[-1].end_tick == statechart.state_history.latest_item.tick_count


def test_a_node_no_snapshot_records_has_no_bars(
    statechart_executor: StatechartExecutor, tmp_path: Path
) -> None:
    _run_a_statechart_gaining_a_node(statechart_executor)
    statechart = statechart_executor.statechart
    unrecorded = NodeSucceedingOnObservingTrue(
        name="unrecorded", observation=ObservationStateValue.TRUE
    )
    statechart.add_node(node=unrecorded)
    plotter = HistoryGanttChartPlotter(statechart=statechart)

    assert plotter.life_cycle_spans(node=unrecorded) == []
    assert plotter.observation_spans(node=unrecorded) == []
    plotter.plot_gantt_chart(file_name=str(tmp_path / "gantt_chart.pdf"))


# %% rows of a statechart that grew


def test_the_chart_has_a_row_for_every_node_the_statechart_holds_at_the_end(
    statechart_executor: StatechartExecutor,
) -> None:
    _run_a_statechart_gaining_a_node(statechart_executor)
    statechart = statechart_executor.statechart

    assert HistoryGanttChartPlotter(statechart=statechart).num_bars == len(
        statechart.nodes
    )
