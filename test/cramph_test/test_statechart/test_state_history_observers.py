"""
History subscriptions follow the recorded snapshots and observer ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.node import CancelStatechart, StatechartNode
from cramph.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    LifeCycleCallback,
    NodeAssertionError,
    NodeRecordingItsCallbacks,
    NodeSucceedingOnObservingTrue,
)
from cramph.state_history import StateHistory, StateHistoryItem, StateHistoryObserver
from cramph.statechart import Statechart
from krrood.symbolic_math.symbolic_math import Scalar

# %% observers


@dataclass(kw_only=True)
class HistoryRecorder(StateHistoryObserver):
    """
    Remember snapshots available when a history change is delivered.
    """

    snapshots: list[StateHistoryItem] = field(default_factory=list)
    """
    The appended snapshots in notification order.
    """

    def on_state_change(self, *, history: StateHistory) -> None:
        """
        Retain the snapshot already appended to the observed history.

        :param history: The history whose newest snapshot was published.
        """
        self.snapshots.append(history.latest_item)


@dataclass(kw_only=True)
class ObserverReplacement(StateHistoryObserver):
    """
    Replace one subscription while a notification is being delivered.
    """

    removed: StateHistoryObserver
    """
    The subscription to remove.
    """

    added: StateHistoryObserver
    """
    The subscription to add.
    """

    def on_state_change(self, *, history: StateHistory) -> None:
        """
        Replace subscriptions for subsequent history changes.

        :param history: The history whose subscriptions change.
        """
        history.remove_observer(observer=self.removed)
        history.add_observer(observer=self.added)


# %% history changes


@pytest.fixture
def history_chart(statechart_context: StatechartContext) -> Statechart:
    """
    A compiled chart whose node state can be changed independently of execution.
    """
    chart = Statechart(context=statechart_context)
    chart.add_node(node=ConstTrueNode())
    StatechartExecutor(context=statechart_context).compile(statechart=chart)
    return chart


def append_snapshot(chart: Statechart) -> StateHistoryItem:
    """
    Append a snapshot of the chart's current states.

    :param chart: The chart whose history receives the snapshot.
    :return: The snapshot offered to the history.
    """
    snapshot = StateHistoryItem(
        tick_count=len(chart.state_history),
        life_cycle_state=chart.life_cycle_state,
        observation_state=chart.observation_state,
    )
    chart.state_history.append(next_item=snapshot)
    return snapshot


def test_history_notifies_only_when_a_snapshot_changes(history_chart) -> None:
    """
    Repeated ticks do not produce duplicate history notifications.
    """
    recorder = HistoryRecorder()
    history_chart.state_history.add_observer(observer=recorder)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.PAUSED
    paused = append_snapshot(history_chart)
    append_snapshot(history_chart)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.RUNNING
    running = append_snapshot(history_chart)

    assert recorder.snapshots == [paused, running]
    assert recorder.snapshots[-1] is history_chart.state_history.latest_item


def test_an_empty_history_has_no_latest_item() -> None:
    history = StateHistory()

    assert history.latest_item is None
    assert history.item_before_latest is None


def test_item_before_latest_is_the_snapshot_recorded_before_the_latest(
    history_chart,
) -> None:
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.PAUSED
    paused = append_snapshot(history_chart)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.RUNNING
    running = append_snapshot(history_chart)

    assert history_chart.state_history.item_before_latest is paused
    assert history_chart.state_history.latest_item is running


def test_history_registration_is_idempotent_by_identity(history_chart) -> None:
    """
    Equal observer instances remain distinct subscriptions.
    """
    first = HistoryRecorder()
    second = HistoryRecorder()
    history_chart.state_history.add_observer(observer=first)
    history_chart.state_history.add_observer(observer=first)
    history_chart.state_history.add_observer(observer=second)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.PAUSED
    snapshot = append_snapshot(history_chart)

    assert len(history_chart.state_history.observers) == 2
    assert history_chart.state_history.observers[0] is first
    assert history_chart.state_history.observers[1] is second
    assert first.snapshots == [snapshot]
    assert second.snapshots == [snapshot]


def test_history_removes_only_the_owned_observer(history_chart) -> None:
    """
    Removing an observer twice preserves a different equal observer.
    """
    first = HistoryRecorder()
    second = HistoryRecorder()
    history_chart.state_history.add_observer(observer=first)
    history_chart.state_history.add_observer(observer=second)
    history_chart.state_history.remove_observer(observer=first)
    history_chart.state_history.remove_observer(observer=first)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.PAUSED
    snapshot = append_snapshot(history_chart)

    assert first.snapshots == []
    assert second.snapshots == [snapshot]


def test_subscription_changes_apply_after_current_notification(history_chart) -> None:
    """
    Observers can detach and attach without changing the delivery in progress.
    """
    removed = HistoryRecorder()
    added = HistoryRecorder()
    replacement = ObserverReplacement(removed=removed, added=added)
    history_chart.state_history.add_observer(observer=replacement)
    history_chart.state_history.add_observer(observer=removed)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.PAUSED
    paused = append_snapshot(history_chart)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.RUNNING
    running = append_snapshot(history_chart)

    assert removed.snapshots == [paused]
    assert added.snapshots == [running]


# %% cancelled execution


def test_cancelled_tick_publishes_its_final_snapshot(
    statechart_context: StatechartContext,
) -> None:
    """
    A cancellation records the tick it started in before its error propagates.
    """
    cancel = CancelStatechart(exception=NodeAssertionError(reason="cancelled"))
    chart = Statechart(context=statechart_context)
    chart.add_node(node=cancel)
    recorder = HistoryRecorder()
    chart.state_history.add_observer(observer=recorder)
    executor = StatechartExecutor(context=statechart_context)

    with pytest.raises(NodeAssertionError) as caught:
        executor.compile(statechart=chart)
        executor.tick()

    assert caught.value is cancel.exception
    final = recorder.snapshots[-1]
    assert final.life_cycle_state[cancel] is LifeCycleValue.RUNNING
    assert final.observation_state == chart.observation_state
    assert final.life_cycle_state == chart.life_cycle_state


def test_cancelled_tick_preserves_error_when_observer_fails(
    statechart_context: StatechartContext,
) -> None:
    """
    A subscriber failure cannot replace the chart's cancellation reason.
    """
    trigger = ConstTrueNode()
    cancel = CancelStatechart.when_true(
        node=trigger, exception=NodeAssertionError(reason="cancelled")
    )
    chart = Statechart(context=statechart_context)
    chart.add_nodes(nodes=[trigger, cancel])
    executor = StatechartExecutor(context=statechart_context)
    executor.compile(statechart=chart)
    recorder = HistoryRecorder()
    observer = Mock(spec=StateHistoryObserver)
    observer.on_state_change.side_effect = ValueError("observer failed")
    chart.state_history.add_observer(observer=recorder)
    chart.state_history.add_observer(observer=observer)

    with pytest.raises(NodeAssertionError) as caught:
        while True:
            executor.tick()

    assert caught.value is cancel.exception
    assert recorder.snapshots[-1].life_cycle_state[cancel] is LifeCycleValue.RUNNING


@dataclass(kw_only=True)
class ChartCancelledWithNodesInEveryState:
    """
    A chart cancelled in its first tick, with one node in each state a node can be in
    when that happens.
    """

    chart: Statechart
    """
    The cancelled chart.
    """

    cancel: CancelStatechart
    """
    The node that cancelled the chart.
    """

    running: NodeRecordingItsCallbacks
    """
    A node that was still running.
    """

    paused: StatechartNode
    """
    A node that was paused.
    """

    succeeded: StatechartNode
    """
    A node that had succeeded, which started the cancellation.
    """

    not_started: StatechartNode
    """
    A node that had not started.
    """

    raised: NodeAssertionError
    """
    The exception the cancellation raised.
    """


@pytest.fixture
def chart_cancelled_with_nodes_in_every_state(
    statechart_context: StatechartContext,
) -> ChartCancelledWithNodesInEveryState:
    """
    :return: The chart, ticked until its cancellation raised.
    """
    running = NodeRecordingItsCallbacks()
    paused = ConstFalseNode()
    paused.pause_condition = Scalar.const_true()
    succeeded = NodeSucceedingOnObservingTrue(observation=ObservationStateValue.TRUE)
    not_started = ConstFalseNode()
    not_started.start_condition = Scalar.const_false()
    cancel = CancelStatechart.when_true(
        node=succeeded, exception=NodeAssertionError(reason="cancelled")
    )
    chart = Statechart(context=statechart_context)
    chart.add_nodes(nodes=[running, paused, succeeded, not_started, cancel])
    executor = StatechartExecutor(context=statechart_context)
    executor.compile(statechart=chart)
    with pytest.raises(NodeAssertionError) as raised:
        executor.tick()
    return ChartCancelledWithNodesInEveryState(
        chart=chart,
        cancel=cancel,
        running=running,
        paused=paused,
        succeeded=succeeded,
        not_started=not_started,
        raised=raised.value,
    )


def test_cancellation_interrupts_unfinished_nodes(
    chart_cancelled_with_nodes_in_every_state: ChartCancelledWithNodesInEveryState,
) -> None:
    """
    Running and paused nodes end as interrupted; nodes that had not started or had
    ended, and the cancelling node itself, keep their state.
    """
    cancelled = chart_cancelled_with_nodes_in_every_state

    assert {
        node: node.life_cycle_state
        for node in [
            cancelled.running,
            cancelled.paused,
            cancelled.succeeded,
            cancelled.not_started,
            cancelled.cancel,
        ]
    } == {
        cancelled.running: LifeCycleValue.INTERRUPTED,
        cancelled.paused: LifeCycleValue.INTERRUPTED,
        cancelled.succeeded: LifeCycleValue.SUCCEEDED,
        cancelled.not_started: LifeCycleValue.NOT_STARTED,
        cancelled.cancel: LifeCycleValue.RUNNING,
    }


def test_cancellation_is_the_last_record_of_the_history(
    chart_cancelled_with_nodes_in_every_state: ChartCancelledWithNodesInEveryState,
) -> None:
    """
    The history ends with the states the cancellation left the nodes in.
    """
    cancelled = chart_cancelled_with_nodes_in_every_state

    final = cancelled.chart.state_history.latest_item
    assert final.life_cycle_state == cancelled.chart.life_cycle_state
    assert final.life_cycle_state[cancelled.running] is LifeCycleValue.INTERRUPTED


def test_cancellation_record_names_the_cancelling_node(
    chart_cancelled_with_nodes_in_every_state: ChartCancelledWithNodesInEveryState,
) -> None:
    """
    Only the record of the cancellation names the node that cancelled the chart.
    """
    cancelled = chart_cancelled_with_nodes_in_every_state

    *earlier, final = cancelled.chart.state_history.items
    assert [item.cancelled_by for item in earlier] == [None] * len(earlier)
    assert final.cancelled_by is cancelled.cancel


def test_cancellation_record_keeps_the_raised_exception(
    chart_cancelled_with_nodes_in_every_state: ChartCancelledWithNodesInEveryState,
) -> None:
    """
    Only the record of the cancellation keeps the exception the chart was cancelled
    with, so the reason survives once the exception itself is gone.
    """
    cancelled = chart_cancelled_with_nodes_in_every_state

    *earlier, final = cancelled.chart.state_history.items
    assert [item.cancellation_error for item in earlier] == [None] * len(earlier)
    assert final.cancellation_error is cancelled.raised


def test_cancellation_runs_the_end_callback_of_interrupted_nodes(
    chart_cancelled_with_nodes_in_every_state: ChartCancelledWithNodesInEveryState,
) -> None:
    """
    An interrupted node is told that it ended, as for every other terminal state.
    """
    assert chart_cancelled_with_nodes_in_every_state.running.take_callbacks() == [
        LifeCycleCallback.START,
        LifeCycleCallback.END,
    ]


# %% incomplete ticks


def test_failed_settle_does_not_publish_partial_snapshot(
    monkeypatch, history_chart
) -> None:
    """
    A tick that fails while settling records nothing.
    """
    recorder = HistoryRecorder()
    history_chart.state_history.add_observer(observer=recorder)
    snapshots = list(history_chart.state_history.items)
    failure = NodeAssertionError(reason="settling failed")

    def fail_settle() -> None:
        """
        Change one state before settling fails to complete.
        """
        history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValue.PAUSED
        raise failure

    monkeypatch.setattr(history_chart.compiler.compiled_tick, "settle", fail_settle)
    with pytest.raises(NodeAssertionError) as caught:
        history_chart.tick()

    assert caught.value is failure
    assert history_chart.state_history.items == snapshots
    assert recorder.snapshots == []


# %% which nodes started and ended in the newest snapshot


def _compiled_with(
    statechart_context: StatechartContext, node: StatechartNode
) -> StatechartExecutor:
    """
    :return: An executor that compiled a statechart running only `node`, which ticks it
        once.
    """
    executor = StatechartExecutor(context=statechart_context)
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=node)
    executor.compile(statechart=statechart)
    return executor


def test_a_node_that_just_started_is_among_those_started_in_the_newest_snapshot(
    statechart_context: StatechartContext,
) -> None:
    running = ConstTrueNode(name="running")

    history = _compiled_with(statechart_context, running).statechart.state_history

    assert (
        history.nodes_started_in_latest_item(),
        history.nodes_ended_in_latest_item(),
    ) == ([running], [])


def test_a_node_that_just_ended_is_among_those_ended_in_the_newest_snapshot(
    statechart_context: StatechartContext,
) -> None:
    succeeding = NodeSucceedingOnObservingTrue(
        name="succeeding", observation=ObservationStateValue.TRUE
    )

    executor = _compiled_with(statechart_context, succeeding)

    executor.tick()

    assert executor.statechart.state_history.nodes_ended_in_latest_item() == [
        succeeding
    ]


# %% snapshots of a state


def test_a_state_snapshot_reads_its_values_against_the_same_statechart(
    history_chart,
) -> None:
    """
    A snapshot is keyed by the nodes of the statechart it was taken of, not a copy.
    """
    snapshot = history_chart.life_cycle_state.snapshot()

    assert snapshot.statechart is history_chart


def test_a_state_snapshot_keeps_its_values_when_the_state_changes(
    history_chart,
) -> None:
    """
    Changing the live state after taking a snapshot leaves the snapshot as it was.
    """
    node = history_chart.nodes[0]
    history_chart.life_cycle_state[node] = LifeCycleValue.PAUSED
    snapshot = history_chart.life_cycle_state.snapshot()
    history_chart.life_cycle_state[node] = LifeCycleValue.RUNNING

    assert snapshot[node] == LifeCycleValue.PAUSED
