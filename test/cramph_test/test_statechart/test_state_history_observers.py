"""
History subscriptions follow the recorded snapshots and observer ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.node import CancelStatechart, StatechartNode
from cramph.nodes_for_testing import (
    ConstTrueNode,
    NodeAssertionError,
    NodeSucceedingOnObservingTrue,
)
from cramph.state_history import StateHistory, StateHistoryItem, StateHistoryObserver
from cramph.statechart import Statechart

# %% observers


@dataclass
class HistoryRecorder(StateHistoryObserver):
    """
    Remember snapshots available when a history change is delivered.
    """

    snapshots: list[StateHistoryItem] = field(default_factory=list)
    """
    The appended snapshots in notification order.
    """

    def on_state_change(self, history: StateHistory) -> None:
        """
        Retain the snapshot already appended to the observed history.

        :param history: The history whose newest snapshot was published.
        """
        self.snapshots.append(history.latest_item)


@dataclass
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

    def on_state_change(self, history: StateHistory) -> None:
        """
        Replace subscriptions for subsequent history changes.

        :param history: The history whose subscriptions change.
        """
        history.remove_observer(self.removed)
        history.add_observer(self.added)


# %% history changes


@pytest.fixture
def history_chart(statechart_context: StatechartContext) -> Statechart:
    """
    A compiled chart whose node state can be changed independently of execution.
    """
    chart = Statechart(context=statechart_context)
    chart.add_node(ConstTrueNode())
    StatechartExecutor(statechart_context).compile(statechart=chart)
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
    chart.state_history.append(snapshot)
    return snapshot


def test_history_notifies_only_when_a_snapshot_changes(history_chart) -> None:
    """
    Repeated ticks do not produce duplicate history notifications.
    """
    recorder = HistoryRecorder()
    history_chart.state_history.add_observer(recorder)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.PAUSED
    paused = append_snapshot(history_chart)
    append_snapshot(history_chart)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.RUNNING
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
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.PAUSED
    paused = append_snapshot(history_chart)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.RUNNING
    running = append_snapshot(history_chart)

    assert history_chart.state_history.item_before_latest is paused
    assert history_chart.state_history.latest_item is running


def test_history_registration_is_idempotent_by_identity(history_chart) -> None:
    """
    Equal observer instances remain distinct subscriptions.
    """
    first = HistoryRecorder()
    second = HistoryRecorder()
    history_chart.state_history.add_observer(first)
    history_chart.state_history.add_observer(first)
    history_chart.state_history.add_observer(second)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.PAUSED
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
    history_chart.state_history.add_observer(first)
    history_chart.state_history.add_observer(second)
    history_chart.state_history.remove_observer(first)
    history_chart.state_history.remove_observer(first)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.PAUSED
    snapshot = append_snapshot(history_chart)

    assert first.snapshots == []
    assert second.snapshots == [snapshot]


def test_subscription_changes_apply_after_current_notification(history_chart) -> None:
    """
    Observers can detach and attach without changing the delivery in progress.
    """
    removed = HistoryRecorder()
    added = HistoryRecorder()
    replacement = ObserverReplacement(removed, added)
    history_chart.state_history.add_observer(replacement)
    history_chart.state_history.add_observer(removed)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.PAUSED
    paused = append_snapshot(history_chart)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.RUNNING
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
    chart.add_node(cancel)
    recorder = HistoryRecorder()
    chart.state_history.add_observer(recorder)
    executor = StatechartExecutor(statechart_context)

    with pytest.raises(NodeAssertionError) as caught:
        executor.compile(statechart=chart)
        executor.tick()

    assert caught.value is cancel.exception
    final = recorder.snapshots[-1]
    assert final.life_cycle_state[cancel] is LifeCycleValues.RUNNING
    assert final.observation_state == chart.observation_state
    assert final.life_cycle_state == chart.life_cycle_state


def test_cancelled_tick_preserves_error_when_observer_fails(
    statechart_context: StatechartContext,
) -> None:
    """
    A subscriber failure cannot replace the chart's cancellation reason.
    """
    trigger = ConstTrueNode()
    cancel = CancelStatechart.when_true(trigger, NodeAssertionError(reason="cancelled"))
    chart = Statechart(context=statechart_context)
    chart.add_nodes([trigger, cancel])
    executor = StatechartExecutor(statechart_context)
    executor.compile(statechart=chart)
    recorder = HistoryRecorder()
    observer = Mock(spec=StateHistoryObserver)
    observer.on_state_change.side_effect = ValueError("observer failed")
    chart.state_history.add_observer(recorder)
    chart.state_history.add_observer(observer)

    with pytest.raises(NodeAssertionError) as caught:
        while True:
            executor.tick()

    assert caught.value is cancel.exception
    assert recorder.snapshots[-1].life_cycle_state[cancel] is LifeCycleValues.RUNNING


# %% incomplete ticks


def test_failed_settle_does_not_publish_partial_snapshot(
    monkeypatch, history_chart
) -> None:
    """
    A tick that fails while settling records nothing.
    """
    recorder = HistoryRecorder()
    history_chart.state_history.add_observer(recorder)
    snapshots = list(history_chart.state_history.items)
    failure = NodeAssertionError(reason="settling failed")

    def fail_settle() -> None:
        """
        Change one state before settling fails to complete.
        """
        history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.PAUSED
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
    statechart.add_node(node)
    executor.compile(statechart)
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
        name="succeeding", observation=ObservationStateValues.TRUE
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
    history_chart.life_cycle_state[node] = LifeCycleValues.PAUSED
    snapshot = history_chart.life_cycle_state.snapshot()
    history_chart.life_cycle_state[node] = LifeCycleValues.RUNNING

    assert snapshot[node] == LifeCycleValues.PAUSED
