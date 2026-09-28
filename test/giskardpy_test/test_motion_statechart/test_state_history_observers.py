"""
History subscriptions follow native state snapshots and observer ownership.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import CancelMotion, MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    StateHistory,
    StateHistoryItem,
    StateHistoryObserver,
)


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
        self.snapshots.append(history.history[-1])


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
def history_chart() -> MotionStatechart:
    """
    A native chart whose node state can be changed independently of execution.
    """
    chart = MotionStatechart()
    chart.add_node(MotionStatechartNode())
    return chart


def append_snapshot(chart: MotionStatechart) -> StateHistoryItem:
    """
    Append a snapshot of the chart's current native states.

    :param chart: The chart whose history receives the snapshot.
    :return: The snapshot offered to the history.
    """
    snapshot = StateHistoryItem(
        control_cycle=len(chart.history),
        life_cycle_state=chart.life_cycle_state,
        observation_state=chart.observation_state,
    )
    chart.history.append(snapshot)
    return snapshot


def test_history_notifies_only_when_a_snapshot_changes(history_chart) -> None:
    """
    Repeated control cycles do not produce duplicate history notifications.
    """
    recorder = HistoryRecorder()
    history_chart.history.add_observer(recorder)
    initial = append_snapshot(history_chart)
    append_snapshot(history_chart)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.RUNNING
    running = append_snapshot(history_chart)

    assert recorder.snapshots == [initial, running]
    assert recorder.snapshots[-1] is history_chart.history.history[-1]


def test_history_registration_is_idempotent_by_identity(history_chart) -> None:
    """
    Equal observer instances remain distinct subscriptions.
    """
    first = HistoryRecorder()
    second = HistoryRecorder()
    history_chart.history.add_observer(first)
    history_chart.history.add_observer(first)
    history_chart.history.add_observer(second)
    snapshot = append_snapshot(history_chart)

    assert len(history_chart.history.observers) == 2
    assert history_chart.history.observers[0] is first
    assert history_chart.history.observers[1] is second
    assert first.snapshots == [snapshot]
    assert second.snapshots == [snapshot]


def test_history_removes_only_the_owned_observer(history_chart) -> None:
    """
    Removing an observer twice preserves a different equal observer.
    """
    first = HistoryRecorder()
    second = HistoryRecorder()
    history_chart.history.add_observer(first)
    history_chart.history.add_observer(second)
    history_chart.history.remove_observer(first)
    history_chart.history.remove_observer(first)
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
    history_chart.history.add_observer(replacement)
    history_chart.history.add_observer(removed)
    initial = append_snapshot(history_chart)
    history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.RUNNING
    running = append_snapshot(history_chart)

    assert removed.snapshots == [initial]
    assert added.snapshots == [running]


# %% cancelled execution
def test_cancelled_tick_publishes_its_final_native_snapshot(mini_world) -> None:
    """
    A cancellation retains the changed observation before its error propagates.
    """
    failure = RuntimeError()
    cancel = CancelMotion(exception=failure)
    chart = MotionStatechart()
    chart.add_node(cancel)
    recorder = HistoryRecorder()
    chart.history.add_observer(recorder)
    context = MotionStatechartContext(world=mini_world)
    chart.compile(context)
    chart.tick(context)

    with pytest.raises(RuntimeError) as caught:
        chart.tick(context)

    assert caught.value is failure
    final = recorder.snapshots[-1]
    assert final.observation_state[cancel] is ObservationStateValues.TRUE
    assert final.observation_state == chart.observation_state
    assert final.life_cycle_state == chart.life_cycle_state


# %% incomplete updates
@pytest.mark.parametrize(
    "update", ["_update_observation_state", "_update_life_cycle_state"]
)
def test_failed_update_does_not_publish_partial_snapshot(
    monkeypatch, history_chart, mini_world, update
) -> None:
    """
    An incomplete native update cannot create a recorded control cycle.
    """
    recorder = HistoryRecorder()
    history_chart.history.add_observer(recorder)
    context = MotionStatechartContext(world=mini_world)
    history_chart.compile(context)
    snapshots = list(history_chart.history.history)
    observed_snapshots = list(recorder.snapshots)
    failure = RuntimeError("native update failed")

    def fail_update(context: MotionStatechartContext) -> None:
        """
        Change one state before the updater fails to complete.

        :param context: The active control context.
        """
        history_chart.life_cycle_state[history_chart.nodes[0]] = LifeCycleValues.RUNNING
        raise failure

    monkeypatch.setattr(history_chart, update, fail_update)
    with pytest.raises(type(failure)) as caught:
        history_chart.tick(context)

    assert caught.value is failure
    assert history_chart.history.history == snapshots
    assert recorder.snapshots == observed_snapshots


def test_cancelled_tick_preserves_error_when_observer_fails(
    monkeypatch, mini_world
) -> None:
    """
    A subscriber failure cannot replace the chart's cancellation reason.
    """
    failure = RuntimeError("motion cancelled")
    observer_failure = ValueError("observer failed")
    cancel = CancelMotion(exception=failure)
    chart = MotionStatechart()
    chart.add_node(cancel)
    context = MotionStatechartContext(world=mini_world)
    chart.compile(context)
    chart.tick(context)
    recorder = HistoryRecorder()
    observer = Mock(spec=StateHistoryObserver)
    observer.on_state_change.side_effect = observer_failure
    chart.history.add_observer(recorder)
    chart.history.add_observer(observer)

    with pytest.raises(type(failure)) as caught:
        chart.tick(context)

    assert caught.value is failure
    assert (
        recorder.snapshots[-1].observation_state[cancel] is ObservationStateValues.TRUE
    )
