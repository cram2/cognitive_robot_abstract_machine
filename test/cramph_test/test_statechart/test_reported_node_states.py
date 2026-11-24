"""
Tests for :class:`~cramph.reported_statechart.ReportedNodeStates`, the states a server
reports for the nodes of a statechart, naming every node by its id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValue
from cramph.exceptions import ReportedNodeNotInStatechartError
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.reported_statechart import ReportedNodeStates
from cramph.state_history import StateHistory, StateHistoryObserver
from cramph.statechart import Statechart
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.world import World

# %% mimics


@dataclass(kw_only=True)
class ObserverRecordingTicks(StateHistoryObserver):
    """
    Records the tick of every snapshot a history tells it about.
    """

    ticks: list[int] = field(default_factory=list)
    """
    The ticks of the snapshots, in the order they were recorded.
    """

    def on_state_change(self, *, history: StateHistory) -> None:
        self.ticks.append(history.latest_item.tick_count)


@dataclass(kw_only=True)
class ServerAndCopy:
    """
    A statechart a server ticks, and a copy of it that a receiver holds.
    """

    server_executor: StatechartExecutor
    """
    The executor ticking the server's statechart.
    """

    copy: Statechart
    """
    The receiver's copy, read from the structure of the server's statechart.
    """

    @property
    def server_statechart(self) -> Statechart:
        """
        The statechart the server ticks.
        """
        return self.server_executor.statechart

    def report(self) -> ReportedNodeStates:
        """
        :return: The states the server reports now, after a trip through a JSON string.
        """
        reported = ReportedNodeStates.from_statechart(
            statechart=self.server_statechart,
            tick_count=self.server_executor.tick_count,
        )
        return from_json(json.loads(json.dumps(to_json(reported))))

    def copied_node(self, *, name: str) -> CountTicks:
        """
        :return: The node of :attr:`copy` with the given name.
        """
        return next(node for node in self.copy.nodes if node.name == name)


# %% helpers


def copy_of(*, statechart: Statechart) -> Statechart:
    """
    :return: A copy of `statechart`, read in another context from its structure after a
        trip through a JSON string.
    """
    return Statechart.from_json(
        data=json.loads(json.dumps(statechart.to_json())),
        context=StatechartContext(world=World()),
    )


@pytest.fixture
def server_and_copy(
    statechart_context: StatechartContext,
    statechart_executor: StatechartExecutor,
) -> ServerAndCopy:
    """
    A server's statechart of three counters, and a copy of it.
    """
    statechart = Statechart(context=statechart_context)
    statechart.add_nodes(
        nodes=[
            CountTicks(name="first", ticks=1),
            CountTicks(name="second", ticks=1),
            CountTicks(name="third", ticks=5),
        ]
    )
    copy = copy_of(statechart=statechart)
    statechart_executor.compile(statechart=statechart)
    return ServerAndCopy(server_executor=statechart_executor, copy=copy)


def tick(*, executor: StatechartExecutor, ticks: int) -> None:
    """
    Tick the statechart of `executor` `ticks` times.
    """
    for _ in range(ticks):
        executor.tick()


# %% taking over the reported states


def test_the_reported_states_reach_the_nodes_of_the_copy(
    server_and_copy: ServerAndCopy,
) -> None:
    """
    Every state the server reports is written into the node of the copy with the same
    id.
    """
    tick(executor=server_and_copy.server_executor, ticks=2)

    server_and_copy.report().apply_to(statechart=server_and_copy.copy)

    assert server_and_copy.copy.life_cycle_state == (
        server_and_copy.server_statechart.life_cycle_state
    )
    assert server_and_copy.copy.observation_state == (
        server_and_copy.server_statechart.observation_state
    )
    assert server_and_copy.copy.last_observation_state == (
        server_and_copy.server_statechart.last_observation_state
    )


def test_a_node_at_another_index_on_the_server_takes_over_its_own_state(
    statechart_context: StatechartContext,
    statechart_executor: StatechartExecutor,
) -> None:
    """
    The server removed a node the copy still holds, so every later node has a smaller
    index on the server than in the copy.

    The kept node runs on the server by then.
    """
    statechart = Statechart(context=statechart_context)
    removed = CountTicks(name="removed", ticks=5)
    statechart.add_nodes(nodes=[removed, (kept := CountTicks(name="kept", ticks=5))])
    server_and_copy = ServerAndCopy(
        server_executor=statechart_executor, copy=copy_of(statechart=statechart)
    )
    statechart.remove_node(node=removed)
    statechart_executor.compile(statechart=statechart)
    tick(executor=statechart_executor, ticks=2)

    server_and_copy.report().apply_to(statechart=server_and_copy.copy)

    assert (
        server_and_copy.copied_node(name="removed").life_cycle_state,
        server_and_copy.copied_node(name="kept").life_cycle_state,
    ) == (LifeCycleValue.NOT_STARTED, kept.life_cycle_state)


def test_a_node_the_server_does_not_report_keeps_its_state(
    server_and_copy: ServerAndCopy,
) -> None:
    """
    A node the receiver added, which the server does not hold yet, keeps its state.
    """
    server_and_copy.copy.add_node(node=(added := CountTicks(name="added", ticks=1)))
    tick(executor=server_and_copy.server_executor, ticks=2)

    server_and_copy.report().apply_to(statechart=server_and_copy.copy)

    assert added.life_cycle_state == LifeCycleValue.NOT_STARTED


def test_a_reported_node_the_copy_does_not_hold_is_rejected(
    server_and_copy: ServerAndCopy,
) -> None:
    """
    A node joined the server after the copy was read, and the copy never learned of it.
    """
    server_and_copy.server_statechart.add_node(node=CountTicks(name="new", ticks=1))

    with pytest.raises(ReportedNodeNotInStatechartError):
        server_and_copy.report().apply_to(statechart=server_and_copy.copy)


# %% the history of the copy


def test_the_observers_of_the_copy_learn_the_reported_tick(
    server_and_copy: ServerAndCopy,
) -> None:
    """
    Taking over the states records them in the history of the copy at the reported tick.
    """
    observer = ObserverRecordingTicks()
    server_and_copy.copy.state_history.add_observer(observer=observer)
    tick(executor=server_and_copy.server_executor, ticks=3)
    report = server_and_copy.report()

    report.apply_to(statechart=server_and_copy.copy)

    assert observer.ticks == [report.tick_count]


def test_the_same_states_reported_twice_are_recorded_once(
    server_and_copy: ServerAndCopy,
) -> None:
    """
    A report repeating the states of the one before adds nothing to the history.
    """
    tick(executor=server_and_copy.server_executor, ticks=3)
    report = server_and_copy.report()

    report.apply_to(statechart=server_and_copy.copy)
    report.apply_to(statechart=server_and_copy.copy)

    assert len(server_and_copy.copy.state_history.items) == 1
