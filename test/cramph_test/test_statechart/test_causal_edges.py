"""
Tests for the runs of a statechart's nodes, :mod:`cramph.node_runs`, and the causal
edges :mod:`cramph.causal_edges` reads from the statechart.
"""

from __future__ import annotations

from cramph.causal_edges import (
    ContainmentEdge,
    ContainmentEdges,
    ControlEdge,
    ControlEdges,
    WorldAccessIndex,
)
from cramph.composites import Sequence
from cramph.data_types import LifeCycleValue, TransitionKind
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import EndStatechart
from cramph.node_runs import StatechartRuns
from cramph.statechart import Statechart

# %% running a statechart


def ran_to_its_end(
    *, executor: StatechartExecutor, statechart: Statechart
) -> StatechartRuns:
    """
    :return: The runs of `statechart`, executed by `executor` until it ended.
    """
    executor.compile(statechart=statechart)
    executor.tick_until_end()
    return StatechartRuns(statechart=statechart)


# %% the runs of a node


def test_a_node_that_ran_once_has_one_run_ending_in_its_final_state(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A step of a sequence that ran once has a single run, from the tick it started on to
    the tick it succeeded on.
    """
    step = CountTicks(ticks=2)
    sequence = Sequence(nodes=[step])
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=sequence)
    statechart.add_node(node=EndStatechart.when_true(node=sequence))

    runs = ran_to_its_end(executor=statechart_executor, statechart=statechart)

    [run] = runs.runs_of_node[step]
    assert run.number == 1
    assert run.final_state == LifeCycleValue.SUCCEEDED
    assert run.end_tick is not None
    assert run.start_tick < run.end_tick


# %% edges read from the statechart


def test_the_run_of_a_child_is_part_of_the_run_of_its_parent(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    Each run of a child of a sequence acts on the run of the sequence that held it.
    """
    first, second = CountTicks(ticks=1), CountTicks(ticks=1)
    sequence = Sequence(nodes=[first, second])
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=sequence)
    statechart.add_node(node=EndStatechart.when_true(node=sequence))

    runs = ran_to_its_end(executor=statechart_executor, statechart=statechart)
    edges = ContainmentEdges().edges(runs=runs, access=WorldAccessIndex(runs=runs))

    [sequence_run] = runs.runs_of_node[sequence]
    assert {
        ContainmentEdge(cause=runs.latest_run(node=child), effect=sequence_run)
        for child in (first, second)
    } <= set(edges)


def test_a_start_condition_reading_another_node_is_a_control_edge(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A node whose start condition reads another node is acted on by the run of that node.
    """
    trigger = CountTicks(ticks=1)
    started = CountTicks(ticks=1)
    started.start_condition = trigger.observes_true
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes(nodes=[trigger, started])
    statechart.add_node(node=EndStatechart.when_true(node=started))

    runs = ran_to_its_end(executor=statechart_executor, statechart=statechart)
    edges = ControlEdges().edges(runs=runs, access=WorldAccessIndex(runs=runs))

    assert (
        ControlEdge(
            cause=runs.latest_run(node=trigger),
            effect=runs.latest_run(node=started),
            transition_kind=TransitionKind.START,
        )
        in edges
    )


def test_a_node_holds_its_children_and_is_held_by_its_parent(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A node is related to itself, to the nodes it holds and to those holding it, but not
    to its siblings.
    """
    first, second = CountTicks(ticks=1), CountTicks(ticks=1)
    sequence = Sequence(nodes=[first, second])
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=sequence)

    assert first.holds_or_is_held_by(other=first)
    assert sequence.holds_or_is_held_by(other=first)
    assert first.holds_or_is_held_by(other=sequence)
    assert not first.holds_or_is_held_by(other=second)
