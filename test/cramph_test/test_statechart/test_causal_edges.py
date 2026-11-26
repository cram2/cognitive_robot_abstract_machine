"""
Tests for the runs of a statechart's nodes, :mod:`cramph.node_runs`, and the causal
edges :mod:`cramph.causal_edges` reads from the statechart.
"""

from __future__ import annotations

from cramph.causal_edges import (
    ContainmentEdge,
    ContainmentEdges,
    ConditionEdge,
    ConditionEdges,
    WorldAccessOfNodes,
)
from cramph.composites import Sequence
from cramph.data_types import LifeCycleValue, TransitionKind
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.node import EndStatechart
from cramph.node_runs import StatechartRuns
from cramph.state_history import NodeRun
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

    [run] = runs.runs_by_node[step]
    assert run.sequence_number == 1
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
    edges = ContainmentEdges().find_edges(
        runs=runs, access=WorldAccessOfNodes(runs=runs)
    )

    [sequence_run] = runs.runs_by_node[sequence]
    assert {
        ContainmentEdge(cause=runs.latest_run_of_node(node=child), effect=sequence_run)
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
    edges = ConditionEdges().find_edges(runs=runs, access=WorldAccessOfNodes(runs=runs))

    assert (
        ConditionEdge(
            cause=runs.latest_run_of_node(node=trigger),
            effect=runs.latest_run_of_node(node=started),
            transition_kind=TransitionKind.START,
        )
        in edges
    )


def interrupted_by_a_later_node(
    *, executor: StatechartExecutor
) -> tuple[StatechartRuns, CountTicks, CountTicks]:
    """
    :return: The runs of a statechart in which a waiting node is interrupted by a node
        that started after it, the waiting node, and the interrupting node.
    """
    waiting, first, later = (
        CountTicks(ticks=100),
        CountTicks(ticks=2),
        CountTicks(ticks=1),
    )
    later.start_condition = first.observes_true
    waiting.interrupt_condition = later.observes_true
    timer = CountTicks(ticks=10)
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(nodes=[waiting, first, later, timer])
    statechart.add_node(node=EndStatechart.when_true(node=timer))
    return ran_to_its_end(executor=executor, statechart=statechart), waiting, later


def test_an_interrupt_condition_reads_the_run_that_had_started_when_its_node_ended(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    A node interrupted by a node that started after it is acted on by that node's run.
    """
    runs, waiting, later = interrupted_by_a_later_node(executor=statechart_executor)
    edges = ConditionEdges().find_edges(runs=runs, access=WorldAccessOfNodes(runs=runs))

    assert (
        ConditionEdge(
            cause=runs.latest_run_of_node(node=later),
            effect=runs.latest_run_of_node(node=waiting),
            transition_kind=TransitionKind.INTERRUPT,
        )
        in edges
    )


def test_a_condition_held_only_at_the_ticks_its_nodes_made_it_true(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    An interrupt condition held on the tick it ended its node, not on the tick the node
    started.
    """
    runs, waiting, _ = interrupted_by_a_later_node(executor=statechart_executor)
    [condition] = [
        condition
        for condition in waiting.conditions
        if condition.kind is TransitionKind.INTERRUPT
    ]
    run = runs.latest_run_of_node(node=waiting)
    history = runs.statechart.state_history

    assert [
        condition.was_true_at_tick(history=history, tick=tick)
        for tick in (run.start_tick, run.end_tick)
    ] == [False, True]


def test_runs_that_only_touch_at_one_tick_do_not_overlap() -> None:
    """
    A run that ends on the tick another starts ended before it.
    """
    node = CountTicks(ticks=1)
    earlier, later = (
        NodeRun(
            node=node,
            sequence_number=number,
            start_tick=start_tick,
            end_tick=end_tick,
            final_state=LifeCycleValue.SUCCEEDED,
        )
        for number, start_tick, end_tick in [(1, 0, 3), (2, 3, 5)]
    )

    assert (earlier.overlaps(other=later), later.overlaps(other=earlier)) == (
        False,
        False,
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

    assert first.is_ancestor_or_descendant_of(other=first)
    assert sequence.is_ancestor_or_descendant_of(other=first)
    assert first.is_ancestor_or_descendant_of(other=sequence)
    assert not first.is_ancestor_or_descendant_of(other=second)
