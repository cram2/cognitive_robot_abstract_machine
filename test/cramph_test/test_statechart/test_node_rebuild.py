from __future__ import annotations

from cramph.composites import Parallel, RepeatUntil, Sequence, StartedWhenTrue
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.monitors import CountNodeResets
from cramph.node import DerivedConditionVariable, EndStatechart
from cramph.nodes_for_testing import (
    NodeFailingOnObservingFalse,
    NodeObservingLastObservation,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart

from .node_rebuild import RebuildLeavingNodesOut

# %% helpers


def _succeeding(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds on the tick after it starts.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


# %% building a node anew


def test_a_sequence_is_built_anew_holding_its_children_built_anew() -> None:
    first, second = _succeeding("first"), _succeeding("second")
    sequence = Sequence(name="sequence", nodes=[first, second])
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(sequence)

    assert rebuilt is not sequence
    assert rebuilt.name == sequence.name
    assert rebuilt.nodes == [rebuild.built[first], rebuild.built[second]]
    assert all(rebuild.built[child] is not child for child in [first, second])


def test_a_sequence_holding_no_child_built_anew_is_left_out() -> None:
    child = _succeeding("child")
    rebuild = RebuildLeavingNodesOut(left_out=[child])

    assert rebuild.rebuilt(Sequence(nodes=[child])) is None


def test_a_parallel_built_anew_needs_as_many_successes() -> None:
    parallel = Parallel(nodes=[_succeeding("a"), _succeeding("b")], minimum_success=1)

    rebuilt = RebuildLeavingNodesOut().rebuilt(parallel)

    assert rebuilt.minimum_success == parallel.minimum_success


def test_a_monitored_node_is_built_anew_with_its_monitor_built_anew() -> None:
    monitor, monitored_node = _succeeding("monitor"), _succeeding("monitored")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(
        StartedWhenTrue(monitor=monitor, monitored_node=monitored_node)
    )

    assert rebuilt.monitor is rebuild.built[monitor]
    assert rebuilt.monitored_node is rebuild.built[monitored_node]


def test_a_monitored_node_without_its_monitored_node_is_left_out() -> None:
    monitored_node = _succeeding("monitored")
    rebuild = RebuildLeavingNodesOut(left_out=[monitored_node])

    rebuilt = rebuild.rebuilt(
        StartedWhenTrue(monitor=_succeeding("monitor"), monitored_node=monitored_node)
    )

    assert rebuilt is None


def test_a_monitor_counting_resets_counts_those_of_the_node_built_anew() -> None:
    counted = _succeeding("counted")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(CountNodeResets(node=counted, target=2))

    assert rebuilt.node is rebuild.built[counted]
    assert rebuilt.target == 2


def test_a_repeated_task_is_built_anew_with_its_stop_monitor_built_anew() -> None:
    task = NodeFailingOnObservingFalse(
        name="task", observation=ObservationStateValues.FALSE
    )
    stop_retry_monitor = _succeeding("stop")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(
        RepeatUntil(task=task, stop_retry_monitor=stop_retry_monitor)
    )

    assert rebuilt.task is rebuild.built[task]
    assert rebuilt.stop_retry_monitor is rebuild.built[stop_retry_monitor]


def test_a_repeated_task_without_its_stop_monitor_is_left_out() -> None:
    stop_retry_monitor = _succeeding("stop")
    rebuild = RebuildLeavingNodesOut(left_out=[stop_retry_monitor])

    rebuilt = rebuild.rebuilt(
        RepeatUntil(
            task=NodeFailingOnObservingFalse(observation=ObservationStateValues.FALSE),
            stop_retry_monitor=stop_retry_monitor,
        )
    )

    assert rebuilt is None


def test_a_node_reading_another_node_reads_it_built_anew() -> None:
    watched_node = _succeeding("watched")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(NodeObservingLastObservation(watched_node=watched_node))

    assert rebuilt.watched_node is rebuild.built[watched_node]


def test_a_sequence_keeps_its_children_that_are_not_left_out() -> None:
    first, second = _succeeding("first"), _succeeding("second")
    rebuild = RebuildLeavingNodesOut(left_out=[first])

    rebuilt = rebuild.rebuilt(Sequence(nodes=[first, second]))

    assert rebuilt.nodes == [rebuild.built[second]]


# %% a condition reading the counterparts of the nodes it reads


def test_a_condition_reads_the_counterpart_of_a_node_it_reads() -> None:
    read, counterpart = _succeeding("read"), _succeeding("counterpart")

    carried_over = DerivedConditionVariable.reading_counterparts(
        read.is_succeeded, lambda node: counterpart
    )

    assert carried_over.free_variables() == [counterpart.is_succeeded]


def test_a_condition_reads_a_node_without_counterpart_as_it_is_now(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    ended = _succeeding("ended")
    statechart.add_node(ended)
    statechart.add_node(EndStatechart.when_true(ended))
    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end(timeout=30)
    assert ended.life_cycle_state == LifeCycleValues.SUCCEEDED

    succeeded = DerivedConditionVariable.reading_counterparts(
        ended.is_succeeded, lambda node: None
    )
    failed = DerivedConditionVariable.reading_counterparts(
        ended.is_failed, lambda node: None
    )

    assert succeeded.is_constant_true()
    assert failed.is_constant_false()
