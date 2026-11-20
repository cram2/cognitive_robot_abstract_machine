from __future__ import annotations

from cramph.composites import Parallel, RepeatUntil, Sequence, StartedWhenTrue
from cramph.data_types import LifeCycleValue, ObservationStateValue
from cramph.executor import StatechartExecutor
from cramph.monitors import CountNodeResets
from cramph.node import DerivedConditionVariable, EndStatechart
from cramph.nodes_for_testing import (
    ConstFalseNode,
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
        name=name, observation=ObservationStateValue.TRUE
    )


def _parallel_with_a_child_of_each_outcome(
    statechart_executor: StatechartExecutor,
    minimum_success: int,
    maximum_ticks: int = 10,
) -> Parallel:
    """
    :param maximum_ticks: How many ticks the children get to end in.
    :return: A parallel needing `minimum_success` successes, run until its first child
        succeeded and its second failed, while its third still runs.
    """
    parallel = Parallel(
        nodes=[
            _succeeding("succeeded"),
            NodeFailingOnObservingFalse(
                name="failed", observation=ObservationStateValue.FALSE
            ),
            ConstFalseNode(name="running"),
        ],
        minimum_success=minimum_success,
    )
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=parallel)
    statechart_executor.compile(statechart=statechart)
    succeeded, failed, running = parallel.nodes
    for _ in range(maximum_ticks):
        if (
            succeeded.life_cycle_state == LifeCycleValue.SUCCEEDED
            and failed.life_cycle_state == LifeCycleValue.FAILED
        ):
            break
        statechart_executor.tick()
    assert [child.life_cycle_state for child in parallel.nodes] == [
        LifeCycleValue.SUCCEEDED,
        LifeCycleValue.FAILED,
        LifeCycleValue.RUNNING,
    ]
    return parallel


# %% building a node anew


def test_a_sequence_is_built_anew_holding_its_children_built_anew() -> None:
    first, second = _succeeding("first"), _succeeding("second")
    sequence = Sequence(name="sequence", nodes=[first, second])
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(node=sequence)

    assert rebuilt is not sequence
    assert rebuilt.name == sequence.name
    assert rebuilt.nodes == [rebuild.built[first], rebuild.built[second]]
    assert all(rebuild.built[child] is not child for child in [first, second])


def test_a_sequence_holding_no_child_built_anew_is_left_out() -> None:
    child = _succeeding("child")
    rebuild = RebuildLeavingNodesOut(left_out=[child])

    assert rebuild.rebuilt(node=Sequence(nodes=[child])) is None


def test_a_parallel_built_anew_needs_as_many_successes() -> None:
    parallel = Parallel(nodes=[_succeeding("a"), _succeeding("b")], minimum_success=1)

    rebuilt = RebuildLeavingNodesOut().rebuilt(node=parallel)

    assert rebuilt.minimum_success == parallel.minimum_success


def test_a_parallel_needs_one_success_fewer_without_a_child_that_succeeded(
    statechart_executor: StatechartExecutor,
) -> None:
    parallel = _parallel_with_a_child_of_each_outcome(statechart_executor, 2)
    succeeded, _, _ = parallel.nodes

    rebuilt = RebuildLeavingNodesOut(left_out=[succeeded]).rebuilt(node=parallel)

    assert rebuilt.minimum_success == parallel.minimum_success - 1


def test_a_parallel_needs_as_many_successes_without_a_child_that_failed(
    statechart_executor: StatechartExecutor,
) -> None:
    parallel = _parallel_with_a_child_of_each_outcome(statechart_executor, 2)
    _, failed, _ = parallel.nodes

    rebuilt = RebuildLeavingNodesOut(left_out=[failed]).rebuilt(node=parallel)

    assert rebuilt.minimum_success == parallel.minimum_success


def test_a_parallel_needs_one_success_fewer_without_a_child_still_running(
    statechart_executor: StatechartExecutor,
) -> None:
    parallel = _parallel_with_a_child_of_each_outcome(statechart_executor, 2)
    _, _, running = parallel.nodes

    rebuilt = RebuildLeavingNodesOut(left_out=[running]).rebuilt(node=parallel)

    assert rebuilt.minimum_success == parallel.minimum_success - 1


def test_a_parallel_needs_no_success_once_more_children_are_left_out_than_it_needs(
    statechart_executor: StatechartExecutor,
) -> None:
    parallel = _parallel_with_a_child_of_each_outcome(statechart_executor, 1)
    succeeded, _, running = parallel.nodes

    rebuilt = RebuildLeavingNodesOut(left_out=[succeeded, running]).rebuilt(
        node=parallel
    )

    assert rebuilt.minimum_success == 0


def test_a_monitored_node_is_built_anew_with_its_monitor_built_anew() -> None:
    monitor, monitored_node = _succeeding("monitor"), _succeeding("monitored")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(
        node=StartedWhenTrue(monitor=monitor, monitored_node=monitored_node)
    )

    assert rebuilt.monitor is rebuild.built[monitor]
    assert rebuilt.monitored_node is rebuild.built[monitored_node]


def test_a_monitored_node_without_its_monitored_node_is_left_out() -> None:
    monitored_node = _succeeding("monitored")
    rebuild = RebuildLeavingNodesOut(left_out=[monitored_node])

    rebuilt = rebuild.rebuilt(
        node=StartedWhenTrue(
            monitor=_succeeding("monitor"), monitored_node=monitored_node
        )
    )

    assert rebuilt is None


def test_a_monitor_counting_resets_counts_those_of_the_node_built_anew() -> None:
    counted = _succeeding("counted")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(node=CountNodeResets(node=counted, target=2))

    assert rebuilt.node is rebuild.built[counted]
    assert rebuilt.target == 2


def test_a_repeated_task_is_built_anew_with_its_stop_monitor_built_anew() -> None:
    task = NodeFailingOnObservingFalse(
        name="task", observation=ObservationStateValue.FALSE
    )
    stop_retry_monitor = _succeeding("stop")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(
        node=RepeatUntil(task=task, stop_retry_monitor=stop_retry_monitor)
    )

    assert rebuilt.task is rebuild.built[task]
    assert rebuilt.stop_retry_monitor is rebuild.built[stop_retry_monitor]


def test_a_repeated_task_without_its_stop_monitor_is_left_out() -> None:
    stop_retry_monitor = _succeeding("stop")
    rebuild = RebuildLeavingNodesOut(left_out=[stop_retry_monitor])

    rebuilt = rebuild.rebuilt(
        node=RepeatUntil(
            task=NodeFailingOnObservingFalse(observation=ObservationStateValue.FALSE),
            stop_retry_monitor=stop_retry_monitor,
        )
    )

    assert rebuilt is None


def test_a_node_reading_another_node_reads_it_built_anew() -> None:
    watched_node = _succeeding("watched")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(
        node=NodeObservingLastObservation(watched_node=watched_node)
    )

    assert rebuilt.watched_node is rebuild.built[watched_node]


def test_a_sequence_keeps_its_children_that_are_not_left_out() -> None:
    first, second = _succeeding("first"), _succeeding("second")
    rebuild = RebuildLeavingNodesOut(left_out=[first])

    rebuilt = rebuild.rebuilt(node=Sequence(nodes=[first, second]))

    assert rebuilt.nodes == [rebuild.built[second]]


# %% a condition reading the counterparts of the nodes it reads


def test_a_condition_reads_the_counterpart_of_a_node_it_reads() -> None:
    read, counterpart = _succeeding("read"), _succeeding("counterpart")

    carried_over = DerivedConditionVariable.reading_counterparts(
        condition=read.is_succeeded, counterpart_of=lambda node: counterpart
    )

    assert carried_over.free_variables() == [counterpart.is_succeeded]


def test_a_condition_reads_a_node_without_counterpart_as_it_is_now(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    ended = _succeeding("ended")
    statechart.add_node(node=ended)
    statechart.add_node(node=EndStatechart.when_true(node=ended))
    statechart_executor.compile(statechart=statechart)
    statechart_executor.tick_until_end(timeout=30)
    assert ended.life_cycle_state == LifeCycleValue.SUCCEEDED

    succeeded = DerivedConditionVariable.reading_counterparts(
        condition=ended.is_succeeded, counterpart_of=lambda node: None
    )
    failed = DerivedConditionVariable.reading_counterparts(
        condition=ended.is_failed, counterpart_of=lambda node: None
    )

    assert succeeded.is_constant_true()
    assert failed.is_constant_false()
