from __future__ import annotations

import pytest

from cramph.composites import (
    PausedWhileTrue,
    ControlStructureNode,
    Parallel,
    Sequence,
    TryAll,
    TryInOrder,
)
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.exceptions import (
    NodeAlreadyAChildError,
    NodeAlreadyBelongsToDifferentNodeError,
    NodeIsNotAChildError,
    NotHeldByControlStructureNodeError,
    NoControlStructureNodeAboveError,
    StatechartAlreadyCompiledError,
)
from cramph.executor import StatechartExecutor
from cramph.node import EndStatechart, StatechartNode
from cramph.nodes_for_testing import (
    CompositeNodeWithChainedChildren,
    ConstFalseNode,
    ConstTrueNode,
    NodeFailingOnObservingFalse,
    NodeObservingAFixedValue,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart

# %% helpers


def _succeeding(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds as soon as it observes, which it does right away.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


def _failing(name: str) -> NodeFailingOnObservingFalse:
    """
    :return: A node that fails as soon as it observes, which it does right away.
    """
    return NodeFailingOnObservingFalse(
        name=name, observation=ObservationStateValues.FALSE
    )


def _run(
    executor: StatechartExecutor, statechart: Statechart, goal: StatechartNode
) -> None:
    """
    Compiles `statechart` with an end once `goal` ended, and ticks it until then.
    """
    statechart.add_node(EndStatechart.when_true(goal))
    executor.compile(statechart)
    executor.tick_until_end(timeout=100)


def _start_order(
    statechart: Statechart, nodes: list[StatechartNode]
) -> list[StatechartNode]:
    """
    :return: Those of `nodes` that started, ordered by the tick they started in.
    """
    first_running_tick = {}
    for node in nodes:
        history = statechart.state_history.get_life_cycle_history_of_node(node)
        started = [
            tick
            for tick, state in enumerate(history)
            if state != LifeCycleValues.NOT_STARTED
        ]
        if started:
            first_running_tick[node] = started[0]
    assert len(set(first_running_tick.values())) == len(first_running_tick)
    return sorted(first_running_tick, key=first_running_tick.get)


# %% sequence


def test_a_step_inserted_before_another_runs_right_before_it(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, second, third = _succeeding("1"), _succeeding("2"), _succeeding("3")
    statechart.add_node(sequence := Sequence(nodes=[first, third]))

    sequence.insert_before(third, second)
    _run(statechart_executor, statechart, sequence)

    assert _start_order(statechart, [third, second, first]) == [first, second, third]
    assert sequence.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_step_inserted_before_the_first_one_runs_first(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, second = _succeeding("1"), _succeeding("2")
    statechart.add_node(sequence := Sequence(nodes=[second]))

    sequence.insert_before(second, first)
    _run(statechart_executor, statechart, sequence)

    assert _start_order(statechart, [second, first]) == [first, second]


def test_a_step_inserted_after_the_last_one_runs_last(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, second, third = _succeeding("1"), _succeeding("2"), _succeeding("3")
    statechart.add_node(sequence := Sequence(nodes=[first, second]))

    sequence.insert_after(second, third)
    _run(statechart_executor, statechart, sequence)

    assert _start_order(statechart, [third, second, first]) == [first, second, third]


def test_a_step_inserted_after_another_runs_right_after_it(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, second, third = _succeeding("1"), _succeeding("2"), _succeeding("3")
    statechart.add_node(sequence := Sequence(nodes=[first, third]))

    sequence.insert_after(first, second)
    _run(statechart_executor, statechart, sequence)

    assert _start_order(statechart, [third, second, first]) == [first, second, third]


def test_a_replaced_step_leaves_the_statechart_and_its_replacement_runs_in_its_place(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, replaced, third = _succeeding("1"), _succeeding("x"), _succeeding("3")
    second = _succeeding("2")
    statechart.add_node(sequence := Sequence(nodes=[first, replaced, third]))

    statechart.replace(replaced, second)
    _run(statechart_executor, statechart, sequence)

    assert replaced not in statechart.nodes
    assert _start_order(statechart, [third, second, first]) == [first, second, third]


def test_a_replaced_node_of_a_step_no_longer_decides_when_the_step_succeeds(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    The sequence ends the parallel once it is at its goal, which the parallel only is
    while none of its nodes failed, so it has to stop asking the node that left.
    """
    statechart = Statechart(context=statechart_executor.context)
    replaced, kept, replacement = _succeeding("x"), _succeeding("1"), _succeeding("2")
    parallel = Parallel(nodes=[replaced, kept])
    statechart.add_node(sequence := Sequence(nodes=[parallel]))

    statechart.replace(replaced, replacement)
    _run(statechart_executor, statechart, sequence)

    assert replaced not in statechart.nodes
    assert parallel.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_sequence_that_did_not_join_yet_only_changes_its_list(
    statechart_executor: StatechartExecutor,
) -> None:
    first, second, third = _succeeding("1"), _succeeding("2"), _succeeding("3")
    sequence = Sequence(nodes=[first, third])

    sequence.insert_before(third, second)

    assert sequence.nodes == [first, second, third]
    assert not second.belongs_to_statechart()


# %% trying alternatives in order


def test_an_inserted_alternative_starts_once_the_one_before_it_failed(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, second, third = _failing("1"), _failing("2"), _succeeding("3")
    statechart.add_node(alternatives := TryInOrder(nodes=[first, third]))

    alternatives.insert_after(first, second)
    _run(statechart_executor, statechart, alternatives)

    assert _start_order(statechart, [third, second, first]) == [first, second, third]
    assert alternatives.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% children running side by side


def test_an_alternative_inserted_into_try_all_can_decide_it(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    failing = _failing("fails")
    statechart.add_node(alternatives := TryAll(nodes=[failing]))

    alternatives.insert_after(failing, _succeeding("works"))
    _run(statechart_executor, statechart, alternatives)

    assert alternatives.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_node_inserted_into_a_parallel_counts_towards_its_observation(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    at_goal = NodeObservingAFixedValue(observation=ObservationStateValues.TRUE)
    statechart.add_node(parallel := Parallel(nodes=[at_goal]))

    parallel.insert_after(
        at_goal, NodeObservingAFixedValue(observation=ObservationStateValues.FALSE)
    )
    statechart_executor.compile(statechart)
    statechart_executor.tick()

    assert parallel.observation_state == ObservationStateValues.FALSE


def test_a_parallel_can_still_arrive_through_a_node_inserted_after_it_joined(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    failing = _failing("fails")
    statechart.add_node(parallel := Parallel(nodes=[failing], minimum_success=1))

    parallel.insert_after(
        failing, NodeObservingAFixedValue(observation=ObservationStateValues.TRUE)
    )
    statechart_executor.compile(statechart)
    statechart_executor.tick()
    statechart_executor.tick()

    assert failing.life_cycle_state == LifeCycleValues.FAILED
    assert parallel.life_cycle_state == LifeCycleValues.RUNNING
    assert parallel.observation_state == ObservationStateValues.TRUE


# %% replacing a node by one holding it


def _at_goal(name: str) -> NodeObservingAFixedValue:
    """
    :return: A node that observes being at its goal and leaves ending it to its parent.
    """
    return NodeObservingAFixedValue(name=name, observation=ObservationStateValues.TRUE)


def test_a_step_replaced_by_a_node_holding_it_stays_in_the_statechart_below_that_node(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, held, third = _succeeding("1"), _at_goal("held"), _succeeding("3")
    beside = _at_goal("beside")
    statechart.add_node(sequence := Sequence(nodes=[first, held, third]))

    statechart.replace(held, replacement := Parallel(nodes=[beside, held]))
    _run(statechart_executor, statechart, sequence)

    assert sequence.nodes == [first, replacement, third]
    assert replacement.nodes == [beside, held]
    assert held.parent_node is replacement
    assert held in statechart.nodes
    assert _start_order(statechart, [third, replacement, first]) == [
        first,
        replacement,
        third,
    ]
    assert sequence.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_held_step_keeps_its_position_among_the_children_of_its_new_parent(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    held, before, after = (
        _succeeding("held"),
        _succeeding("before"),
        _succeeding("after"),
    )
    statechart.add_node(sequence := Sequence(nodes=[held]))

    statechart.replace(held, replacement := Sequence(nodes=[before, held, after]))
    _run(statechart_executor, statechart, sequence)

    assert replacement.nodes == [before, held, after]
    assert _start_order(statechart, [after, held, before]) == [before, held, after]


def test_a_held_step_is_no_longer_ended_by_its_former_parent(
    statechart_executor: StatechartExecutor,
) -> None:
    """
    The sequence ended the step once it was at its goal; a parallel ends none of its
    children, so below one the step keeps running while its sibling is not at its goal.
    """
    statechart = Statechart(context=statechart_executor.context)
    held = _at_goal("held")
    statechart.add_node(Sequence(nodes=[held]))
    statechart.replace(
        held,
        Parallel(
            nodes=[
                NodeObservingAFixedValue(observation=ObservationStateValues.FALSE),
                held,
            ]
        ),
    )

    statechart_executor.compile(statechart)
    statechart_executor.tick()
    statechart_executor.tick()

    assert held.life_cycle_state == LifeCycleValues.RUNNING


def test_replacing_a_step_of_a_compiled_statechart_is_rejected_and_changes_nothing(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    held = _at_goal("held")
    statechart.add_node(sequence := Sequence(nodes=[held]))
    statechart.add_node(EndStatechart.when_true(sequence))
    statechart_executor.compile(statechart)
    replacement = Parallel(nodes=[_at_goal("beside"), held])

    with pytest.raises(StatechartAlreadyCompiledError):
        statechart.replace(held, replacement)

    assert sequence.nodes == [held]
    assert replacement.nodes[1] is held
    assert held.parent_node is sequence


# %% replacing a top-level node


def test_a_top_level_node_replaced_by_a_node_holding_it_stays_below_that_node(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(held := _succeeding("held"))
    beside = _at_goal("beside")

    statechart.replace(held, replacement := Sequence(nodes=[held, beside]))
    _run(statechart_executor, statechart, replacement)

    assert statechart.top_level_nodes[0] is replacement
    assert replacement.nodes == [held, beside]
    assert held.parent_node is replacement
    assert replacement.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_replaced_top_level_node_leaves_the_statechart(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(replaced := Sequence(nodes=[_succeeding("below")]))

    statechart.replace(replaced, replacement := _succeeding("instead"))

    assert statechart.nodes == [replacement]
    assert not replaced.belongs_to_statechart()


def test_replacing_a_top_level_node_of_a_compiled_statechart_is_rejected(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(held := _at_goal("held"))
    statechart_executor.compile(statechart)
    replacement = Parallel(nodes=[_at_goal("beside"), held])

    with pytest.raises(StatechartAlreadyCompiledError):
        statechart.replace(held, replacement)

    assert held.parent_node is None
    assert statechart.top_level_nodes == [held]


# %% the closest control structure node above a node


def test_the_closest_language_node_above_a_nested_node_is_the_first_one_on_its_path(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    nested = _succeeding("nested")
    paused = PausedWhileTrue(
        name="paused", monitor=ConstFalseNode(), monitored_node=nested
    )
    statechart.add_node(sequence := Sequence(nodes=[paused]))

    assert ControlStructureNode.closest_above(nested) is sequence


def test_a_step_inserted_before_a_nested_node_runs_before_what_holds_it(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, nested = _succeeding("1"), _succeeding("nested")
    paused = PausedWhileTrue(
        name="paused", monitor=ConstFalseNode(), monitored_node=nested
    )
    statechart.add_node(sequence := Sequence(nodes=[paused]))

    ControlStructureNode.closest_above(nested).insert_before(nested, first)
    _run(statechart_executor, statechart, sequence)

    assert _start_order(statechart, [paused, first]) == [first, paused]


def test_a_node_without_a_language_node_above_it_has_none_closest_above_it(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(top_level := _succeeding("top level"))

    with pytest.raises(NoControlStructureNodeAboveError):
        ControlStructureNode.closest_above(top_level)


# %% the step holding a node


def test_the_step_holding_a_nested_node_is_the_child_of_the_closest_language_node(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    nested = _succeeding("nested")
    paused = PausedWhileTrue(
        name="paused", monitor=ConstFalseNode(), monitored_node=nested
    )
    statechart.add_node(Sequence(nodes=[paused]))

    assert ControlStructureNode.step_holding(nested) is paused


def test_a_step_holds_itself(statechart_executor: StatechartExecutor) -> None:
    statechart = Statechart(context=statechart_executor.context)
    step = _succeeding("step")
    statechart.add_node(Sequence(nodes=[step]))

    assert ControlStructureNode.step_holding(step) is step


def test_without_a_language_node_above_the_top_level_node_holds_a_node(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    nested = _succeeding("nested")
    paused = PausedWhileTrue(
        name="paused", monitor=ConstFalseNode(), monitored_node=nested
    )
    statechart.add_node(paused)

    assert ControlStructureNode.step_holding(nested) is paused


# %% a failed replacement


def test_a_failed_replacement_by_a_node_holding_the_step_leaves_it_where_it_was(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, second = _succeeding("1"), _succeeding("2")
    statechart.add_node(sequence := Sequence(nodes=[first, second]))
    replacement = Sequence(nodes=[first, second])

    with pytest.raises(NodeAlreadyBelongsToDifferentNodeError):
        statechart.replace(second, replacement)
    _run(statechart_executor, statechart, sequence)

    assert not replacement.belongs_to_statechart()
    assert sequence.nodes == [first, second]
    assert second.parent_node is sequence
    assert _start_order(statechart, [second, first]) == [first, second]


def test_a_failed_replacement_by_another_node_leaves_the_step_where_it_was(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first, second = _succeeding("1"), _succeeding("2")
    statechart.add_node(sequence := Sequence(nodes=[first, second]))
    replacement = Sequence(nodes=[first])

    with pytest.raises(NodeAlreadyBelongsToDifferentNodeError):
        statechart.replace(second, replacement)
    _run(statechart_executor, statechart, sequence)

    assert not replacement.belongs_to_statechart()
    assert sequence.nodes == [first, second]
    assert second.parent_node is sequence
    assert _start_order(statechart, [second, first]) == [first, second]


def test_only_a_node_a_language_node_or_the_top_level_holds_can_be_replaced(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    holder = CompositeNodeWithChainedChildren(name="holder")
    statechart.add_node(holder)
    nested = holder.sub_node1

    with pytest.raises(NotHeldByControlStructureNodeError) as raised:
        statechart.replace(nested, Sequence(nodes=[_succeeding("1"), nested]))

    assert raised.value.node is nested


# %% misuse


def test_inserting_next_to_a_node_that_is_not_below_the_goal_is_rejected(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(sequence := Sequence(nodes=[_succeeding("1")]))

    with pytest.raises(NodeIsNotAChildError):
        sequence.insert_before(_succeeding("stranger"), _succeeding("2"))


def test_inserting_a_node_that_is_a_child_of_the_goal_already_is_rejected(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first = _succeeding("1")
    statechart.add_node(sequence := Sequence(nodes=[first]))

    with pytest.raises(NodeAlreadyAChildError):
        sequence.insert_after(first, first)
