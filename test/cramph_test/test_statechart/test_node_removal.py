from __future__ import annotations

import pytest

from cramph.composites import Sequence
from cramph.data_types import LifeCycleValues
from cramph.exceptions import (
    RemovedNodeStillReferencedError,
    StatechartAlreadyCompiledError,
)
from cramph.executor import StatechartExecutor
from cramph.node import CancelStatechart, EndStatechart
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import Statechart

# %% removing nodes


def test_removing_a_node_keeps_the_indices_of_the_others_contiguous(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([first := ConstTrueNode(), removed := ConstTrueNode()])
    statechart.add_node(last := ConstTrueNode())

    statechart.remove_node(removed)

    assert statechart.nodes == [first, last]
    assert [node.index for node in statechart.nodes] == [0, 1]
    assert statechart.get_node_by_index(last.index) is last


def test_removing_a_node_shrinks_the_state_of_the_statechart(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([ConstTrueNode(), removed := ConstTrueNode()])

    statechart.remove_node(removed)

    assert len(statechart.life_cycle_state.data) == len(statechart.nodes)
    assert len(statechart.observation_state.data) == len(statechart.nodes)
    assert len(statechart.last_observation_state.data) == len(statechart.nodes)


def test_a_removed_node_no_longer_belongs_to_the_statechart(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(removed := ConstTrueNode())

    statechart.remove_node(removed)

    assert not removed.belongs_to_statechart()
    assert removed.index is None


def test_removing_a_child_takes_its_descendants_along(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(ConstTrueNode())
    statechart.add_node(sequence := Sequence(nodes=[task := ConstTrueNode()]))
    statechart.add_node(later := ConstTrueNode())

    statechart.remove_node(task)

    assert task not in statechart.nodes
    assert sequence.nodes == []
    assert later.parent_node is None
    assert statechart.get_node_by_index(later.index) is later


def test_a_child_keeps_its_parent_when_a_node_before_them_is_removed(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(removed := ConstTrueNode())
    statechart.add_node(sequence := Sequence(nodes=[task := ConstTrueNode()]))

    statechart.remove_node(removed)

    assert task.parent_node is sequence


def test_removing_a_terminal_node_forgets_it_as_a_way_to_end(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node := ConstTrueNode())
    statechart.add_node(end := EndStatechart.when_true(node))
    statechart.add_node(cancel := CancelStatechart.when_true(node, Exception("x")))

    statechart.remove_node(end)
    statechart.remove_node(cancel)

    assert statechart._end_nodes == []
    assert statechart._cancel_nodes == []


def test_a_node_another_node_still_refers_to_cannot_be_removed(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(referenced := ConstTrueNode())
    statechart.add_node(EndStatechart.when_true(referenced))

    with pytest.raises(RemovedNodeStillReferencedError):
        statechart.remove_node(referenced)


def test_a_compiled_statechart_keeps_its_nodes(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node := ConstTrueNode())
    statechart_executor.compile(statechart)

    with pytest.raises(StatechartAlreadyCompiledError):
        statechart.remove_node(node)


def test_a_node_no_compile_covered_yet_leaves_a_compiled_statechart(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(compiled := ConstTrueNode())
    statechart_executor.compile(statechart)

    with statechart.modify():
        statechart.add_nodes([removed := ConstTrueNode(), kept := ConstTrueNode()])
        statechart.remove_node(removed)

    assert statechart.nodes == [compiled, kept]
    assert [node.index for node in statechart.nodes] == [0, 1]
    assert statechart.compiler.latest_compile_covers(kept)


def test_a_compiled_statechart_runs_after_a_joining_node_was_removed_again(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(ConstTrueNode())
    statechart_executor.compile(statechart)
    statechart_executor.tick()

    with statechart.modify():
        statechart.add_node(removed := ConstTrueNode())
        statechart.add_node(sequence := Sequence(nodes=[ConstTrueNode()]))
        statechart.add_node(EndStatechart.when_true(sequence))
        statechart.remove_node(removed)
    statechart_executor.tick_until_end()

    assert sequence.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_statechart_runs_after_a_node_was_removed(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(removed := ConstTrueNode())
    statechart.add_node(sequence := Sequence(nodes=[ConstTrueNode()]))
    statechart.add_node(EndStatechart.when_true(sequence))

    statechart.remove_node(removed)
    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end()

    assert sequence.life_cycle_state == LifeCycleValues.SUCCEEDED
