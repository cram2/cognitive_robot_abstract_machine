from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cramph.node import EndedByOwner
from cramph.composites import Parallel, Sequence
from cramph.data_types import LifeCycleValues
from cramph.exceptions import (
    PrerequisiteNotExpandedError,
)
from cramph.executor import StatechartExecutor
from cramph.node import CompositeNode, StatechartNode
from cramph.nodes_for_testing import (
    ConstTrueNode,
    NodeFailingOnObservingFalse,
    NodeObservingAFixedValue,
    NodeSucceedingOnObservingTrue,
)
from cramph.data_types import ObservationStateValues
from cramph.statechart import Statechart

# %% mimics


@dataclass(eq=False, repr=False, kw_only=True)
class CompositeNodeCountingTheChildrenOfAnother(EndedByOwner, CompositeNode):
    """
    A composite node that reads the children of another composite node while it expands.
    """

    watched: CompositeNode
    """
    The composite node whose children are counted.
    """

    counted_children: int | None = field(default=None, init=False)
    """
    How many children :attr:`watched` had when this node expanded.
    """

    @property
    def prerequisite_nodes(self) -> list[StatechartNode]:
        return [self.watched]

    def expand(self) -> None:
        self.counted_children = len(self.watched.nodes)


@dataclass(eq=False, repr=False, kw_only=True)
class SequenceAddingAStepOfItsOwn(Sequence):
    """
    A sequence that adds one more step while it expands, after the ones it was handed.
    """

    def expand(self) -> None:
        self.add_node(ConstTrueNode(name="own step"))


# %% expanding a composite node once it joins


def test_a_composite_node_expands_when_it_joins_a_statechart(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    first = NodeSucceedingOnObservingTrue(observation=ObservationStateValues.TRUE)
    second = NodeSucceedingOnObservingTrue(observation=ObservationStateValues.TRUE)
    statechart.add_node(Sequence(nodes=[first, second]))

    assert statechart.nodes[1:] == [first, second]
    assert str(second.start_condition) == str(first.is_succeeded)


def test_a_nested_composite_node_expands_with_its_parent(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    step = NodeSucceedingOnObservingTrue(observation=ObservationStateValues.TRUE)
    statechart.add_node(Sequence(nodes=[inner := Sequence(nodes=[step])]))

    assert inner in statechart.nodes
    assert step in statechart.nodes
    assert step.parent_node is inner


def test_a_control_structure_node_expanding_on_its_own_adopts_the_nodes_it_was_handed(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    handed = NodeSucceedingOnObservingTrue(observation=ObservationStateValues.TRUE)
    statechart.add_node(sequence := SequenceAddingAStepOfItsOwn(nodes=[handed]))

    [own_step] = sequence.nodes[1:]
    assert handed.parent_node is sequence
    assert str(own_step.start_condition) == str(handed.is_succeeded)


def test_a_node_handed_to_a_joined_composite_node_joins_right_away(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(sequence := Sequence())
    sequence.add_node(
        first := NodeSucceedingOnObservingTrue(observation=ObservationStateValues.TRUE)
    )
    sequence.add_node(
        second := NodeSucceedingOnObservingTrue(observation=ObservationStateValues.TRUE)
    )

    assert second in statechart.nodes
    assert str(second.start_condition) == str(first.is_succeeded)


def test_a_composite_node_reads_a_prerequisite_that_joined_before_it(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(sequence := Sequence(nodes=[ConstTrueNode(), ConstTrueNode()]))
    statechart.add_node(
        counter := CompositeNodeCountingTheChildrenOfAnother(watched=sequence)
    )

    assert counter.counted_children == len(sequence.nodes)


def test_a_composite_node_joining_before_its_prerequisite_is_rejected(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    sequence = Sequence(nodes=[ConstTrueNode()])

    with pytest.raises(PrerequisiteNotExpandedError):
        statechart.add_node(CompositeNodeCountingTheChildrenOfAnother(watched=sequence))


def test_a_compiled_statechart_takes_new_nodes(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(ConstTrueNode())
    statechart_executor.compile(statechart)

    statechart.add_node(added := ConstTrueNode())

    assert added in statechart.nodes
    assert statechart.compiler.is_compiled


# %% conditions a composite node wires over its own children


def test_a_parallel_keeps_failing_on_its_children_after_its_fail_condition_is_set(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(
        parallel := Parallel(
            nodes=[
                NodeFailingOnObservingFalse(observation=ObservationStateValues.FALSE),
                NodeObservingAFixedValue(observation=ObservationStateValues.TRUE),
            ]
        )
    )
    never = NodeObservingAFixedValue(observation=ObservationStateValues.FALSE)
    statechart.add_node(never)
    parallel.fail_condition = never.observes_true
    statechart_executor.compile(statechart)
    statechart_executor.tick()

    assert parallel.life_cycle_state == LifeCycleValues.FAILED
