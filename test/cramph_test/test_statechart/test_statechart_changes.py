"""
Tests that a statechart records the changes made to its structure in blocks, and that
another statechart applying those blocks gets the same structure without expanding any
node again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from cramph.composites import (
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
    Sequence,
)
from cramph.context import StatechartContext
from cramph.data_types import ObservationStateValue, TransitionKind
from cramph.executor import StatechartExecutor
from cramph.node import CompositeNode, EndStatechart, StatechartNode
from cramph.nodes_for_testing import (
    CompositeNodeWithNestedCompositeChild,
    ConstTrueNode,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart
from cramph.statechart_changes import ChildrenChange, ConditionChange
from semantic_digital_twin.world import World

from .test_composite_node_choosing_its_child import ChooserAnsweringInTurn

# %% mimics


@dataclass(kw_only=True, eq=True, repr=False)
class NodeComparingItsFields(ConstTrueNode):
    """
    A node that equals every node of its class with the same fields, so it cannot be
    hashed and two such nodes are told apart only by being different objects.
    """


# %% helpers


@dataclass(kw_only=True)
class NodeStructure:
    """
    What a node is in the structure of its statechart, named by node ids so that two
    copies of a statechart can be compared.
    """

    node_id: str
    """
    The id of the node.
    """

    name: str
    """
    The name of the node.
    """

    parent_id: str | None
    """
    The id of the parent of the node, None on the top level.
    """

    children_ids: list[str]
    """
    The ids of the children of the node, in order.
    """

    conditions: list[str]
    """
    Every transition condition of the node, rendered.
    """

    @classmethod
    def from_node(cls, *, node: StatechartNode) -> NodeStructure:
        """
        :param node: A node of a statechart.
        :return: Its place in the structure of that statechart.
        """
        parent = node.parent_node
        return cls(
            node_id=node._node_id,
            name=node.name,
            parent_id=None if parent is None else parent._node_id,
            children_ids=[child._node_id for child in node.children],
            conditions=[str(condition) for condition in node.conditions],
        )


def structure(*, statechart: Statechart) -> list[NodeStructure]:
    """
    :return: The structure of every node of `statechart`, in index order.
    """
    return [NodeStructure.from_node(node=node) for node in statechart.nodes]


def replay(*, blocks: list, receiver: Statechart) -> Statechart:
    """
    Applies `blocks` to `receiver`, each after a trip through a JSON string.

    :return:`receiver`.
    """
    for block in blocks:
        receiver.apply_change_block_from_json(
            data=json.loads(json.dumps(block.to_json()))
        )
    return receiver


def empty_statechart() -> Statechart:
    """
    :return: A statechart without nodes, in a context of its own.
    """
    return Statechart(context=StatechartContext(world=World()))


def node_arriving_at_once(*, name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds on the tick after it starts.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValue.TRUE
    )


@pytest.fixture
def sender() -> Statechart:
    """
    A statechart with a sequence of two steps and a composite node holding a composite
    node that expands into two chained children.
    """
    statechart = empty_statechart()
    first = node_arriving_at_once(name="first")
    second = node_arriving_at_once(name="second")
    statechart.add_node(node=Sequence(name="sequence", nodes=[first, second]))
    statechart.add_node(node=CompositeNodeWithNestedCompositeChild(name="outer"))
    return statechart


# %% replaying every block


def test_replaying_every_block_rebuilds_the_structure(sender):
    """
    The receiver holds the same nodes in the same order, with the same parents, children
    and conditions, so no composite node expanded a second time.
    """
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )

    assert structure(statechart=receiver) == structure(statechart=sender)


def test_the_changes_of_one_modification_form_one_block(sender):
    """
    Nested modifications close their changes into a block only once the outermost one
    ends.
    """
    block_count = len(sender.structure_change_recorder.change_blocks)

    with sender.modify():
        with sender.modify():
            sender.add_node(node=ConstTrueNode(name="inner"))
        sender.add_node(node=ConstTrueNode(name="outer"))

    assert len(sender.structure_change_recorder.change_blocks) == block_count + 1


def test_a_failed_modification_records_no_change(sender):
    """
    A modification that raises leaves the statechart as it was, so there is nothing to
    repeat.
    """
    block_count = len(sender.structure_change_recorder.change_blocks)

    with pytest.raises(ZeroDivisionError):
        with sender.modify():
            sender.add_node(node=ConstTrueNode(name="dropped"))
            1 / 0

    assert len(sender.structure_change_recorder.change_blocks) == block_count


def test_a_removal_in_a_failed_modification_is_undone_in_both_statecharts(sender):
    """
    The removed node is back in the sender, so the receiver, which never learnt of the
    removal, holds the same nodes.
    """
    extra = ConstTrueNode(name="extra")
    sender.add_node(node=extra)
    structure_before = structure(statechart=sender)

    with pytest.raises(ZeroDivisionError):
        with sender.modify():
            sender.remove_node(node=extra)
            1 / 0
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )

    assert structure(statechart=sender) == structure_before
    assert structure(statechart=receiver) == structure_before


def test_a_failed_modification_puts_back_a_removed_node_before_dropping_an_added_one(
    sender,
):
    """
    The node removed with everything below it gets its place back, and the node added
    after the removal leaves again.
    """
    outer = sender.get_nodes_by_type(node_type=CompositeNode)[-1]
    added = ConstTrueNode(name="added")
    structure_before = structure(statechart=sender)

    with pytest.raises(ZeroDivisionError):
        with sender.modify():
            sender.remove_node(node=outer)
            sender.add_node(node=added)
            1 / 0

    assert structure(statechart=sender) == structure_before
    assert not added.belongs_to_statechart()


def test_a_failed_modification_keeps_the_reset_requested_for_a_node_it_removed(sender):
    """
    The reset requested before the modification still holds for the node put back.
    """
    extra = ConstTrueNode(name="extra")
    sender.add_node(node=extra)
    sender.request_reset(node=extra)

    with pytest.raises(ZeroDivisionError):
        with sender.modify():
            sender.remove_node(node=extra)
            1 / 0

    assert sender.take_requested_resets() == [extra]


def test_a_rebuild_requested_in_a_failed_modification_is_forgotten_in_both_statecharts(
    sender,
):
    """
    Neither statechart builds the node again on its next compile.
    """
    sequence = sender.nodes[0]
    rebuilds_before = list(sender.compiler.nodes_to_rebuild)

    with pytest.raises(ZeroDivisionError):
        with sender.modify():
            sender.request_rebuild(node=sequence)
            1 / 0
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )

    assert sender.compiler.nodes_to_rebuild == rebuilds_before
    assert [node._node_id for node in receiver.compiler.nodes_to_rebuild] == [
        node._node_id for node in rebuilds_before
    ]


def test_a_condition_set_outside_a_modification_closes_a_block_of_its_own(sender):
    """
    Outside a modification there is no outermost block to wait for.
    """
    first, second = sender.nodes[1:3]

    second.pause_condition = first.is_running

    [change] = sender.structure_change_recorder.change_blocks[-1].changes
    assert isinstance(change, ConditionChange)
    assert (change.owner, change.kind) == (second, TransitionKind.PAUSE)


def test_equal_nodes_are_recorded_one_by_one(sender):
    """
    Nodes that compare equal are still different nodes, each with its own conditions.
    """
    first = NodeComparingItsFields(name="twin")
    second = NodeComparingItsFields(name="twin")

    sender.add_nodes(nodes=[first, second])
    second.start_condition = first.is_succeeded
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )

    assert structure(statechart=receiver) == structure(statechart=sender)


# %% changes to nodes the receiver holds already


def test_a_receiver_reads_the_rewired_condition_of_the_step_after_an_insertion(sender):
    """
    Inserting a step into a sequence makes the step after it wait for the new step, so
    the receiver has to rewire that step too.
    """
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )
    block_count = len(sender.structure_change_recorder.change_blocks)
    sequence, first, second = sender.nodes[:3]

    sequence.insert_before(reference=second, node=node_arriving_at_once(name="middle"))
    replay(
        blocks=sender.structure_change_recorder.change_blocks[block_count:],
        receiver=receiver,
    )

    assert structure(statechart=receiver) == structure(statechart=sender)


def test_a_removed_node_leaves_the_receiver(sender):
    """
    The removed node takes everything below it along, in both statecharts.
    """
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )
    block_count = len(sender.structure_change_recorder.change_blocks)

    sender.remove_node(node=sender.get_nodes_by_type(node_type=CompositeNode)[-1])
    replay(
        blocks=sender.structure_change_recorder.change_blocks[block_count:],
        receiver=receiver,
    )

    assert structure(statechart=receiver) == structure(statechart=sender)


def test_a_condition_reading_a_node_that_did_not_join_yet_waits_for_it(sender):
    """
    A condition naming a node the receiver could not resolve yet is written once that
    node joined.
    """
    first = sender.nodes[1]
    late = ConstTrueNode(name="late")

    first.pause_condition = late.is_running
    sender.add_node(node=late)
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )

    assert structure(statechart=receiver) == structure(statechart=sender)


# %% changing the children of a composite node


def test_removing_and_inserting_a_child_records_the_children_as_they_are_then(sender):
    """
    Each change to the children of a node of the statechart closes a block holding the
    children the node has afterwards.
    """
    sequence, first, second = sender.nodes[:3]

    sequence.remove_child(child=second)
    [removal] = sender.structure_change_recorder.change_blocks[-1].changes
    sequence.insert_child(position=0, node=second)
    [insertion] = sender.structure_change_recorder.change_blocks[-1].changes

    assert isinstance(removal, ChildrenChange)
    assert (removal.node, removal.children) == (sequence, [first])
    assert isinstance(insertion, ChildrenChange)
    assert (insertion.node, insertion.children) == (sequence, [second, first])


def test_removing_a_child_keeps_a_child_comparing_equal_to_it():
    """
    Only the child itself stops being listed, not another child equal to it.
    """
    kept = NodeComparingItsFields(name="twin")
    removed = NodeComparingItsFields(name="twin")
    parent = Sequence(name="parent", nodes=[kept, removed])

    parent.remove_child(child=removed)

    assert len(parent.nodes) == 1
    assert parent.nodes[0] is kept


# %% requests


def test_a_requested_reset_reaches_the_receiver(sender):
    """
    The receiver resets the same node on its next settled tick.
    """
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )
    block_count = len(sender.structure_change_recorder.change_blocks)
    sequence = sender.nodes[0]

    sender.request_reset(node=sequence)
    replay(
        blocks=sender.structure_change_recorder.change_blocks[block_count:],
        receiver=receiver,
    )

    assert [node._node_id for node in receiver.take_requested_resets()] == [
        sequence._node_id
    ]


def test_a_requested_rebuild_reaches_the_receiver(sender):
    """
    The receiver builds the same node again on its next compile.
    """
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )
    block_count = len(sender.structure_change_recorder.change_blocks)
    sequence = sender.nodes[0]

    sender.request_rebuild(node=sequence)
    replay(
        blocks=sender.structure_change_recorder.change_blocks[block_count:],
        receiver=receiver,
    )

    assert [node._node_id for node in receiver.compiler.nodes_to_rebuild] == [
        sequence._node_id
    ]


def test_a_reset_requested_before_removing_the_node_in_one_block_reaches_the_receiver(
    sender,
):
    """
    The receiver resets the node before it removes it, as the sender did.
    """
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )
    block_count = len(sender.structure_change_recorder.change_blocks)
    extra = ConstTrueNode(name="extra")
    sender.add_node(node=extra)

    with sender.modify():
        sender.request_reset(node=extra)
        sender.remove_node(node=extra)
    replay(
        blocks=sender.structure_change_recorder.change_blocks[block_count:],
        receiver=receiver,
    )

    assert structure(statechart=receiver) == structure(statechart=sender)


def test_a_rebuild_requested_before_removing_the_node_in_one_block_reaches_the_receiver(
    sender,
):
    """
    The receiver forgets the rebuild of the removed node, as the sender did.
    """
    receiver = replay(
        blocks=sender.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )
    block_count = len(sender.structure_change_recorder.change_blocks)
    extra = ConstTrueNode(name="extra")
    sender.add_node(node=extra)

    with sender.modify():
        sender.request_rebuild(node=extra)
        sender.remove_node(node=extra)
    replay(
        blocks=sender.structure_change_recorder.change_blocks[block_count:],
        receiver=receiver,
    )

    assert [node._node_id for node in receiver.compiler.nodes_to_rebuild] == [
        node._node_id for node in sender.compiler.nodes_to_rebuild
    ]


# %% chosen children


def test_the_chosen_children_reach_the_receiver(
    statechart_executor: StatechartExecutor,
):
    """
    A node choosing its child chose it on the sender only, and the receiver learns which
    child it chose.
    """
    child = node_arriving_at_once(name="child")
    statechart_executor.context.add_extension(
        extension=ChildChooserAccess(
            choosers=[ChooserAnsweringInTurn(children=[child])]
        )
    )
    statechart = Statechart(context=statechart_executor.context)
    choosing_node = CompositeNodeChoosingItsChild(name="choosing")
    statechart.add_node(node=choosing_node)
    statechart.add_node(node=EndStatechart.when_true(node=choosing_node))
    statechart_executor.compile(statechart=statechart)
    statechart_executor.tick_until_end(timeout=30)

    receiver = replay(
        blocks=statechart.structure_change_recorder.change_blocks,
        receiver=empty_statechart(),
    )

    received_choosing_node = receiver.get_node_by_index(index=choosing_node.index)
    assert [node._node_id for node in received_choosing_node.chosen_children] == [
        child._node_id
    ]
