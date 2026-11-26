"""
Tests for looking a node of a statechart up by its id, see
:meth:`~cramph.statechart.Statechart.get_node_by_id`.
"""

from __future__ import annotations

import json

import pytest

from cramph.composites import Sequence
from cramph.exceptions import NodeWithIdNotFoundError
from cramph.executor import StatechartExecutor
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import Statechart

# %% nodes that belong to the statechart


def test_an_added_node_is_found_by_its_id(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes(nodes=[ConstTrueNode(), added := ConstTrueNode()])

    assert statechart.get_node_by_id(node_id=added._node_id) is added


def test_a_child_joining_with_its_composite_is_found_by_its_id(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=Sequence(nodes=[child := ConstTrueNode()]))

    assert statechart.get_node_by_id(node_id=child._node_id) is child


def test_the_nodes_kept_by_a_removal_are_found_by_their_id(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes(nodes=[first := ConstTrueNode(), removed := ConstTrueNode()])
    statechart.add_node(node=(last := ConstTrueNode()))

    statechart.remove_node(node=removed)

    assert [
        statechart.get_node_by_id(node_id=kept._node_id) for kept in [first, last]
    ] == [first, last]


def test_the_copy_of_a_node_is_found_by_the_id_of_the_node_it_copies(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=(original := ConstTrueNode()))

    statechart_copy = statechart.create_structure_copy()

    assert statechart_copy.get_node_by_id(
        node_id=original._node_id
    ) is statechart_copy.get_node_by_index(index=original.index)


def test_a_deserialized_node_is_found_by_the_id_it_was_serialized_with(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=(original := ConstTrueNode()))

    received = Statechart.from_json(
        data=json.loads(json.dumps(statechart.create_structure_copy().to_json())),
        context=statechart_executor.context,
    )

    assert received.get_node_by_id(
        node_id=original._node_id
    ) is received.get_node_by_index(index=original.index)


# %% nodes that do not belong to the statechart


def test_a_removed_node_is_not_found_by_its_id(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes(nodes=[ConstTrueNode(), removed := ConstTrueNode()])

    statechart.remove_node(node=removed)

    with pytest.raises(NodeWithIdNotFoundError):
        statechart.get_node_by_id(node_id=removed._node_id)


def test_a_descendant_removed_with_its_composite_is_not_found_by_its_id(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=(removed := Sequence(nodes=[child := ConstTrueNode()])))

    statechart.remove_node(node=removed)

    with pytest.raises(NodeWithIdNotFoundError):
        statechart.get_node_by_id(node_id=child._node_id)


def test_a_node_of_another_statechart_is_not_found_by_its_id(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(node=ConstTrueNode())
    other = Statechart(context=statechart_executor.context)
    other.add_node(node=(elsewhere := ConstTrueNode()))

    with pytest.raises(NodeWithIdNotFoundError):
        statechart.get_node_by_id(node_id=elsewhere._node_id)
