from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from typing_extensions import Callable

from cramph.composites import (
    ControlStructureNode,
    Parallel,
    PausedUntilTrue,
    RepeatUntil,
    Sequence,
    TryAll,
    TryInOrder,
)
from cramph.data_types import ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.node import CompositeNode, EndedByOwner, StatechartNode
from cramph.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    NodeObservingAFixedValue,
    NodeSucceedingOnObservingTrue,
)
from cramph.statechart import Statechart

# %% the tree the navigation is read from


@pytest.fixture()
def nested_statechart(statechart_executor: StatechartExecutor) -> Statechart:
    """
    :return: A statechart holding one tree, whose middle child is itself a composite::

        root
        |-- first
        |-- middle
        |   |-- inner_first
        |   +-- inner_last
        +-- last

    Its nodes are reached by name through :func:`node_named`.
    """
    statechart = Statechart(context=statechart_executor.context)
    middle = Parallel(
        name="middle",
        nodes=[ConstTrueNode(name="inner_first"), ConstTrueNode(name="inner_last")],
    )
    statechart.add_node(
        Parallel(
            name="root",
            nodes=[ConstTrueNode(name="first"), middle, ConstTrueNode(name="last")],
        )
    )
    return statechart


def node_named(statechart: Statechart, name: str) -> StatechartNode:
    """
    :param statechart: The statechart to search.
    :param name: The name of the node to find.
    :return: The one node of `statechart` carrying that name.
    """
    matches = [node for node in statechart.nodes if node.name == name]
    assert len(matches) == 1, f"{name} names {len(matches)} nodes"
    return matches[0]


def names_of(nodes: list[StatechartNode]) -> list[str]:
    """
    :param nodes: The nodes to name.
    :return: Their names, in the order they were given.
    """
    return [node.name for node in nodes]


# %% children


def test_a_node_that_runs_nothing_has_no_children(
    nested_statechart: Statechart,
) -> None:
    assert node_named(nested_statechart, "first").children == []


def test_a_composite_answers_the_nodes_it_runs_in_order(
    nested_statechart: Statechart,
) -> None:
    root = node_named(nested_statechart, "root")

    assert root.children == root.nodes


def test_children_are_exactly_the_nodes_whose_parent_this_is(
    nested_statechart: Statechart,
) -> None:
    for node in nested_statechart.nodes:
        assert node.children == [
            candidate
            for candidate in nested_statechart.nodes
            if candidate.parent_node is node
        ]


# %% descendants


def test_descendants_follow_each_child_with_its_own_subtree(
    nested_statechart: Statechart,
) -> None:
    root = node_named(nested_statechart, "root")

    assert names_of(root.descendants) == [
        "first",
        "middle",
        "inner_first",
        "inner_last",
        "last",
    ]


def test_a_node_that_runs_nothing_has_no_descendants(
    nested_statechart: Statechart,
) -> None:
    assert node_named(nested_statechart, "last").descendants == []


# %% path


def test_the_path_leads_from_the_parent_up_to_the_root(
    nested_statechart: Statechart,
) -> None:
    inner_first = node_named(nested_statechart, "inner_first")

    assert inner_first.path == [
        node_named(nested_statechart, "middle"),
        node_named(nested_statechart, "root"),
    ]


def test_a_top_level_node_has_an_empty_path(nested_statechart: Statechart) -> None:
    assert node_named(nested_statechart, "root").path == []


def test_the_depth_of_a_node_is_the_length_of_its_path(
    nested_statechart: Statechart,
) -> None:
    for node in nested_statechart.nodes:
        assert node.depth == len(node.path)


# %% is_leaf


def test_a_node_that_runs_nothing_is_a_leaf(nested_statechart: Statechart) -> None:
    assert node_named(nested_statechart, "inner_last").is_leaf


def test_a_composite_is_not_a_leaf(nested_statechart: Statechart) -> None:
    assert not node_named(nested_statechart, "middle").is_leaf


# %% siblings


def test_the_siblings_of_a_child_are_the_other_children_in_order(
    nested_statechart: Statechart,
) -> None:
    assert names_of(node_named(nested_statechart, "middle").siblings) == [
        "first",
        "last",
    ]


def test_the_siblings_of_a_top_level_node_are_the_other_top_level_nodes(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([first := ConstTrueNode(), other := ConstTrueNode()])

    assert first.siblings == [other]


def test_the_only_top_level_node_has_no_siblings(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(only := ConstTrueNode())

    assert only.siblings == []


def test_siblings_of_the_same_name_are_told_apart_by_identity(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(
        Parallel(
            name="root",
            nodes=[
                first := ConstTrueNode(name="same"),
                second := ConstTrueNode(name="same"),
            ],
        )
    )

    assert first.siblings == [second]
    assert second.siblings == [first]


# %% neighbours


def test_the_siblings_split_into_those_left_and_right_of_this_node(
    nested_statechart: Statechart,
) -> None:
    middle = node_named(nested_statechart, "middle")

    assert names_of(middle.left_siblings) == ["first"]
    assert names_of(middle.right_siblings) == ["last"]


def test_the_neighbours_are_the_closest_sibling_on_either_side(
    nested_statechart: Statechart,
) -> None:
    middle = node_named(nested_statechart, "middle")

    assert middle.left_neighbour is node_named(nested_statechart, "first")
    assert middle.right_neighbour is node_named(nested_statechart, "last")


def test_the_leftmost_node_has_no_left_neighbour(nested_statechart: Statechart) -> None:
    first = node_named(nested_statechart, "first")

    assert first.left_siblings == []
    assert first.left_neighbour is None


def test_the_rightmost_node_has_no_right_neighbour(
    nested_statechart: Statechart,
) -> None:
    last = node_named(nested_statechart, "last")

    assert last.right_siblings == []
    assert last.right_neighbour is None


@dataclass(kw_only=True)
class ComparedByItsSpeed:
    """
    Gives a node a parameter and, with it, an equality over that parameter alone.
    """

    speed: float = 1.0
    """
    The parameter two nodes of the same class are compared by.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class NodeComparedByItsSpeed(ConstTrueNode, ComparedByItsSpeed):
    """
    A node that compares equal to every node of its class running at the same speed.
    """


def test_siblings_that_compare_equal_are_told_apart(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(
        Parallel(
            nodes=[
                NodeComparedByItsSpeed(name="first"),
                NodeComparedByItsSpeed(name="second"),
                NodeComparedByItsSpeed(name="third"),
            ]
        )
    )
    second = node_named(statechart, "second")

    assert names_of(second.left_siblings) == ["first"]
    assert names_of(second.right_siblings) == ["third"]


# %% the closest node of a type that runs before a node


def test_the_closest_preceding_node_is_the_last_in_a_left_siblings_subtree(
    nested_statechart: Statechart,
) -> None:
    last = node_named(nested_statechart, "last")

    assert nested_statechart.get_preceding_node_by_type(
        last, ConstTrueNode
    ) is node_named(nested_statechart, "inner_last")


def test_the_preceding_nodes_are_listed_closest_first(
    nested_statechart: Statechart,
) -> None:
    last = node_named(nested_statechart, "last")

    assert names_of(
        nested_statechart.get_preceding_nodes_by_type(last, ConstTrueNode)
    ) == ["inner_last", "inner_first", "first"]


def test_a_preceding_node_is_found_left_of_an_ancestor(
    nested_statechart: Statechart,
) -> None:
    inner_first = node_named(nested_statechart, "inner_first")

    assert nested_statechart.get_preceding_node_by_type(
        inner_first, ConstTrueNode
    ) is node_named(nested_statechart, "first")


def test_a_left_sibling_itself_precedes_a_node(nested_statechart: Statechart) -> None:
    last = node_named(nested_statechart, "last")

    assert nested_statechart.get_preceding_node_by_type(last, Parallel) is node_named(
        nested_statechart, "middle"
    )


def test_an_ancestor_does_not_precede_its_descendants(
    nested_statechart: Statechart,
) -> None:
    inner_last = node_named(nested_statechart, "inner_last")

    assert nested_statechart.get_preceding_node_by_type(inner_last, Parallel) is None


# %% the children of a goal


def test_the_children_of_a_sequence_are_the_steps_it_was_handed(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    steps = [ConstTrueNode(name="step"), ConstTrueNode(name="next")]
    statechart.add_node(sequence := Sequence(nodes=list(steps)))

    assert sequence.children == steps


# %% navigating while the tree is still expanding


@dataclass(eq=False, repr=False, kw_only=True)
class CompositeNodeNavigatingWhileItExpands(EndedByOwner, CompositeNode):
    """
    A node that reads the tree around it while it expands, which is the moment the control
    structure node above it is still adopting its children.
    """

    preceding_fixed_value_node: NodeObservingAFixedValue | None = field(
        default=None, init=False
    )
    """
    The closest :class:`NodeObservingAFixedValue` before this node, as found while
    expanding.
    """

    children_of_language_node: list[StatechartNode] = field(
        default_factory=list, init=False
    )
    """
    The children of the closest control structure node above this node, as read while
    expanding.
    """

    def expand(self) -> None:
        self.preceding_fixed_value_node = self.statechart.get_preceding_node_by_type(
            self, NodeObservingAFixedValue
        )
        self.children_of_language_node = ControlStructureNode.closest_above(
            self
        ).children


def _self_deciding_step() -> NodeObservingAFixedValue:
    """
    :return: A node a control structure node takes as its child as it is, without an
        attempt around it.
    """
    return NodeSucceedingOnObservingTrue(observation=ObservationStateValues.TRUE)


def _step_decided_by_its_owner() -> NodeObservingAFixedValue:
    """
    :return: A node a control structure node ends once it observes True.
    """
    return NodeObservingAFixedValue(observation=ObservationStateValues.TRUE)


def _paused_until_true(monitored_node: StatechartNode) -> PausedUntilTrue:
    """
    :return: A composite running `monitored_node` one layer below itself.
    """
    return PausedUntilTrue(monitor=ConstFalseNode(), monitored_node=monitored_node)


@pytest.mark.parametrize(
    "preceding_step_factory", [_self_deciding_step, _step_decided_by_its_owner]
)
@pytest.mark.parametrize("language_node_type", [Sequence, TryInOrder, TryAll, Parallel])
def test_a_step_finds_the_step_before_it_while_it_expands(
    statechart_executor: StatechartExecutor,
    language_node_type: type[ControlStructureNode],
    preceding_step_factory: Callable[[], NodeObservingAFixedValue],
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    preceding_step = preceding_step_factory()
    navigating = CompositeNodeNavigatingWhileItExpands()

    statechart.add_node(language_node_type(nodes=[preceding_step, navigating]))

    assert navigating.preceding_fixed_value_node is preceding_step


@pytest.mark.parametrize(
    "preceding_step_factory", [_self_deciding_step, _step_decided_by_its_owner]
)
@pytest.mark.parametrize("language_node_type", [Sequence, TryInOrder, TryAll, Parallel])
def test_a_node_below_a_step_finds_the_step_before_that_while_it_expands(
    statechart_executor: StatechartExecutor,
    language_node_type: type[ControlStructureNode],
    preceding_step_factory: Callable[[], NodeObservingAFixedValue],
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    preceding_step = preceding_step_factory()
    navigating = CompositeNodeNavigatingWhileItExpands()

    statechart.add_node(
        language_node_type(nodes=[preceding_step, _paused_until_true(navigating)])
    )

    assert navigating.preceding_fixed_value_node is preceding_step


def test_an_expanding_step_sees_the_children_before_it_and_the_nodes_still_to_join(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    navigating = CompositeNodeNavigatingWhileItExpands()
    following_step = _step_decided_by_its_owner()
    sequence = Sequence(
        nodes=[_step_decided_by_its_owner(), navigating, following_step]
    )

    statechart.add_node(sequence)

    assert navigating.children_of_language_node == [
        *sequence.children[:2],
        following_step,
    ]


def test_an_inserted_step_finds_the_step_before_it_while_it_expands(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    preceding_step = _step_decided_by_its_owner()
    statechart.add_node(sequence := Sequence(nodes=[preceding_step]))
    navigating = CompositeNodeNavigatingWhileItExpands()

    sequence.insert_after(preceding_step, navigating)

    assert navigating.preceding_fixed_value_node is preceding_step


def test_a_replacing_step_finds_the_step_before_it_while_it_expands(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    preceding_step, replaced = _step_decided_by_its_owner(), ConstTrueNode()
    statechart.add_node(Sequence(nodes=[preceding_step, replaced]))
    navigating = CompositeNodeNavigatingWhileItExpands()

    statechart.replace(replaced, navigating)

    assert navigating.preceding_fixed_value_node is preceding_step


def test_a_repeated_task_finds_the_step_before_its_repetition_while_it_expands(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    preceding_step = _step_decided_by_its_owner()
    navigating = CompositeNodeNavigatingWhileItExpands()
    repetition = RepeatUntil(task=navigating, stop_retry_monitor=ConstFalseNode())

    statechart.add_node(Sequence(nodes=[preceding_step, repetition]))

    assert navigating.preceding_fixed_value_node is preceding_step
