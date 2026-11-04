from __future__ import annotations

import pytest

from cramph.composites import Parallel
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.node import EndStatechart, StatechartNode
from cramph.nodes_for_testing import ConstTrueNode, NodeSucceedingOnObservingTrue
from cramph.plotters.interactive_graph import StatechartGraphVisualizer
from cramph.statechart import Statechart
from krrood.rustworkx_utils.graph_visualizer_base import (
    GraphLayout,
    GraphVisualizerBackend,
)

# %% the statechart that is inspected


@pytest.fixture()
def two_tree_statechart(statechart_executor: StatechartExecutor) -> Statechart:
    """
    :return: A statechart holding two trees, the first nesting a composite::

        root                  other
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
    statechart.add_node(ConstTrueNode(name="other"))
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


# %% layers


def test_layers_hold_the_nodes_at_each_depth_in_order(
    two_tree_statechart: Statechart,
) -> None:
    names = [[node.name for node in layer] for layer in two_tree_statechart.layers]

    assert names == [
        ["root", "other"],
        ["first", "middle", "last"],
        ["inner_first", "inner_last"],
    ]


def test_an_empty_statechart_has_no_layers(
    statechart_executor: StatechartExecutor,
) -> None:
    assert Statechart(context=statechart_executor.context).layers == []


# %% how a statechart is drawn


def test_the_drawing_connects_each_node_to_the_nodes_it_runs(
    two_tree_statechart: Statechart,
) -> None:
    visualizer = StatechartGraphVisualizer(two_tree_statechart).create_visualizer(
        backend=GraphVisualizerBackend.CYTOSCAPE, layout=GraphLayout.LAYERED
    )

    expected_edges = {
        (node.index, child.index)
        for node in two_tree_statechart.nodes
        for child in node.children
    }
    assert set(visualizer.graph.edge_list()) == expected_edges


def test_a_node_is_drawn_labelled_by_its_unique_name(
    two_tree_statechart: Statechart,
) -> None:
    middle = node_named(two_tree_statechart, "middle")
    visualizer = StatechartGraphVisualizer(two_tree_statechart).create_visualizer(
        backend=GraphVisualizerBackend.CYTOSCAPE, layout=GraphLayout.LAYERED
    )

    assert visualizer.node_label(middle.index) == middle.unique_name


def test_a_node_is_drawn_in_the_color_of_its_current_state(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    node = NodeSucceedingOnObservingTrue(
        name="node", observation=ObservationStateValues.TRUE
    )
    statechart.add_node(node)
    statechart.add_node(EndStatechart.when_true(node))
    visualizer = StatechartGraphVisualizer(statechart).create_visualizer(
        backend=GraphVisualizerBackend.CYTOSCAPE, layout=GraphLayout.LAYERED
    )

    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end(timeout=100)

    assert node.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert visualizer.node_color(node.index) == LifeCycleValues.SUCCEEDED.color.to_hex()
