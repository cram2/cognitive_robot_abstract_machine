"""
Tests for causal graphs: their edges, the directed acyclic graphs they stand for, their
nodes for relational models, and the cells that rank an adjustment set's support.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
import pytest
from krrood.adapters.json_serializer import from_json, to_json
from random_events.variable import Continuous

from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline

from probabilistic_model.causal_structure.discretization import Discretizer
from probabilistic_model.causal_structure.exceptions import (
    EquivalenceClassTooLarge,
    InconsistentGraph,
    NotADirectedEdge,
    NotAnEndOfTheEdge,
    UnknownGraphNode,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    CausalGraphJSONKey,
    ColliderPatternInCausalGraph,
    EdgeMark,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.learning import AdjacencyMatrixMark

from .graphs import (
    collider_with_an_undirected_child,
    confounded,
    mediated,
    undirected_chain,
    undirected_parent,
    without_member_graphs,
)

# %% values that are neither numbers nor text


class Shape(Enum):
    """
    Values a column can hold that cannot be ordered.
    """

    ROUND = 1
    """
    A round shape.
    """

    SQUARE = 2
    """
    A square shape.
    """


# %% the graphs an equivalence class stands for


def test_member_graphs_keep_v_structures():
    [member] = collider_with_an_undirected_child().member_graphs()

    assert {(parent.name, child.name) for parent, child in member.edges()} == {
        ("A", "C"),
        ("B", "C"),
        ("C", "D"),
    }


def test_unshielded_colliders_name_both_parents_in_name_order():
    graph = collider_with_an_undirected_child()
    assert graph.unshielded_colliders() == {
        ColliderPatternInCausalGraph(graph.node("A"), graph.node("C"), graph.node("B"))
    }


def test_a_long_undirected_chain_has_one_member_graph_per_source():
    graph = undirected_chain(20)

    assert sorted(
        node.name
        for member in graph.member_graphs()
        for node in member.nodes()
        if member.in_degree(node) == 0
    ) == sorted(graph.node_names)


def test_more_member_graphs_than_listed_raises():
    graph = undirected_chain(3)
    graph.maximum_member_graphs = len(graph.nodes()) - 1

    with pytest.raises(EquivalenceClassTooLarge):
        graph.member_graphs()


def test_graph_without_member_graphs_raises():
    with pytest.raises(InconsistentGraph):
        without_member_graphs().member_graphs()


# %% building a graph


def test_an_edge_to_a_node_the_graph_lacks_raises():
    graph = confounded()

    outside = GraphNode(variable=Continuous("W"), level=NodeLevel.EXAMPLE_ATTRIBUTE)

    with pytest.raises(UnknownGraphNode):
        graph.add_edge(GraphEdge.directed_edge(graph.node("X"), outside))


def test_the_edge_between_two_nodes_is_found_from_either_end():
    graph = confounded()
    cause, confounder = graph.node("X"), graph.node("Z")

    assert (
        graph.edge_between(cause, confounder)
        == graph.edge_between(confounder, cause)
        == GraphEdge.directed_edge(confounder, cause)
    )


def test_nodes_without_an_edge_are_not_adjacent():
    """
    Nodes without an edge are not adjacent.
    """
    graph = mediated()

    assert not graph.adjacent(first=graph.node("X"), second=graph.node("Y"))


def test_a_node_an_undirected_edge_may_point_into_is_a_possible_ancestor():
    """
    A node an undirected edge may point into is a possible ancestor.
    """
    graph = undirected_parent()

    assert graph.possible_ancestors(node=graph.node("X")) == {graph.node("A")}


# %% the kinds of edges


@dataclass(frozen=True, kw_only=True)
class EdgeKinds:
    """
    Which kinds an edge with two given marks belongs to.
    """

    is_directed: bool = False
    """
    A tail at one end and an arrow at the other.
    """

    is_undirected: bool = False
    """
    A tail at both ends.
    """

    is_bidirected: bool = False
    """
    An arrow at both ends.
    """

    is_partially_directed: bool = False
    """
    A circle at one end and an arrow at the other.
    """

    is_nondirected: bool = False
    """
    A circle at both ends.
    """

    implies_selection_bias: bool = False
    """
    A tail at one end and no arrow at the other.
    """


@pytest.mark.parametrize(
    "mark_at_first, mark_at_second, kinds",
    [
        (
            EdgeMark.TAIL,
            EdgeMark.TAIL,
            EdgeKinds(is_undirected=True, implies_selection_bias=True),
        ),
        (EdgeMark.TAIL, EdgeMark.ARROW, EdgeKinds(is_directed=True)),
        (EdgeMark.TAIL, EdgeMark.CIRCLE, EdgeKinds(implies_selection_bias=True)),
        (EdgeMark.ARROW, EdgeMark.TAIL, EdgeKinds(is_directed=True)),
        (EdgeMark.ARROW, EdgeMark.ARROW, EdgeKinds(is_bidirected=True)),
        (EdgeMark.ARROW, EdgeMark.CIRCLE, EdgeKinds(is_partially_directed=True)),
        (EdgeMark.CIRCLE, EdgeMark.TAIL, EdgeKinds(implies_selection_bias=True)),
        (EdgeMark.CIRCLE, EdgeMark.ARROW, EdgeKinds(is_partially_directed=True)),
        (EdgeMark.CIRCLE, EdgeMark.CIRCLE, EdgeKinds(is_nondirected=True)),
    ],
)
def test_an_edge_belongs_to_the_kinds_its_marks_give(
    mark_at_first: EdgeMark, mark_at_second: EdgeMark, kinds: EdgeKinds
):
    """
    Each pair of marks gives exactly the edge kinds the marks define.
    """
    graph = confounded()
    edge = GraphEdge(graph.node("X"), graph.node("Y"), mark_at_first, mark_at_second)

    assert (
        EdgeKinds(
            is_directed=edge.is_directed,
            is_undirected=edge.is_undirected,
            is_bidirected=edge.is_bidirected,
            is_partially_directed=edge.is_partially_directed,
            is_nondirected=edge.is_nondirected,
            implies_selection_bias=edge.implies_selection_bias,
        )
        == kinds
    )


def test_the_edge_constructors_put_their_marks_at_the_named_ends():
    """
    The constructors of ↔, o→ and o–o put each mark at the end they name.
    """
    graph = confounded()
    first, second = graph.node("X"), graph.node("Y")

    assert [
        GraphEdge.bidirected_edge(first=first, second=second),
        GraphEdge.partially_directed_edge(circle_end=first, arrow_end=second),
        GraphEdge.nondirected_edge(first=first, second=second),
    ] == [
        GraphEdge(first, second, EdgeMark.ARROW, EdgeMark.ARROW),
        GraphEdge(first, second, EdgeMark.CIRCLE, EdgeMark.ARROW),
        GraphEdge(first, second, EdgeMark.CIRCLE, EdgeMark.CIRCLE),
    ]


@pytest.mark.parametrize(
    "mark_at_first, mark_at_second",
    [
        (EdgeMark.ARROW, EdgeMark.ARROW),
        (EdgeMark.CIRCLE, EdgeMark.ARROW),
        (EdgeMark.CIRCLE, EdgeMark.CIRCLE),
        (EdgeMark.TAIL, EdgeMark.TAIL),
    ],
)
def test_an_edge_that_is_not_directed_has_no_parent_or_child(
    mark_at_first: EdgeMark, mark_at_second: EdgeMark
):
    """
    Only a directed edge has a parent and a child.
    """
    graph = confounded()
    edge = GraphEdge(graph.node("X"), graph.node("Y"), mark_at_first, mark_at_second)

    with pytest.raises(NotADirectedEdge):
        edge.parent
    with pytest.raises(NotADirectedEdge):
        edge.child


def test_a_directed_edge_written_backwards_names_its_parent_and_child():
    """
    A directed edge whose arrow is at its first end still names its tail as parent.
    """
    graph = confounded()
    edge = GraphEdge(graph.node("Y"), graph.node("X"), EdgeMark.ARROW, EdgeMark.TAIL)

    assert (edge.parent, edge.child) == (graph.node("X"), graph.node("Y"))


def test_an_edge_reads_its_mark_and_its_other_end_from_either_end():
    """
    An edge tells the mark at, and the node across from, either of its ends.
    """
    graph = confounded()
    circle_end, arrow_end = graph.node("Z"), graph.node("X")
    edge = GraphEdge.partially_directed_edge(circle_end=circle_end, arrow_end=arrow_end)

    assert (
        edge.mark_at(node=circle_end),
        edge.mark_at(node=arrow_end),
        edge.other_end(node=circle_end),
        edge.other_end(node=arrow_end),
    ) == (EdgeMark.CIRCLE, EdgeMark.ARROW, arrow_end, circle_end)


def test_an_edge_asked_about_a_node_at_neither_end_raises():
    """
    An edge refuses to answer for a node at neither of its ends.
    """
    graph = confounded()
    edge = GraphEdge.directed_edge(graph.node("Z"), graph.node("X"))

    with pytest.raises(NotAnEndOfTheEdge):
        edge.mark_at(node=graph.node("Y"))
    with pytest.raises(NotAnEndOfTheEdge):
        edge.other_end(node=graph.node("Y"))


def test_an_edge_is_drawn_with_its_marks_between_its_nodes():
    """
    An edge is drawn as its nodes' names with its marks between them.
    """
    graph = confounded()
    z, x = graph.node("Z"), graph.node("X")

    assert [
        str(GraphEdge.partially_directed_edge(circle_end=z, arrow_end=x)),
        str(GraphEdge.bidirected_edge(first=z, second=x)),
        str(GraphEdge.directed_edge(z, x)),
    ] == ["Z o-> X", "Z <-> X", "Z --> X"]


def test_causal_learns_circle_code_reads_as_a_circle():
    """
    causal-learn's code for a circle reads as a circle mark.
    """
    assert AdjacencyMatrixMark(2).mark is EdgeMark.CIRCLE


# %% storing a graph


def test_a_causal_graphs_json_keeps_its_member_graph_limit():
    """
    A causal graph round-trips through JSON as a causal graph with its member limit.
    """
    graph = undirected_chain(2)
    graph.maximum_member_graphs = 7

    restored = from_json(to_json(graph))

    assert (
        type(restored),
        restored.maximum_member_graphs,
        to_json(graph)[CausalGraphJSONKey.MAXIMUM_MEMBER_GRAPHS],
        restored,
    ) == (CausalGraph, 7, 7, graph)


def test_graph_round_trips_through_json():
    graph = confounded()

    assert from_json(to_json(graph)) == graph


# %% the nodes of a relational graph


def disturbed_neighbour() -> GraphNode:
    """
    :return: The node of an attribute every neighbour of a scene has.
    """
    return GraphNode(
        variable=Continuous("Scene.neighbours.disturbed"),
        level=NodeLevel.PART_ATTRIBUTE,
        collection="neighbours",
    )


def test_variable_of_part_names_the_indexed_copy():
    assert (
        disturbed_neighbour().variable_of_part(3).name
        == "Scene.neighbours[3].disturbed"
    )


def test_part_index_of_reads_the_position_back_from_the_copy():
    node = disturbed_neighbour()

    assert node.part_index_of(node.variable_of_part(3).name) == 3


@pytest.mark.parametrize(
    "variable_name",
    [
        "Scene.neighbours.disturbed",
        "Scene.obstacles[3].disturbed",
        "Scene.neighbours[3].displacement",
        "Scene.neighbours[first].disturbed",
    ],
)
def test_part_index_of_is_none_for_a_variable_that_is_no_copy(variable_name):
    assert disturbed_neighbour().part_index_of(variable_name) is None


def test_part_index_of_is_none_for_a_node_of_no_part():
    node = GraphNode(
        variable=Continuous("Scene.friction"), level=NodeLevel.EXAMPLE_ATTRIBUTE
    )

    assert node.part_index_of("Scene.neighbours[3].disturbed") is None


def test_reference_of_a_copy_names_the_node_and_the_part():
    """
    The reference of a copy names the node and the part.
    """
    node = disturbed_neighbour()
    graph = CausalGraph.from_edges(nodes=[node], edges=[])

    reference = graph.reference_of(node.variable_of_part(3).name)

    assert (reference.node, reference.part_index) == (node, 3)
    assert reference.variable == node.variable_of_part(3)


# %% cells for the support of an adjustment


def test_continuous_columns_are_discretized_into_bin_count_levels():
    discretizer = Discretizer()
    rows = pd.DataFrame({"x": [position / 10 for position in range(100)]})

    codes = discretizer.fit_transform(rows)

    assert codes["x"].nunique() == discretizer.bin_count


def test_columns_with_few_values_keep_one_level_per_value_in_column_order():
    rows = pd.DataFrame(
        {
            "colour": ["red", "blue", "red", "green"] * 25,
            "x": [position / 10 for position in range(100)],
        }
    )

    codes = Discretizer().fit_transform(rows)

    assert list(codes.columns) == list(rows.columns)
    assert codes["colour"].tolist() == [2, 0, 2, 1] * 25


def test_a_column_of_enum_members_keeps_one_level_per_member():
    rows = pd.DataFrame({"kind": [Shape.ROUND, Shape.SQUARE] * 5})

    codes = Discretizer().fit_transform(rows)

    assert codes["kind"].nunique() == 2
    assert codes["kind"].tolist() == codes["kind"][:2].tolist() * 5


def test_the_discretizer_works_as_a_step_of_a_scikit_learn_pipeline():
    discretizer = Discretizer(bin_count=3)
    rows = pd.DataFrame({"x": [position / 10 for position in range(100)]})

    codes = make_pipeline(clone(discretizer)).fit_transform(rows)

    assert codes.equals(discretizer.fit_transform(rows))


def test_transforming_before_fitting_raises():
    with pytest.raises(NotFittedError):
        Discretizer().transform(pd.DataFrame({"x": [0.0]}))
