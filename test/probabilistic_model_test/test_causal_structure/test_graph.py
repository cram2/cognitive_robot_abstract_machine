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
    UnknownGraphNode,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    Collider,
    GraphEdge,
    GraphNode,
    NodeLevel,
)

from .graphs import (
    collider_with_an_undirected_child,
    confounded,
    mediated,
    undirected_chain,
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
        Collider(graph.node("A"), graph.node("C"), graph.node("B"))
    }


def test_too_many_undirected_edges_raises():
    graph = undirected_chain(CausalGraph.maximum_undirected_edges + 1)

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
    graph = mediated()

    assert not graph.adjacent(graph.node("X"), graph.node("Y"))


# %% storing a graph


def test_graph_round_trips_through_json():
    graph = confounded()

    assert from_json(to_json(graph)) == graph


# %% the nodes of a relational graph


def test_variable_of_part_names_the_indexed_copy():
    node = GraphNode(
        variable=Continuous("Scene.neighbours.disturbed"),
        level=NodeLevel.PART_ATTRIBUTE,
        collection="neighbours",
    )

    assert node.variable_of_part(3).name == "Scene.neighbours[3].disturbed"


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
