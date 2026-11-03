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
    GraphEdge,
    GraphNode,
    NodeLevel,
)

from .graphs import (
    collider_with_an_undirected_child,
    confounded,
    undirected_chain,
    without_member_graphs,
)

# %% the graphs an equivalence class stands for


def test_member_graphs_keep_v_structures():
    [member] = collider_with_an_undirected_child().member_graphs()

    assert set(member.edges()) == {("A", "C"), ("B", "C"), ("C", "D")}


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

    with pytest.raises(UnknownGraphNode):
        graph.add_edge(GraphEdge.directed("X", "W"))


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


def test_the_discretizer_works_as_a_step_of_a_scikit_learn_pipeline():
    discretizer = Discretizer(bin_count=3)
    rows = pd.DataFrame({"x": [position / 10 for position in range(100)]})

    codes = make_pipeline(clone(discretizer)).fit_transform(rows)

    assert codes.equals(discretizer.fit_transform(rows))


def test_transforming_before_fitting_raises():
    with pytest.raises(NotFittedError):
        Discretizer().transform(pd.DataFrame({"x": [0.0]}))
