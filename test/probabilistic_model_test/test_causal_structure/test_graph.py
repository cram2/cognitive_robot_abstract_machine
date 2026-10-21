import pandas as pd
import pytest
from krrood.adapters.json_serializer import from_json, to_json
from random_events.variable import Continuous

from probabilistic_model.causal_structure.discretization import (
    Discretization,
)
from probabilistic_model.causal_structure.exceptions import (
    EquivalenceClassTooLarge,
    InconsistentGraph,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
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

    assert set(member.edges) == {("A", "C"), ("B", "C"), ("C", "D")}


def test_too_many_undirected_edges_raises():
    graph = undirected_chain(CausalGraph.maximum_undirected_edges + 1)

    with pytest.raises(EquivalenceClassTooLarge):
        graph.member_graphs()


def test_graph_without_member_graphs_raises():
    with pytest.raises(InconsistentGraph):
        without_member_graphs().member_graphs()


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
    discretization = Discretization()
    rows = pd.DataFrame({"x": [position / 10 for position in range(100)]})

    codes = discretization.apply(rows, ["x"])

    assert codes["x"].nunique() == discretization.bin_count
