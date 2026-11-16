import itertools

import networkx
import numpy as np
import pytest

from probabilistic_model.causal_structure.graph import MemberGraph

# %% building graphs


def graph_of(
    edges: list[tuple[str, str]], isolated: tuple[str, ...] = ()
) -> MemberGraph:
    """
    :param edges: The edges, each as its parent and its child.
    :param isolated: Nodes without edges.
    :return: The graph.
    """
    graph = MemberGraph()
    graph.add_nodes_from(isolated)
    graph.add_edges_from(edges)
    return graph


# %% d-separation


@pytest.mark.parametrize(
    "edges, given, separated",
    [
        ([("X", "M"), ("M", "Y")], set(), False),
        ([("X", "M"), ("M", "Y")], {"M"}, True),
        ([("M", "X"), ("M", "Y")], set(), False),
        ([("M", "X"), ("M", "Y")], {"M"}, True),
        ([("X", "M"), ("Y", "M")], set(), True),
        ([("X", "M"), ("Y", "M")], {"M"}, False),
        ([("X", "M"), ("Y", "M"), ("M", "D")], {"D"}, False),
    ],
    ids=[
        "chain",
        "chain given its middle",
        "fork",
        "fork given its root",
        "collider",
        "collider given itself",
        "collider given its descendant",
    ],
)
def test_d_separation_of_three_nodes(edges, given, separated):
    assert graph_of(edges).is_directionally_separated({"X"}, {"Y"}, given) is separated


def test_d_separation_agrees_with_networkx_on_random_graphs():
    generator = np.random.default_rng(0)
    names = [f"N{index}" for index in range(7)]
    for _ in range(30):
        edges = [
            (parent, child)
            for parent, child in itertools.combinations(names, 2)
            if generator.random() < 0.3
        ]
        graph = graph_of(edges, isolated=tuple(names))
        reference = networkx.DiGraph(edges)
        reference.add_nodes_from(names)
        for first, second in itertools.combinations(names, 2):
            others = [name for name in names if name not in (first, second)]
            given = set(generator.choice(others, size=2, replace=False))
            assert graph.is_directionally_separated({first}, {second}, given) == (
                networkx.is_d_separator(reference, {first}, {second}, given)
            )


# %% the graph's structure


def test_ancestors_and_descendants_follow_directed_paths():
    graph = graph_of([("A", "B"), ("B", "C"), ("D", "C")])

    assert graph.ancestors("C") == {"A", "B", "D"}
    assert graph.descendants("A") == {"B", "C"}


def test_a_cycle_is_found():
    graph = graph_of([("A", "B"), ("B", "C"), ("C", "A")])

    assert not graph.is_acyclic
    assert set(graph.cycle()) == {"A", "B", "C"}


def test_removing_edges_leaves_the_original_graph_unchanged():
    graph = graph_of([("A", "B"), ("B", "C")])

    without = graph.without_edges([("A", "B")])

    assert without.edges() == [("B", "C")]
    assert set(graph.edges()) == {("A", "B"), ("B", "C")}
