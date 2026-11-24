"""
Tests for ancestral graphs: their possibly directed paths, visible edges, validity,
representative maximal ancestral graph and m-separation.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode as CausalLearnNode
from causallearn.search.ConstraintBased.FCI import defVisible
from krrood.adapters.json_serializer import from_json, to_json

from probabilistic_model.causal_structure.ancestral_graph import AncestralGraph
from probabilistic_model.causal_structure.graph import EdgeMark, GraphEdge, GraphNode

from .ancestral_graphs import (
    bare_confounded,
    declared_confounder_of_covariates,
    measured_confounder,
    node_named,
    pure_hidden_confounding,
    random_true_graph,
    two_parents,
)

# %% building graphs by hand


def ancestral_graph_of(*edges: tuple[str, EdgeMark, EdgeMark, str]) -> AncestralGraph:
    """
    :param edges: Each edge as its first node's name, the marks at its first and its
        second node, and its second node's name.
    :return: The ancestral graph over the nodes the edges name.
    """
    names = sorted({name for edge in edges for name in (edge[0], edge[3])})
    return AncestralGraph.from_edges(
        [node_named(name) for name in names],
        [
            GraphEdge(node_named(first), node_named(second), at_first, at_second)
            for first, at_first, at_second, second in edges
        ],
    )


def names_of(nodes: set[GraphNode]) -> set[str]:
    """
    :param nodes: Graph nodes.
    :return: Their names.
    """
    return {node.name for node in nodes}


# %% possibly directed paths


def test_possible_ancestors_follow_every_mark_but_an_arrow_at_the_earlier_node():
    """
    Possible ancestors follow every mark but an arrow at the earlier node.
    """
    graph = measured_confounder().partial_ancestral_graph()

    assert (
        names_of(graph.possible_ancestors(graph.node("Y"))),
        names_of(graph.possible_descendants(node=graph.node("X"))),
        names_of(graph.possible_ancestors(graph.node("V"))),
    ) == ({"C", "V", "X"}, {"Y"}, set())


def test_an_edge_into_the_cause_is_no_step_of_a_possibly_directed_path():
    """
    An edge into the cause is no step of a possibly directed path.
    """
    graph = pure_hidden_confounding().partial_ancestral_graph()

    assert (
        names_of(graph.possible_ancestors(graph.node("X"))),
        names_of(graph.possible_descendants(node=graph.node("X"))),
    ) == ({"W", "Y"}, set())


def test_a_path_whose_later_node_points_into_an_earlier_one_is_not_possibly_directed():
    """
    A path whose later node points into an earlier one is not possibly directed.
    """
    graph = ancestral_graph_of(
        ("A", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "B"),
        ("B", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "C"),
        ("C", EdgeMark.CIRCLE, EdgeMark.ARROW, "A"),
    )

    assert names_of(graph.possible_descendants(node=graph.node("A"))) == {"B"}


def simple_possibly_directed_paths(
    graph: AncestralGraph, start: GraphNode, end: GraphNode
) -> list[tuple[GraphNode, ...]]:
    """
    :param graph: An ancestral graph.
    :param start: A node.
    :param end: Another node.
    :return: Every simple path from the start to the end, found by trying every
        sequence of nodes, on which no edge between two of its nodes shows an arrow at
        the earlier one.
    """
    others = [node for node in graph.nodes() if node not in (start, end)]
    paths = []
    for length in range(len(others) + 1):
        for middle in itertools.permutations(others, length):
            path = (start, *middle, end)
            consecutive = all(
                graph.adjacent(first, second) for first, second in zip(path, path[1:])
            )
            if consecutive and not any(
                (edge := graph.edge_between(earlier, later)) is not None
                and edge.mark_at(node=earlier) is EdgeMark.ARROW
                for earlier, later in itertools.combinations(path, 2)
            ):
                paths.append(path)
    return paths


def test_possibly_causal_paths_equal_every_qualifying_sequence_of_nodes():
    """
    Possibly causal paths equal every qualifying sequence of nodes.
    """
    generator = np.random.default_rng(3)
    for _ in range(60):
        graph = random_true_graph(generator=generator).partial_ancestral_graph()
        for cause, effect in itertools.permutations(graph.nodes()[:6], 2):
            assert sorted(
                tuple(path)
                for path in graph.possibly_causal_paths(cause=cause, effect=effect)
            ) == sorted(simple_possibly_directed_paths(graph, cause, effect))


# %% visible edges


def test_an_edge_out_of_a_node_with_a_non_adjacent_parent_is_visible():
    """
    An edge out of a node with a non adjacent parent is visible.
    """
    for graph in (
        measured_confounder().partial_ancestral_graph(),
        two_parents().partial_ancestral_graph(),
    ):
        assert graph.is_visible(
            edge=GraphEdge.directed_edge(graph.node("X"), graph.node("Y"))
        )


def test_a_directed_edge_whose_every_other_parent_is_adjacent_is_not_visible():
    """
    A directed edge whose every other parent is adjacent is not visible.
    """
    graph = bare_confounded()

    assert not graph.is_visible(
        edge=GraphEdge.directed_edge(graph.node("X"), graph.node("Y"))
    )


def test_a_collider_path_of_parents_of_the_child_makes_an_edge_visible():
    """
    A collider path of parents of the child makes an edge visible.
    """
    graph = ancestral_graph_of(
        ("V", EdgeMark.TAIL, EdgeMark.ARROW, "Q"),
        ("Q", EdgeMark.ARROW, EdgeMark.ARROW, "X"),
        ("Q", EdgeMark.TAIL, EdgeMark.ARROW, "W"),
        ("X", EdgeMark.TAIL, EdgeMark.ARROW, "W"),
    )

    assert graph.is_visible(
        edge=GraphEdge.directed_edge(graph.node("X"), graph.node("W"))
    )


def test_a_collider_path_through_a_node_that_is_no_parent_of_the_child_does_not():
    """
    A collider path through a node that is no parent of the child does not.
    """
    graph = ancestral_graph_of(
        ("V", EdgeMark.TAIL, EdgeMark.ARROW, "Q"),
        ("Q", EdgeMark.ARROW, EdgeMark.ARROW, "X"),
        ("Q", EdgeMark.ARROW, EdgeMark.ARROW, "W"),
        ("X", EdgeMark.TAIL, EdgeMark.ARROW, "W"),
    )

    assert not graph.is_visible(
        edge=GraphEdge.directed_edge(graph.node("X"), graph.node("W"))
    )


def causal_learn_graph_of(
    graph: AncestralGraph,
) -> tuple[GeneralGraph, dict[GraphEdge, Edge]]:
    """
    :param graph: An ancestral graph.
    :return: The same graph as causal-learn holds it, and each of its edges there.
    """
    node_of = {node.name: CausalLearnNode(node.name) for node in graph.nodes()}
    translated = GeneralGraph(list(node_of.values()))
    edge_of = {}
    for edge in graph.edges():
        edge_of[edge] = Edge(
            node_of[edge.first.name],
            node_of[edge.second.name],
            Endpoint[edge.mark_at_first.name],
            Endpoint[edge.mark_at_second.name],
        )
        translated.add_edge(edge_of[edge])
    return translated, edge_of


def test_visibility_agrees_with_causal_learn_on_every_directed_edge():
    """
    Visibility agrees with causal learn on every directed edge.
    """
    generator = np.random.default_rng(4)
    for _ in range(150):
        graph = random_true_graph(generator=generator).partial_ancestral_graph()
        translated, edge_of = causal_learn_graph_of(graph)
        for edge in graph.edges():
            if edge.is_directed:
                assert graph.is_visible(edge=edge) == defVisible(
                    edge_of[edge], translated
                )


# %% whether the graph is a valid ancestral graph


def test_an_edge_with_a_tail_and_no_arrow_is_a_selection_bias_edge():
    """
    An edge with a tail and no arrow is a selection bias edge.
    """
    assert ancestral_graph_of(
        ("A", EdgeMark.TAIL, EdgeMark.CIRCLE, "B"),
        ("B", EdgeMark.TAIL, EdgeMark.ARROW, "C"),
    ).has_selection_bias_edge


def test_a_directed_cycle_is_not_ancestral():
    """
    A directed cycle is not ancestral.
    """
    assert not ancestral_graph_of(
        ("A", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
        ("B", EdgeMark.TAIL, EdgeMark.ARROW, "C"),
        ("C", EdgeMark.TAIL, EdgeMark.ARROW, "A"),
    ).is_ancestral


def test_a_bidirected_edge_between_an_ancestor_and_its_descendant_is_not_ancestral():
    """
    A bidirected edge between an ancestor and its descendant is not ancestral.
    """
    assert not ancestral_graph_of(
        ("A", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
        ("B", EdgeMark.TAIL, EdgeMark.ARROW, "C"),
        ("A", EdgeMark.ARROW, EdgeMark.ARROW, "C"),
    ).is_ancestral


def test_a_maximal_ancestral_graph_is_its_own_representative():
    """
    A maximal ancestral graph is its own representative.
    """
    graph = declared_confounder_of_covariates()

    assert graph.representative_maximal_ancestral_graph == graph


def test_the_representative_turns_circles_into_tails_without_new_colliders():
    """
    The representative turns circles into tails without new colliders.
    """
    graph = measured_confounder().partial_ancestral_graph()
    v, x, c, y = (graph.node(name) for name in "VXCY")

    assert set(graph.representative_maximal_ancestral_graph.edges()) == {
        GraphEdge.directed_edge(v, x),
        GraphEdge.directed_edge(c, x),
        GraphEdge.directed_edge(c, y),
        GraphEdge.directed_edge(x, y),
    }


def test_the_representative_of_a_nondirected_triangle_is_a_directed_acyclic_graph():
    """
    The representative of a nondirected triangle is a directed acyclic graph.
    """
    graph = ancestral_graph_of(
        ("A", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "B"),
        ("B", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "C"),
        ("A", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "C"),
    )

    representative = graph.representative_maximal_ancestral_graph

    assert (
        all(edge.is_directed for edge in representative.edges()),
        representative.is_ancestral,
        len(representative.edges()),
    ) == (True, True, 3)


def test_a_nondirected_cycle_without_a_chord_has_no_representative():
    """
    A nondirected cycle without a chord has no representative.
    """
    graph = ancestral_graph_of(
        ("A", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "B"),
        ("B", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "C"),
        ("C", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "D"),
        ("D", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "A"),
    )

    assert graph.representative_maximal_ancestral_graph is None


# %% m-separation


@pytest.mark.parametrize(
    "edges, given, separated",
    [
        (
            (
                ("A", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
                ("B", EdgeMark.TAIL, EdgeMark.ARROW, "C"),
            ),
            set(),
            False,
        ),
        (
            (
                ("A", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
                ("B", EdgeMark.TAIL, EdgeMark.ARROW, "C"),
            ),
            {"B"},
            True,
        ),
        (
            (
                ("A", EdgeMark.ARROW, EdgeMark.ARROW, "B"),
                ("C", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
            ),
            set(),
            True,
        ),
        (
            (
                ("A", EdgeMark.ARROW, EdgeMark.ARROW, "B"),
                ("C", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
            ),
            {"B"},
            False,
        ),
        (
            (
                ("A", EdgeMark.ARROW, EdgeMark.ARROW, "B"),
                ("C", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
                ("B", EdgeMark.TAIL, EdgeMark.ARROW, "D"),
            ),
            {"D"},
            False,
        ),
        (
            (
                ("A", EdgeMark.ARROW, EdgeMark.ARROW, "B"),
                ("B", EdgeMark.ARROW, EdgeMark.ARROW, "C"),
            ),
            set(),
            True,
        ),
    ],
)
def test_m_separation_blocks_at_given_non_colliders_and_unopened_colliders(
    edges: tuple, given: set[str], separated: bool
):
    """
    M separation blocks at given non colliders and unopened colliders.
    """
    graph = ancestral_graph_of(*edges)

    assert (
        graph.is_m_separated(
            first=graph.node("A"),
            second=graph.node("C"),
            given=frozenset(graph.node(name) for name in given),
        )
        == separated
    )


def test_m_separation_ends_on_a_diamond_beside_a_separate_edge():
    """
    M separation ends on a diamond beside a separate edge.
    """
    graph = ancestral_graph_of(
        ("A", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
        ("A", EdgeMark.TAIL, EdgeMark.ARROW, "C"),
        ("B", EdgeMark.TAIL, EdgeMark.ARROW, "D"),
        ("C", EdgeMark.TAIL, EdgeMark.ARROW, "D"),
        ("F", EdgeMark.TAIL, EdgeMark.ARROW, "G"),
    )
    started = time.monotonic()

    separated = graph.is_m_separated(
        first=graph.node("B"),
        second=graph.node("G"),
        given=frozenset({graph.node("D")}),
    )

    assert (separated, time.monotonic() - started < 1.0) == (True, True)


def test_removing_the_causal_first_edges_leaves_the_other_edges():
    """
    Removing the causal first edges leaves the other edges.
    """
    graph = declared_confounder_of_covariates()
    x, y = graph.node("X"), graph.node("Y")

    assert set(graph.without_causal_first_edges(cause=x, effect=y).edges()) == set(
        graph.edges()
    ) - {GraphEdge.directed_edge(x, y)}


# %% storing a graph


def test_an_ancestral_graph_round_trips_through_json_with_every_kind_of_edge():
    """
    An ancestral graph round trips through json with every kind of edge.
    """
    graph = ancestral_graph_of(
        ("A", EdgeMark.ARROW, EdgeMark.ARROW, "B"),
        ("C", EdgeMark.CIRCLE, EdgeMark.ARROW, "B"),
        ("C", EdgeMark.CIRCLE, EdgeMark.CIRCLE, "D"),
        ("D", EdgeMark.TAIL, EdgeMark.ARROW, "B"),
    )

    restored = from_json(to_json(graph))

    assert (type(restored), restored) == (AncestralGraph, graph)
