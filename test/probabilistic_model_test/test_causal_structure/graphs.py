"""
Causal graphs whose adjustment sets are known from the definition of the adjustment
criterion, built for the structure tests.
"""

from __future__ import annotations

from random_events.variable import Continuous

from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    NodeLevel,
)

# %% building a graph from its edges


def graph_of(
    directed: list[tuple[str, str]],
    undirected: list[tuple[str, str]] | None = None,
    isolated: list[str] | None = None,
) -> CausalGraph:
    """
    :param directed: The directed edges, each as its parent and child.
    :param undirected: The undirected edges, each as its two ends.
    :param isolated: Nodes no edge touches.
    :return: The graph over every node the edges name, and the isolated ones.
    """
    undirected = undirected or []
    names = sorted(
        {name for edge in directed + undirected for name in edge} | set(isolated or [])
    )
    node_of = {
        name: GraphNode(variable=Continuous(name), level=NodeLevel.EXAMPLE_ATTRIBUTE)
        for name in names
    }
    return CausalGraph.from_edges(
        node_of.values(),
        [
            GraphEdge.directed(node_of[parent], node_of[child])
            for parent, child in directed
        ]
        + [
            GraphEdge.undirected(node_of[first], node_of[second])
            for first, second in undirected
        ],
    )


# %% fully directed graphs


def confounded() -> CausalGraph:
    """
    :return: Z → X, Z → Y, X → Y.
    """
    return graph_of([("Z", "X"), ("Z", "Y"), ("X", "Y")])


def mediated() -> CausalGraph:
    """
    :return: X → M → Y.
    """
    return graph_of([("X", "M"), ("M", "Y")])


def cause_with_a_child() -> CausalGraph:
    """
    :return: X → Y, X → C, where C is off every causal path from X to Y.
    """
    return graph_of([("X", "Y"), ("X", "C")])


def effect_with_a_child() -> CausalGraph:
    """
    :return: X → Y → D.
    """
    return graph_of([("X", "Y"), ("Y", "D")])


def m_bias() -> CausalGraph:
    """
    :return: X ← A → M ← B → Y, X → Y.
    """
    return graph_of([("A", "X"), ("A", "M"), ("B", "M"), ("B", "Y"), ("X", "Y")])


def proxied_confounder() -> CausalGraph:
    """
    :return: N → X, N → Y, N → E, X → Y, where E only reflects the confounder N.
    """
    return graph_of([("N", "X"), ("N", "Y"), ("N", "E"), ("X", "Y")])


def two_step_back_door() -> CausalGraph:
    """
    :return: X ← A → B → Y, X → Y, whose back-door path either of A and B blocks.
    """
    return graph_of([("A", "X"), ("A", "B"), ("B", "Y"), ("X", "Y")])


# %% graphs with undirected edges


def undirected_parent() -> CausalGraph:
    """
    :return: A — X, A → Y, X → Y.
    """
    return graph_of([("A", "Y"), ("X", "Y")], [("A", "X")])


def undirected_cause_and_effect() -> CausalGraph:
    """
    :return: X — Y.
    """
    return graph_of([], [("X", "Y")])


def collider_with_an_undirected_child() -> CausalGraph:
    """
    :return: A → C ← B, C — D, whose only member keeps C → D: D → C would add the
        colliders A → C ← D and B → C ← D.
    """
    return graph_of([("A", "C"), ("B", "C")], [("C", "D")])


def without_member_graphs() -> CausalGraph:
    """
    :return: A → B, C → D → A, B — C, which no orientation fits: B → C closes the cycle
        A → B → C → D → A, and C → B adds the collider A → B ← C.
    """
    return graph_of([("A", "B"), ("C", "D"), ("D", "A")], [("B", "C")])


def undirected_chain(length: int) -> CausalGraph:
    """
    :param length: How many undirected edges the chain has.
    :return: N0 — N1 — ... — N{length}.
    """
    return graph_of(
        [], [(f"N{position}", f"N{position + 1}") for position in range(length)]
    )
