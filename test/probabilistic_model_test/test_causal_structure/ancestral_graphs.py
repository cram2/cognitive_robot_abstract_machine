"""
Ancestral graphs whose adjustment sets are known, built for the structure tests: the
partial ancestral graphs causal-learn derives from a directed acyclic graph with hidden
nodes, and maximal ancestral graphs declared by hand.
"""

from __future__ import annotations

import contextlib
import io
import itertools
from dataclasses import dataclass, field

import numpy as np
from causallearn.graph.Dag import Dag
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode as CausalLearnNode
from causallearn.utils.DAG2PAG import dag2pag
from random_events.variable import Continuous

from probabilistic_model.causal_structure.ancestral_graph import AncestralGraph
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    EdgeMark,
    GraphEdge,
    GraphNode,
    NodeLevel,
)

# %% nodes by name


def node_named(name: str) -> GraphNode:
    """
    :param name: The name of a variable.
    :return: The example attribute node of a continuous variable of that name.
    """
    return GraphNode(variable=Continuous(name), level=NodeLevel.EXAMPLE_ATTRIBUTE)


# %% a true graph with hidden nodes


@dataclass(kw_only=True)
class TrueGraph:
    """
    A directed acyclic graph over observed and hidden nodes, the truth an ancestral
    graph over the observed nodes is derived from.
    """

    directed: list[tuple[str, str]]
    """
    The edges, each as its parent and its child.
    """

    hidden: list[str] = field(default_factory=list)
    """
    The nodes the data does not record.
    """

    isolated: list[str] = field(default_factory=list)
    """
    Nodes no edge touches.
    """

    @property
    def names(self) -> list[str]:
        """
        :return: Every node's name, sorted.
        """
        return sorted(
            {name for edge in self.directed for name in edge} | set(self.isolated)
        )

    @property
    def observed_names(self) -> list[str]:
        """
        :return: The names of the recorded nodes, sorted.
        """
        return [name for name in self.names if name not in self.hidden]

    def causal_graph(self) -> CausalGraph:
        """
        :return: The graph itself, hidden nodes included, as a causal graph.
        """
        return CausalGraph.from_edges(
            nodes=[node_named(name) for name in self.names],
            edges=[
                GraphEdge.directed_edge(node_named(parent), node_named(child))
                for parent, child in self.directed
            ],
        )

    def maximal_ancestral_graph(self) -> AncestralGraph:
        """
        :return: The maximal ancestral graph over the observed nodes: two are adjacent
            if their observed ancestors do not d-separate them, and an edge shows a
            tail at a node that is an ancestor of the other end, an arrow otherwise.
        """
        [dag] = self.causal_graph().member_graphs()
        observed = [
            dag_node for dag_node in dag.nodes() if dag_node.name in self.observed_names
        ]
        edges = []
        for first, second in itertools.combinations(observed, 2):
            ancestors = (dag.ancestors(first) | dag.ancestors(second)) & set(observed)
            if dag.is_directionally_separated(
                {first}, {second}, ancestors - {first, second}
            ):
                continue
            edges.append(
                GraphEdge(
                    first,
                    second,
                    EdgeMark.TAIL if first in dag.ancestors(second) else EdgeMark.ARROW,
                    EdgeMark.TAIL if second in dag.ancestors(first) else EdgeMark.ARROW,
                )
            )
        return AncestralGraph.from_edges(nodes=observed, edges=edges)

    def partial_ancestral_graph(self) -> AncestralGraph:
        """
        :return: The partial ancestral graph over the observed nodes that causal-learn
            derives from the graph.
        """
        causal_learn_node_of = {name: CausalLearnNode(name) for name in self.names}
        dag = Dag(list(causal_learn_node_of.values()))
        for parent, child in self.directed:
            dag.add_directed_edge(
                causal_learn_node_of[parent], causal_learn_node_of[child]
            )
        with contextlib.redirect_stdout(io.StringIO()):
            pag = dag2pag(
                dag, islatent=[causal_learn_node_of[name] for name in self.hidden]
            )
        return AncestralGraph.from_edges(
            nodes=[node_named(name) for name in self.observed_names],
            edges=[
                GraphEdge(
                    node_named(edge.get_node1().get_name()),
                    node_named(edge.get_node2().get_name()),
                    mark_of_endpoint(edge.get_endpoint1()),
                    mark_of_endpoint(edge.get_endpoint2()),
                )
                for edge in pag.get_graph_edges()
            ],
        )


def mark_of_endpoint(endpoint: Endpoint) -> EdgeMark:
    """
    :param endpoint: The end of an edge of a causal-learn graph.
    :return: The same mark on a :class:`GraphEdge`.
    """
    return EdgeMark[endpoint.name]


def random_true_graph(
    *, generator: np.random.Generator, edge_probability: float = 0.3
) -> TrueGraph:
    """
    :param generator: The source of randomness.
    :param edge_probability: How likely each pair of nodes is joined.
    :return: A directed acyclic graph over 5 to 8 nodes, its edges following the order
        of the nodes' names, with up to 2 of them hidden.
    """
    names = [f"N{index}" for index in range(int(generator.integers(5, 9)))]
    directed = [
        (first, second)
        for first, second in itertools.combinations(names, 2)
        if generator.random() < edge_probability
    ]
    hidden = [
        str(name)
        for name in generator.choice(
            names, size=int(generator.integers(0, 3)), replace=False
        )
    ]
    return TrueGraph(directed=directed, hidden=hidden, isolated=names)


# %% the oracle cases


def measured_confounder() -> TrueGraph:
    """
    :return: V → X, C → X, C → Y, X → Y, whose confounder C is recorded.
    """
    return TrueGraph(directed=[("V", "X"), ("C", "X"), ("C", "Y"), ("X", "Y")])


def hidden_cause_and_effect_confounder() -> TrueGraph:
    """
    :return: W → X, X → Y, L → X, L → Y, with L hidden.
    """
    return TrueGraph(
        directed=[("W", "X"), ("X", "Y"), ("L", "X"), ("L", "Y")], hidden=["L"]
    )


def hidden_confounder_of_covariates() -> TrueGraph:
    """
    :return: V → X, X → Y, C → X, C → Y, L → C, L → V, with L hidden.
    """
    return TrueGraph(
        directed=[
            ("V", "X"),
            ("X", "Y"),
            ("C", "X"),
            ("C", "Y"),
            ("L", "C"),
            ("L", "V"),
        ],
        hidden=["L"],
    )


def pure_hidden_confounding() -> TrueGraph:
    """
    :return: W → X, L → X, L → Y, with L hidden: X does not cause Y.
    """
    return TrueGraph(directed=[("W", "X"), ("L", "X"), ("L", "Y")], hidden=["L"])


def instrument_with_hidden_confounder() -> TrueGraph:
    """
    :return: Z → X, X → Y, L → X, L → Y, with L hidden.
    """
    return TrueGraph(
        directed=[("Z", "X"), ("X", "Y"), ("L", "X"), ("L", "Y")], hidden=["L"]
    )


def two_parents() -> TrueGraph:
    """
    :return: A → X, B → X, X → Y, with no hidden node.
    """
    return TrueGraph(directed=[("A", "X"), ("B", "X"), ("X", "Y")])


def declared_confounder_of_covariates() -> AncestralGraph:
    """
    :return: The maximal ancestral graph V → X, C → X, C → Y, X → Y, V ↔ C.
    """
    v, x, c, y = (node_named(name) for name in "VXCY")
    return AncestralGraph.from_edges(
        nodes=[v, x, c, y],
        edges=[
            GraphEdge.directed_edge(v, x),
            GraphEdge.directed_edge(c, x),
            GraphEdge.directed_edge(c, y),
            GraphEdge.directed_edge(x, y),
            GraphEdge.bidirected_edge(first=v, second=c),
        ],
    )


def bare_confounded() -> AncestralGraph:
    """
    :return: The maximal ancestral graph C → X, C → Y, X → Y, in which X → Y may still
        hide a confounder.
    """
    c, x, y = (node_named(name) for name in "CXY")
    return AncestralGraph.from_edges(
        nodes=[c, x, y],
        edges=[
            GraphEdge.directed_edge(c, x),
            GraphEdge.directed_edge(c, y),
            GraphEdge.directed_edge(x, y),
        ],
    )


def agrees_with(*, partial: AncestralGraph, maximal: AncestralGraph) -> bool:
    """
    :param partial: A partial ancestral graph.
    :param maximal: A maximal ancestral graph over the same nodes.
    :return: Whether the partial graph has the same adjacencies and shows the same mark
        wherever it shows no circle, as the graph standing for the maximal one must.
    """
    if len(partial.edges()) != len(maximal.edges()):
        return False
    for edge in partial.edges():
        other = maximal.edge_between(edge.first, edge.second)
        if other is None:
            return False
        for node in (edge.first, edge.second):
            mark = edge.mark_at(node=node)
            if mark is not EdgeMark.CIRCLE and mark is not other.mark_at(node=node):
                return False
    return True
