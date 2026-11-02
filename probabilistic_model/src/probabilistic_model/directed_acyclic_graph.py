"""
Directed acyclic graphs over the variables of a model, as a Bayesian network and the
graphs a causal graph stands for have them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import rustworkx as rx
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Generic, Iterable, Iterator, TypeVar

NodeType = TypeVar("NodeType")

# %% the graph


@dataclass
class DirectedAcyclicGraph(Generic[NodeType], SubClassSafeGeneric, ABC):
    """
    A directed acyclic graph whose nodes are the objects it holds.
    """

    graph: rx.PyDAG = field(default_factory=lambda: rx.PyDAG(multigraph=False))
    """
    The graph, whose node data are the nodes.
    """

    def __len__(self) -> int:
        return len(self.graph)

    def __iter__(self) -> Iterator[NodeType]:
        return iter(self.graph.nodes())

    @abstractmethod
    def index_of(self, node: NodeType) -> int:
        """
        :param node: A node of the graph.
        :return: Its index in :attr:`graph`.
        """

    @abstractmethod
    def add_node(self, node: NodeType) -> None:
        """
        Add a node, unless the graph already has it.

        :param node: The node.
        """

    def add_nodes_from(self, nodes: Iterable[NodeType]) -> None:
        """
        :param nodes: The nodes to add, as :meth:`add_node` adds each.
        """
        for node in nodes:
            self.add_node(node)

    def add_edge(self, parent: NodeType, child: NodeType) -> None:
        """
        Add an edge, and its nodes if the graph lacks them.

        :param parent: The node the edge leaves.
        :param child: The node it enters.
        """
        self.add_node(parent)
        self.add_node(child)
        self.graph.add_edge(self.index_of(parent), self.index_of(child), None)

    def add_edges_from(self, edges: Iterable[tuple[NodeType, NodeType]]) -> None:
        """
        :param edges: The edges to add, as :meth:`add_edge` adds each.
        """
        for parent, child in edges:
            self.add_edge(parent, child)

    def nodes(self) -> list[NodeType]:
        """
        :return: The nodes.
        """
        return self.graph.nodes()

    def edges(self) -> list[tuple[NodeType, NodeType]]:
        """
        :return: The edges, each as its parent and its child.
        """
        return [
            (self.graph[parent], self.graph[child])
            for parent, child in self.graph.edge_list()
        ]

    def successors(self, node: NodeType) -> list[NodeType]:
        """
        :param node: A node.
        :return: Its children.
        """
        return self.graph.successors(self.index_of(node))

    def predecessors(self, node: NodeType) -> list[NodeType]:
        """
        :param node: A node.
        :return: Its parents.
        """
        return self.graph.predecessors(self.index_of(node))

    def descendants(self, node: NodeType) -> set[NodeType]:
        """
        :param node: A node.
        :return: The nodes a directed path leads to from it.
        """
        return {
            self.graph[index]
            for index in rx.descendants(self.graph, self.index_of(node))
        }

    def ancestors(self, node: NodeType) -> set[NodeType]:
        """
        :param node: A node.
        :return: The nodes a directed path leads from to it.
        """
        return {
            self.graph[index] for index in rx.ancestors(self.graph, self.index_of(node))
        }

    def in_degree(self, node: NodeType) -> int:
        """
        :param node: A node.
        :return: How many parents it has.
        """
        return self.graph.in_degree(self.index_of(node))

    def has_edge(self, parent: NodeType, child: NodeType) -> bool:
        """
        :param parent: A node.
        :param child: Another node.
        :return: Whether an edge leads from the first to the second.
        """
        return self.graph.has_edge(self.index_of(parent), self.index_of(child))

    @property
    def is_acyclic(self) -> bool:
        """
        :return: Whether no directed path leads from a node back to itself.
        """
        return rx.is_directed_acyclic_graph(self.graph)

    def cycle(self) -> list[NodeType]:
        """
        :return: The nodes of one directed cycle, in its order; none if there is none.
        """
        return [self.graph[parent] for parent, _ in rx.digraph_find_cycle(self.graph)]

    def is_d_separated(
        self, first: set[NodeType], second: set[NodeType], given: set[NodeType]
    ) -> bool:
        """
        :param first: Nodes.
        :param second: Other nodes, none of them in the first.
        :param given: The nodes conditioned on, none of them in the first or second.
        :return: Whether every path between the first and the second nodes is blocked
            by the given nodes (Koller and Friedman, 2009, algorithm 3.1).
        """
        opens_colliders = set(given).union(*(self.ancestors(node) for node in given))
        reached: set[NodeType] = set()
        visited: set[tuple[NodeType, bool]] = set()
        to_visit = [(node, True) for node in first]
        while to_visit:
            node, from_a_child = to_visit.pop()
            if (node, from_a_child) in visited:
                continue
            visited.add((node, from_a_child))
            if node not in given:
                reached.add(node)
            if from_a_child and node not in given:
                to_visit.extend((parent, True) for parent in self.predecessors(node))
                to_visit.extend((child, False) for child in self.successors(node))
            elif not from_a_child:
                if node not in given:
                    to_visit.extend((child, False) for child in self.successors(node))
                if node in opens_colliders:
                    to_visit.extend(
                        (parent, True) for parent in self.predecessors(node)
                    )
        return not reached & set(second)
